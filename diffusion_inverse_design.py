"""
多层薄膜光学滤波器扩散模型逆向设计
基于条件扩散模型实现光谱到结构的逆向设计

主要功能:
1. 条件扩散模型架构 (Conditional Diffusion Model)
2. 混合数据类型处理 (材料离散 + 厚度连续)
3. 物理约束集成 (TMM物理损失)
4. 高质量采样策略 (DDPM/DDIM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter  # 显式导入Parameter
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import argparse
from tqdm import tqdm
import json
from diffusion_train_config import get_config, DiffusionTrainConfig
from copy import deepcopy

# 安全打印函数，避免Windows编码问题
def safe_print(text):
    """安全打印函数，处理Windows中文编码问题"""
    try:
        print(text)
    except UnicodeEncodeError:
        # 如果遇到编码问题，使用ASCII安全的版本
        safe_text = text.encode('ascii', 'ignore').decode('ascii')
        print(f"[编码警告] {safe_text}")

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

def get_device(gpu_id=0):
    """获取可用设备"""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        safe_print(f"使用GPU设备: {device} ({torch.cuda.get_device_name(gpu_id)})")
    else:
        device = torch.device("cpu")
        safe_print("使用CPU设备")
    return device

# ============================================================================
# 数据集处理
# ============================================================================

class MultilayerDiffusionDataset(Dataset):
    """多层薄膜扩散模型数据集"""
    
    def __init__(self, data_path, mode='train', train_ratio=0.8):
        # 加载原始数据
        data = np.load(data_path, allow_pickle=True)
        self.structures = data['structures']
        self.wavelengths = data['wavelengths']
        self.transmission = data['transmission']
        self.reflection = data['reflection']
        
        # 数据集划分
        num_samples = self.structures.shape[0]
        train_size = int(num_samples * train_ratio)
        
        if mode == 'train':
            self.structures = self.structures[:train_size]
            self.transmission = self.transmission[:train_size]
            self.reflection = self.reflection[:train_size]
        else:
            self.structures = self.structures[train_size:]
            self.transmission = self.transmission[train_size:]
            self.reflection = self.reflection[train_size:]
        
        # 建立材料词典
        self.build_material_vocab()
        
        # 预处理数据
        self.preprocess_data()
        
        safe_print(f"[数据] {mode}数据集加载完成: {len(self)} 个样本")
        safe_print(f"[数据] 材料数量: {len(self.material_to_idx)}")
        safe_print(f"[数据] 光谱维度: {self.spectrum_dim}")
        
    def build_material_vocab(self):
        """建立材料词典"""
        unique_materials = set()
        
        for structure in self.structures:
            for layer in structure:
                material = layer[0]
                if isinstance(material, str) and material != '':
                    unique_materials.add(material)
        
        self.materials = sorted(list(unique_materials))
        self.material_to_idx = {mat: idx for idx, mat in enumerate(self.materials)}
        self.idx_to_material = {idx: mat for mat, idx in self.material_to_idx.items()}
        
        # 添加填充token
        self.pad_token_idx = len(self.materials)
        self.vocab_size = len(self.materials) + 1
        
    def preprocess_data(self):
        """预处理结构数据"""
        self.processed_structures = []
        self.layer_masks = []  # 记录有效层的mask
        
        max_layers = 10
        
        for structure in self.structures:
            # 提取有效层
            valid_layers = []
            for layer in structure:
                material = layer[0]
                thickness = float(layer[1]) if layer[1] != '' else 0.0
                if isinstance(material, str) and material != '' and thickness > 0:
                    material_idx = self.material_to_idx[material]
                    valid_layers.append([material_idx, thickness])
            
            # 创建mask
            layer_mask = [1] * len(valid_layers) + [0] * (max_layers - len(valid_layers))
            
            # 填充到最大层数
            while len(valid_layers) < max_layers:
                valid_layers.append([self.pad_token_idx, 0.0])
            
            self.processed_structures.append(valid_layers)
            self.layer_masks.append(layer_mask)
        
        # 转换为tensor
        self.processed_structures = torch.tensor(self.processed_structures, dtype=torch.float32)
        self.layer_masks = torch.tensor(self.layer_masks, dtype=torch.bool)
        self.transmission = torch.tensor(self.transmission, dtype=torch.float32)
        self.reflection = torch.tensor(self.reflection, dtype=torch.float32)
        
        # 合并透射和反射谱
        self.spectra = torch.stack([self.transmission, self.reflection], dim=1)
        self.spectrum_dim = self.spectra.shape[-1]
        
    def __len__(self):
        return len(self.processed_structures)
    
    def __getitem__(self, idx):
        return {
            'structure': self.processed_structures[idx],
            'spectrum': self.spectra[idx],
            'layer_mask': self.layer_masks[idx],
        }

# ============================================================================
# 噪声调度器
# ============================================================================

def cosine_beta_schedule(timesteps, s=0.008):
    """余弦噪声调度 (推荐)"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class NoiseScheduler:
    """扩散模型噪声调度器"""
    
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.betas = cosine_beta_schedule(timesteps)
        
        # 预计算扩散参数
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def add_noise(self, x_start, noise, timesteps):
        """添加噪声 (前向过程)"""
        # 确保所有tensor都在同一设备上
        device = x_start.device
        if self.sqrt_alphas_cumprod.device != device:
            self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        if self.sqrt_one_minus_alphas_cumprod.device != device:
            self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].reshape(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# ============================================================================
# 混合数据类型处理
# ============================================================================

class MixedTypeProcessor:
    """处理混合数据类型 (离散材料 + 连续厚度)"""
    
    def __init__(self, vocab_size, max_layers, thickness_range=(10, 500)):
        self.vocab_size = vocab_size
        self.max_layers = max_layers
        self.thickness_min, self.thickness_max = thickness_range
        
    def normalize_structure(self, structure):
        """标准化结构数据"""
        materials = structure[:, :, 0].long()
        thicknesses = structure[:, :, 1]
        
        # 材料转为one-hot编码
        materials_onehot = F.one_hot(materials, num_classes=self.vocab_size).float()
        
        # 厚度标准化到[-1, 1]
        thicknesses_norm = 2 * (thicknesses - self.thickness_min) / (self.thickness_max - self.thickness_min) - 1
        thicknesses_norm = torch.clamp(thicknesses_norm, -1, 1)
        
        return materials_onehot, thicknesses_norm.unsqueeze(-1)
    
    def denormalize_structure(self, materials_onehot, thicknesses_norm):
        """反标准化结构数据"""
        materials = torch.argmax(materials_onehot, dim=-1)
        thicknesses = (thicknesses_norm.squeeze(-1) + 1) / 2 * (self.thickness_max - self.thickness_min) + self.thickness_min
        thicknesses = torch.clamp(thicknesses, self.thickness_min, self.thickness_max)
        
        structure = torch.stack([materials.float(), thicknesses], dim=-1)
        return structure

# ============================================================================
# 指数移动平均 (EMA)
# ============================================================================

class EMAHelper:
    """指数移动平均助手类"""
    
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 初始化shadow参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """更新EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """应用EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """恢复原始参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# ============================================================================
# 扩散模型网络架构
# ============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码 (时间步编码)"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SpectrumEncoder(nn.Module):
    """光谱编码器"""
    
    def __init__(self, spectrum_dim=71, hidden_dim=256):
        super().__init__()
        
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, hidden_dim)
        
    def forward(self, spectra):
        x = F.relu(self.conv1(spectra))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        
        return x

class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class DiffusionUNet(nn.Module):
    """扩散模型U-Net网络"""
    
    def __init__(self, vocab_size, max_layers=10, spectrum_dim=71, hidden_dim=256, 
                 num_layers=6, num_heads=8):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_layers = max_layers
        self.hidden_dim = hidden_dim
        
        # 时间步编码
        self.time_embedding = SinusoidalPositionEmbeddings(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 光谱编码器
        self.spectrum_encoder = SpectrumEncoder(spectrum_dim, hidden_dim)
        
        # 结构编码
        self.material_embedding = nn.Linear(vocab_size, hidden_dim // 2)
        self.thickness_embedding = nn.Linear(1, hidden_dim // 2)
        self.input_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 位置编码
        self.pos_embedding = Parameter(torch.randn(1, max_layers, hidden_dim))
        
        # Transformer层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads) 
            for _ in range(num_layers)
        ])
        
        # 输出头
        self.material_head = nn.Linear(hidden_dim, vocab_size)
        self.thickness_head = nn.Linear(hidden_dim, 1)
        
        # 条件融合
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.condition_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, materials_noisy, thicknesses_noisy, timesteps, spectra, layer_mask=None, drop_condition=False):
        """
        前向传播 - 支持Classifier-Free Guidance
        Args:
            materials_noisy: [batch_size, max_layers, vocab_size] 噪声材料
            thicknesses_noisy: [batch_size, max_layers, 1] 噪声厚度
            timesteps: [batch_size] 时间步
            spectra: [batch_size, 2, spectrum_dim] 条件光谱
            layer_mask: [batch_size, max_layers] 层mask
            drop_condition: bool, 是否丢弃条件(用于CFG训练)
        """
        batch_size = materials_noisy.size(0)
        
        # 时间步编码
        time_emb = self.time_embedding(timesteps)
        time_emb = self.time_mlp(time_emb)
        
        # 光谱条件编码 - 支持条件丢弃
        if drop_condition:
            # 无条件情况：使用零向量
            spectrum_emb = torch.zeros(batch_size, self.hidden_dim, device=spectra.device)
        else:
            spectrum_emb = self.spectrum_encoder(spectra)
        
        # ... existing code ...
        
        # 结构编码
        mat_emb = self.material_embedding(materials_noisy)
        thick_emb = self.thickness_embedding(thicknesses_noisy)
        
        # 合并材料和厚度特征
        struct_emb = torch.cat([mat_emb, thick_emb], dim=-1)
        struct_emb = self.input_projection(struct_emb)
        
        # 添加位置编码和时间信息
        struct_emb = struct_emb + self.pos_embedding
        time_emb_expanded = time_emb.unsqueeze(1).expand(-1, self.max_layers, -1)
        struct_emb = struct_emb + time_emb_expanded
        
        # Transformer处理
        x = struct_emb
        for transformer in self.transformer_blocks:
            x = transformer(x, mask=~layer_mask if layer_mask is not None else None)
        
        # 条件交叉注意力
        spectrum_emb_expanded = spectrum_emb.unsqueeze(1)
        cross_attn_out, _ = self.cross_attention(x, spectrum_emb_expanded, spectrum_emb_expanded)
        x = self.condition_norm(x + cross_attn_out)
        
        # 预测噪声
        material_noise = self.material_head(x)
        thickness_noise = self.thickness_head(x)
        
        return material_noise, thickness_noise

# ============================================================================
# 损失函数
# ============================================================================

class DiffusionLoss(nn.Module):
    """扩散模型损失函数"""
    
    def __init__(self, material_weight=1.0, thickness_weight=1.0):
        super().__init__()
        self.material_weight = material_weight
        self.thickness_weight = thickness_weight
        
    def forward(self, pred_material_noise, pred_thickness_noise, 
                true_material_noise, true_thickness_noise, layer_mask=None):
        """计算扩散损失"""
        # 材料噪声损失
        material_loss = F.mse_loss(pred_material_noise, true_material_noise)
        
        # 厚度噪声损失
        thickness_loss = F.mse_loss(pred_thickness_noise, true_thickness_noise)
        
        # 如果有层数掉码，只计算有效层的损失
        if layer_mask is not None:
            # 扩展mask维度匹配厚度损失
            mask_expanded = layer_mask.unsqueeze(-1).float()
            thickness_loss = F.mse_loss(
                pred_thickness_noise * mask_expanded, 
                true_thickness_noise * mask_expanded
            )
            
            # 材料损失也应用mask
            mask_material = layer_mask.unsqueeze(-1).expand_as(pred_material_noise).float()
            material_loss = F.mse_loss(
                pred_material_noise * mask_material,
                true_material_noise * mask_material
            )
        
        # 总损失
        total_loss = (self.material_weight * material_loss + 
                      self.thickness_weight * thickness_loss)
        
        return total_loss, {
            'material_loss': material_loss.item(),
            'thickness_loss': thickness_loss.item(),
            'total_loss': total_loss.item()
        }

# ============================================================================
# 训练函数
# ============================================================================

def train_diffusion_model(model, dataloader, num_epochs, device, 
                         scheduler, mixed_processor, lr=1e-4, save_interval=10, 
                         cfg_dropout_prob=0.1, config=None, mode='demo'):
    """训练扩散模型 - 支持Classifier-Free Guidance和优化配置"""
    
    # 使用配置文件参数
    if config is None:
        config = DiffusionTrainConfig.DEMO_CONFIG
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.get('lr', lr), 
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # 使用配置中的损失权重
    loss_weights = DiffusionTrainConfig.LOSS_WEIGHTS
    criterion = DiffusionLoss(
        material_weight=loss_weights['material_weight'], 
        thickness_weight=loss_weights['thickness_weight']
    )
    
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # EMA支持
    ema_helper = None
    if config.get('use_ema', False):
        ema_helper = EMAHelper(model, decay=config.get('ema_decay', 0.9999))
        safe_print(f"[EMA] 启用指数移动平均，衰减率: {config.get('ema_decay', 0.9999)}")
    
    # Warmup学习率调度 - 兼容旧版本PyTorch
    warmup_epochs = config.get('warmup_epochs', 5)
    # 使用手动实现的warmup，因为旧版本PyTorch没有LinearLR
    warmup_factor = 0.1
    warmup_steps = []
    
    history = {
        'epoch_losses': [],
        'material_losses': [],
        'thickness_losses': []
    }
    
    safe_print(f"[训练] 开始训练扩散模型(支持CFG)...")
    safe_print(f"   训练模式: {mode.upper()}")
    safe_print(f"   批次大小: {config['batch_size']}")
    safe_print(f"   训练轮数: {num_epochs}")
    safe_print(f"   学习率: {config.get('lr', lr)}")
    safe_print(f"   CFG丢弃概率: {cfg_dropout_prob}")
    safe_print(f"   梯度裁剪: {config.get('gradient_clip', 1.0)}")
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_material_losses = []
        epoch_thickness_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in pbar:
            structure = batch['structure'].to(device)
            spectrum = batch['spectrum'].to(device)
            layer_mask = batch['layer_mask'].to(device)
            
            batch_size = structure.size(0)
            
            # 标准化结构数据
            materials_onehot, thicknesses_norm = mixed_processor.normalize_structure(structure)
            
            # 随机采样时间步
            timesteps = torch.randint(0, scheduler.timesteps, (batch_size,), device=device)
            
            # 生成噪声
            noise_materials = torch.randn_like(materials_onehot)
            noise_thicknesses = torch.randn_like(thicknesses_norm)
            
            # 添加噪声
            materials_noisy = scheduler.add_noise(materials_onehot, noise_materials, timesteps)
            thicknesses_noisy = scheduler.add_noise(thicknesses_norm, noise_thicknesses, timesteps)
            
            # CFG训练: 随机丢弃条件
            drop_condition = torch.rand(1).item() < cfg_dropout_prob
            
            # 前向传播
            pred_material_noise, pred_thickness_noise = model(
                materials_noisy, thicknesses_noisy, timesteps, spectrum, layer_mask, drop_condition
            )
            
            # 计算损失
            loss, loss_dict = criterion(
                pred_material_noise, pred_thickness_noise,
                noise_materials, noise_thicknesses, 
                layer_mask
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            gradient_clip = config.get('gradient_clip', 1.0)
            # 使用PyRight建议的导入方式来避免类型检查警告
            from torch.nn.utils.clip_grad import clip_grad_norm_
            clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            
            # 更新EMA
            if ema_helper is not None:
                ema_helper.update()
            
            # 记录损失
            epoch_losses.append(loss.item())
            epoch_material_losses.append(loss_dict['material_loss'])
            epoch_thickness_losses.append(loss_dict['thickness_loss'])
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Mat': f"{loss_dict['material_loss']:.4f}",
                'Thick': f"{loss_dict['thickness_loss']:.4f}"
            })
        
        # 更新学习率 - 手动实现warmup
        if epoch < warmup_epochs:
            # 手动设置warmup学习率
            warmup_lr = config.get('lr', lr) * (warmup_factor + (1 - warmup_factor) * epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        else:
            lr_scheduler.step()
            
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录epoch统计
        avg_loss = np.mean(epoch_losses)
        avg_material_loss = np.mean(epoch_material_losses)
        avg_thickness_loss = np.mean(epoch_thickness_losses)
        
        history['epoch_losses'].append(avg_loss)
        history['material_losses'].append(avg_material_loss)
        history['thickness_losses'].append(avg_thickness_loss)
        
        safe_print(f"\nEpoch {epoch+1} 总结: Loss={avg_loss:.4f}, Mat={avg_material_loss:.4f}, Thick={avg_thickness_loss:.4f}, LR={current_lr:.2e}")
        
        # 定期保存
        if (epoch + 1) % save_interval == 0:
            # 保存普通模型
            torch.save(model.state_dict(), f"diffusion_model_epoch_{epoch+1}.pth")
            
            # 保存EMA模型
            if ema_helper is not None:
                ema_helper.apply_shadow()
                torch.save(model.state_dict(), f"diffusion_model_ema_epoch_{epoch+1}.pth")
                ema_helper.restore()
            
            safe_print(f"[保存] 已保存 Epoch {epoch+1} 模型")
    
    # 保存最终模型
    torch.save(model.state_dict(), "diffusion_model_final.pth")
    
    # 保存最终EMA模型
    if ema_helper is not None:
        ema_helper.apply_shadow()
        torch.save(model.state_dict(), "diffusion_model_final_ema.pth")
        ema_helper.restore()
        safe_print(f"[完成] 训练完成！已保存EMA模型")
    else:
        safe_print(f"[完成] 训练完成！")
    
    return history

@torch.no_grad()
def sample_structures_with_cfg(model, target_spectra, mixed_processor, scheduler, 
                              device, guidance_scale=6.0, num_samples=1, num_inference_steps=100):
    """使用Classifier-Free Guidance采样生成结构"""
    
    model.eval()
    
    batch_size = target_spectra.size(0) if target_spectra.dim() > 2 else 1
    if target_spectra.dim() == 2:
        target_spectra = target_spectra.unsqueeze(0)
    
    # 初始化随机噪声
    materials_shape = (batch_size, model.max_layers, model.vocab_size)
    thicknesses_shape = (batch_size, model.max_layers, 1)
    
    materials_noisy = torch.randn(materials_shape, device=device)
    thicknesses_noisy = torch.randn(thicknesses_shape, device=device)
    
    # CFG采样过程
    timesteps = torch.linspace(scheduler.timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
    
    for i, t in enumerate(tqdm(timesteps, desc="CFG采样")):
        timestep_batch = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)
        
        # 有条件预测
        pred_material_cond, pred_thickness_cond = model(
            materials_noisy, thicknesses_noisy, timestep_batch, target_spectra, drop_condition=False
        )
        
        # 无条件预测
        pred_material_uncond, pred_thickness_uncond = model(
            materials_noisy, thicknesses_noisy, timestep_batch, target_spectra, drop_condition=True
        )
        
        # Classifier-Free Guidance
        pred_material_noise = pred_material_uncond + guidance_scale * (pred_material_cond - pred_material_uncond)
        pred_thickness_noise = pred_thickness_uncond + guidance_scale * (pred_thickness_cond - pred_thickness_uncond)
        
        # 去噪一步
        alpha_t = scheduler.alphas[t].to(device)
        alpha_cumprod_t = scheduler.alphas_cumprod[t].to(device)
        beta_t = scheduler.betas[t].to(device)
        
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
        
        materials_denoised = (materials_noisy - beta_t / sqrt_one_minus_alpha_cumprod_t * pred_material_noise) / sqrt_alpha_t
        thicknesses_denoised = (thicknesses_noisy - beta_t / sqrt_one_minus_alpha_cumprod_t * pred_thickness_noise) / sqrt_alpha_t
        
        # 添加随机噪声 (除了最后一步)
        if i < len(timesteps) - 1:
            noise_materials = torch.randn_like(materials_noisy)
            noise_thicknesses = torch.randn_like(thicknesses_noisy)
            
            variance = scheduler.betas[t]
            sigma = torch.sqrt(variance)
            
            materials_noisy = materials_denoised + sigma * noise_materials
            thicknesses_noisy = thicknesses_denoised + sigma * noise_thicknesses
        else:
            materials_noisy = materials_denoised
            thicknesses_noisy = thicknesses_denoised
    
    # 转换为结构格式
    materials_prob = F.softmax(materials_noisy, dim=-1)
    generated_structures = mixed_processor.denormalize_structure(materials_prob, thicknesses_noisy)
    
    return generated_structures

# 原有的sample_structures函数保持不变

@torch.no_grad()
def sample_structures(model, target_spectra, mixed_processor, scheduler, 
                     device, num_samples=1, num_inference_steps=100):
    """使用DDPM采样生成结构"""
    
    model.eval()
    
    batch_size = target_spectra.size(0) if target_spectra.dim() > 2 else 1
    if target_spectra.dim() == 2:
        target_spectra = target_spectra.unsqueeze(0)
    
    # 初始化随机噪声
    materials_shape = (batch_size, model.max_layers, model.vocab_size)
    thicknesses_shape = (batch_size, model.max_layers, 1)
    
    materials_noisy = torch.randn(materials_shape, device=device)
    thicknesses_noisy = torch.randn(thicknesses_shape, device=device)
    
    # DDPM采样过程
    timesteps = torch.linspace(scheduler.timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
    
    for i, t in enumerate(tqdm(timesteps, desc="DDPM采样")):
        timestep_batch = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)
        
        # 预测噪声
        pred_material_noise, pred_thickness_noise = model(
            materials_noisy, thicknesses_noisy, timestep_batch, target_spectra
        )
        
        # 去噪一步
        alpha_t = scheduler.alphas[t].to(device)
        alpha_cumprod_t = scheduler.alphas_cumprod[t].to(device)
        beta_t = scheduler.betas[t].to(device)
        
        # 计算去噪系数
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
        
        # 去噪
        materials_denoised = (materials_noisy - beta_t / sqrt_one_minus_alpha_cumprod_t * pred_material_noise) / sqrt_alpha_t
        thicknesses_denoised = (thicknesses_noisy - beta_t / sqrt_one_minus_alpha_cumprod_t * pred_thickness_noise) / sqrt_alpha_t
        
        # 添加随机噪声 (除了最后一步)
        if i < len(timesteps) - 1:
            noise_materials = torch.randn_like(materials_noisy)
            noise_thicknesses = torch.randn_like(thicknesses_noisy)
            
            variance = scheduler.betas[t].to(device)
            sigma = torch.sqrt(variance)
            
            materials_noisy = materials_denoised + sigma * noise_materials
            thicknesses_noisy = thicknesses_denoised + sigma * noise_thicknesses
        else:
            materials_noisy = materials_denoised
            thicknesses_noisy = thicknesses_denoised
    
    # 转换为结构格式
    # 对材料做 softmax 得到概率分布
    materials_prob = F.softmax(materials_noisy, dim=-1)
    generated_structures = mixed_processor.denormalize_structure(materials_prob, thicknesses_noisy)
    
    return generated_structures

# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="多层薄膜扩散模型逆向设计")
    parser.add_argument('--data_path', type=str, default='dataset/multilayer_dataset.npz', help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=6, help='Transformer层数')
    parser.add_argument('--num_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--timesteps', type=int, default=1000, help='扩散步数')
    parser.add_argument('--train', action='store_true', help='是否训练模型')
    parser.add_argument('--sample', action='store_true', help='是否进行采样测试')
    parser.add_argument('--demo', action='store_true', help='演示模式')
    parser.add_argument('--gpu', action='store_true', help='GPU高性能训练模式(500轮,大批次)')
    parser.add_argument('--ultra', action='store_true', help='GPU超高性能训练模式(750轮,更大模型)')
    parser.add_argument('--ultra_v2', action='store_true', help='GPU超高性能训练模式V2(针对损失优化)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU设备ID')
    args = parser.parse_args()
    
    # 演示模式设置
    if args.demo:
        config = get_config('demo')
        args.train = True
        args.sample = True
        args.batch_size = config['batch_size']      # 16
        args.num_epochs = config['num_epochs']      # 30
        args.lr = config['lr']                      # 2e-4
        args.hidden_dim = config['hidden_dim']      # 256
        args.num_layers = config['num_layers']      # 6
        args.num_heads = config['num_heads']        # 8
        args.timesteps = config['timesteps']        # 1000
        safe_print(f"[演示] 演示模式: 使用优化配置 - Epochs: {args.num_epochs}, Batch: {args.batch_size}, Hidden: {args.hidden_dim}")
    
    # GPU超高性能训练模式V2设置  
    elif args.ultra_v2:
        config = get_config('ultra_v2')
        args.train = True
        args.sample = True  
        args.batch_size = config['batch_size']      # 192
        args.num_epochs = config['num_epochs']      # 750
        args.lr = config['lr']                      # 1.5e-4
        args.hidden_dim = config['hidden_dim']      # 640
        args.num_layers = config['num_layers']      # 10
        args.num_heads = config['num_heads']        # 20
        args.timesteps = config['timesteps']        # 1000
        safe_print(f"[ULTRA-V2] GPU超高性能模式V2: 针对损失优化")
        safe_print(f"   批次大小: {args.batch_size} (适度减小提升稳定性)")
        safe_print(f"   网络规模: Hidden={args.hidden_dim}, Layers={args.num_layers}, Heads={args.num_heads}")
        safe_print(f"   学习率: {args.lr} (提高加速初期收敛)")
        safe_print(f"   CFG丢弃: {config['cfg_dropout_prob']} (降低到GPU水平)")
    
    # GPU超高性能训练模式设置  
    elif args.ultra:
        config = get_config('ultra')
        args.train = True
        args.sample = True  
        args.batch_size = config['batch_size']      # 256
        args.num_epochs = config['num_epochs']      # 750
        args.lr = config['lr']                      # 8e-5
        args.hidden_dim = config['hidden_dim']      # 768
        args.num_layers = config['num_layers']      # 12
        args.num_heads = config['num_heads']        # 24
        args.timesteps = config['timesteps']        # 1000
        safe_print(f"[ULTRA] GPU超高性能模式: 750轮超长训练")
        safe_print(f"   批次大小: {args.batch_size} (每轮约{200000//args.batch_size}个批次)")
        safe_print(f"   网络规模: Hidden={args.hidden_dim}, Layers={args.num_layers}, Heads={args.num_heads}")
        safe_print(f"   学习率: {args.lr} (更保守的学习率)")
        safe_print(f"   预计训练时间: 12-18小时 (取决于GPU性能)")
    
    # GPU高性能训练模式设置
    elif args.gpu:
        config = get_config('gpu')
        args.train = True
        args.sample = True  
        args.batch_size = config['batch_size']      # 128
        args.num_epochs = config['num_epochs']      # 500
        args.lr = config['lr']                      # 1e-4
        args.hidden_dim = config['hidden_dim']      # 512
        args.num_layers = config['num_layers']      # 10
        args.num_heads = config['num_heads']        # 16
        args.timesteps = config['timesteps']        # 1000
        safe_print(f"[GPU] GPU高性能模式: 500轮正式训练")
        safe_print(f"   批次大小: {args.batch_size} (每轮约{200000//args.batch_size}个批次)")
        safe_print(f"   网络规模: Hidden={args.hidden_dim}, Layers={args.num_layers}, Heads={args.num_heads}")
        safe_print(f"   预计训练时间: 8-12小时 (取决于GPU性能)")
    
    # 设备配置
    device = get_device(args.gpu_id)
    
    # 加载数据集
    train_dataset = MultilayerDiffusionDataset(args.data_path, mode='train')
    test_dataset = MultilayerDiffusionDataset(args.data_path, mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    model = DiffusionUNet(
        vocab_size=train_dataset.vocab_size,
        max_layers=10,
        spectrum_dim=train_dataset.spectrum_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads
    ).to(device)
    
    # 创建噪声调度器和数据处理器
    scheduler = NoiseScheduler(timesteps=args.timesteps)
    # 将调度器参数移动到GPU
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    scheduler.betas = scheduler.betas.to(device)
    scheduler.sqrt_alphas_cumprod = scheduler.sqrt_alphas_cumprod.to(device)
    scheduler.sqrt_one_minus_alphas_cumprod = scheduler.sqrt_one_minus_alphas_cumprod.to(device)
    
    mixed_processor = MixedTypeProcessor(
        vocab_size=train_dataset.vocab_size,
        max_layers=10
    )
    
    # 训练模型
    if args.train:
        safe_print("开始训练扩散模型...")
        
        # 获取配置
        if args.ultra_v2:
            mode = 'ultra_v2'
        elif args.ultra:
            mode = 'ultra'
        elif args.gpu:
            mode = 'gpu'
        elif args.demo:
            mode = 'demo'
        else:
            mode = 'full'  # 默认使用完整配置
        config = get_config(mode)
        
        history = train_diffusion_model(
            model=model,
            dataloader=train_loader,
            num_epochs=args.num_epochs,
            device=device,
            scheduler=scheduler,
            mixed_processor=mixed_processor,
            lr=args.lr,
            save_interval=config.get('save_interval', 10),
            cfg_dropout_prob=config.get('cfg_dropout_prob', 0.1),
            config=config,
            mode=mode
        )
        
        # 绘制训练历史
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 支持中文显示
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history['epoch_losses'])
        plt.title('总损失 (Total Loss)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 3, 2)
        plt.plot(history['material_losses'])
        plt.title('材料损失 (Material Loss)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 3, 3)
        plt.plot(history['thickness_losses'])
        plt.title('厚度损失 (Thickness Loss)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('diffusion_training_history.png', dpi=300)
        plt.close()
        safe_print("[保存] 训练历史图已保存")
    
    # 采样测试
    if args.sample:
        if not args.train:
            try:
                model.load_state_dict(torch.load("diffusion_model_final.pth"))
                safe_print("成功加载训练好的模型")
            except:
                safe_print("无法加载模型，请先训练")
                return
        
        safe_print("开始采样测试...")
        
        # 从测试集中获取目标光谱
        test_batch = next(iter(test_loader))
        target_spectra = test_batch['spectrum'][:4].to(device)  # 取前4个样本
        real_structures = test_batch['structure'][:4]
        
        # 生成结构
        generated_structures = sample_structures(
            model=model,
            target_spectra=target_spectra,
            mixed_processor=mixed_processor,
            scheduler=scheduler,
            device=device,
            num_samples=4,
            num_inference_steps=50
        )
        
        # 打印结果对比
        safe_print("\n采样结果对比:")
        for i in range(4):
            safe_print(f"\n样本 {i+1}:")
            safe_print("真实结构:")
            real_struct = real_structures[i].numpy()
            for j, (mat_idx, thick) in enumerate(real_struct):
                if mat_idx < len(train_dataset.materials) and thick > 0:
                    material = train_dataset.materials[int(mat_idx)]
                    safe_print(f"  层{j+1}: {material} ({thick:.1f} nm)")
            
            safe_print("生成结构:")
            gen_struct = generated_structures[i].cpu().numpy()
            for j, (mat_idx, thick) in enumerate(gen_struct):
                if mat_idx < len(train_dataset.materials) and thick > 0:
                    material = train_dataset.materials[int(mat_idx)]
                    safe_print(f"  层{j+1}: {material} ({thick:.1f} nm)")

if __name__ == "__main__":
    main()