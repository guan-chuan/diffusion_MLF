#!/usr/bin/env python3
# pnn.py - Enhanced Version
"""
PNN训练脚本 - 增强版
集成了所有优化建议：
- 修复了torch.clamp梯度阻断问题
- 添加位置编码
- 物理约束损失
- 学习率warmup
- 分别处理T和R的选项
- 更好的训练监控

使用方法:
  # 基础训练（推荐开始）
  python pnn.py --data optimized_dataset/optimized_multilayer_dataset.npz --config basic
  
  # 高性能训练
  python pnn.py --data optimized_dataset/optimized_multilayer_dataset.npz --config large
  
  # 自定义参数
  python pnn.py --data optimized_dataset/optimized_multilayer_dataset.npz --epochs 100 --batch 512
"""

import os
import argparse
import random
import json
from typing import List, Tuple
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MultilayerDataset(Dataset):
    """数据集加载器"""
    def __init__(self, npz_path: str, material_vocab: List[str] = None, max_layers: int = None):
        data = np.load(npz_path, allow_pickle=True)
        structures = data['structures']
        wavelengths = data['wavelengths']
        transmission = data['transmission']
        reflection = data['reflection']

        self.structures_raw = list(structures)
        self.wavelengths = wavelengths
        self.T = np.array(transmission)
        self.R = np.array(reflection)
        self.N = len(self.structures_raw)
        
        if max_layers is None:
            max_layers = max(len(s) for s in self.structures_raw)
        self.max_layers = max_layers

        # 构建材料词汇表
        if material_vocab is None:
            mats = set()
            for s in self.structures_raw:
                for pair in s:
                    mat = pair[0]
                    if isinstance(mat, str) and mat != '':
                        mats.add(mat)
            mats = sorted(list(mats))
            if 'VOID' not in mats:
                mats = ['VOID'] + mats
            self.material_vocab = mats
        else:
            self.material_vocab = material_vocab

        self.mat2idx = {m: i for i, m in enumerate(self.material_vocab)}
        self.vocab_size = len(self.material_vocab)

        # 预处理结构
        self.mat_idx = np.zeros((self.N, self.max_layers), dtype=np.int64)
        self.thickness = np.zeros((self.N, self.max_layers, 1), dtype=np.float32)
        self.layer_mask = np.zeros((self.N, self.max_layers), dtype=np.bool_)

        for i, s in enumerate(self.structures_raw):
            for j in range(self.max_layers):
                if j < len(s):
                    pair = s[j]
                    mat = pair[0] if len(pair) > 0 else ''
                    thk = float(pair[1]) if len(pair) > 1 else 0.0
                    if (not isinstance(mat, str)) or mat == '':
                        idx = self.mat2idx.get('VOID', 0)
                        self.mat_idx[i, j] = idx
                        self.thickness[i, j, 0] = 0.0
                        self.layer_mask[i, j] = False
                    else:
                        idx = self.mat2idx.get(mat, None)
                        if idx is None:
                            idx = self.mat2idx.get('VOID', 0)
                        self.mat_idx[i, j] = idx
                        self.thickness[i, j, 0] = thk
                        self.layer_mask[i, j] = True
                else:
                    idx = self.mat2idx.get('VOID', 0)
                    self.mat_idx[i, j] = idx
                    self.thickness[i, j, 0] = 0.0
                    self.layer_mask[i, j] = False

        # 归一化厚度
        all_thk = self.thickness[self.layer_mask]
        if all_thk.size == 0:
            self.thk_min = 0.0
            self.thk_max = 1.0
        else:
            self.thk_min = float(all_thk.min())
            self.thk_max = float(all_thk.max())
        if self.thk_max <= self.thk_min:
            self.thk_max = self.thk_min + 1.0

        self.thickness_norm = (self.thickness - self.thk_min) / (self.thk_max - self.thk_min)
        self.thickness_norm = np.clip(self.thickness_norm, 0.0, 1.0)

        self.S = self.T.shape[1]
        self.targets = np.concatenate([self.T, self.R], axis=1)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        mat_idx = torch.tensor(self.mat_idx[idx], dtype=torch.long)
        thickness = torch.tensor(self.thickness_norm[idx], dtype=torch.float32)
        mask = torch.tensor(self.layer_mask[idx], dtype=torch.bool)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return mat_idx, thickness, mask, target

# -------------------------
# Enhanced PNN Model
# -------------------------

class PNNTransformer(nn.Module):
    """
    增强版PNN Transformer
    新功能：
    - 位置编码
    - 分别处理T和R
    - 残差连接
    """
    def __init__(self, vocab_size: int, max_layers: int, spectrum_dim: int,
                 emb_dim: int = 128, n_heads: int = 4, n_encoder_layers: int = 3,
                 mlp_hidden: int = 256, dropout: float = 0.1, use_sigmoid: bool = False,
                 use_position_encoding: bool = True, separate_TR: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_layers = max_layers
        self.spectrum_dim = spectrum_dim
        self.emb_dim = emb_dim
        self.use_sigmoid = use_sigmoid
        self.use_position_encoding = use_position_encoding
        self.separate_TR = separate_TR

        self.mat_emb = nn.Embedding(vocab_size, emb_dim)
        self.thk_fc = nn.Linear(1, emb_dim)
        
        # 位置编码
        if use_position_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, max_layers, emb_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads,
            dim_feedforward=emb_dim*4, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        self.pool_fc = nn.Linear(emb_dim, emb_dim)
        
        # 分别处理T和R或统一处理
        if separate_TR:
            self.mlp_T = nn.Sequential(
                nn.Linear(emb_dim, mlp_hidden),
                nn.GELU(),
                nn.LayerNorm(mlp_hidden),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, spectrum_dim)
            )
            self.mlp_R = nn.Sequential(
                nn.Linear(emb_dim, mlp_hidden),
                nn.GELU(),
                nn.LayerNorm(mlp_hidden),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, spectrum_dim)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(emb_dim, mlp_hidden),
                nn.GELU(),
                nn.LayerNorm(mlp_hidden),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, spectrum_dim*2)
            )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, mat_idx_or_emb, thickness, mask):
        # 判断输入类型
        if mat_idx_or_emb.dtype == torch.long:
            # 原逻辑（整数索引→embedding lookup）
            x_mat = self.mat_emb(mat_idx_or_emb)
        else:
            # 新逻辑（连续embedding直接输入）
            x_mat = mat_idx_or_emb

        x_thk = self.thk_fc(thickness)
        x = x_mat + x_thk

        if self.use_position_encoding:
            x = x + self.pos_encoding[:, :x.size(1), :]

        src_key_padding_mask = ~mask
        x_enc = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x_enc = x_enc + x

        mask_f = mask.unsqueeze(-1).float()
        x_masked = x_enc * mask_f
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = x_masked.sum(dim=1) / denom
        pooled = self.pool_fc(pooled)

        if self.separate_TR:
            T_out = self.mlp_T(pooled)
            R_out = self.mlp_R(pooled)
            out = torch.cat([T_out, R_out], dim=1)
        else:
            out = self.mlp(pooled)

        if self.use_sigmoid:
            out = torch.sigmoid(out)

        return out


class PNNMLP(nn.Module):
    """简单MLP版本"""
    def __init__(self, vocab_size, max_layers, spectrum_dim, emb_dim=64, 
                 hidden=[256,256], dropout=0.1, use_sigmoid=False):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.mat_emb = nn.Embedding(vocab_size, emb_dim)
        in_dim = max_layers * (emb_dim + 1)
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(h))
            layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, spectrum_dim*2))
        self.net = nn.Sequential(*layers)

    def forward(self, mat_idx, thickness, mask):
        emb = self.mat_emb(mat_idx)
        emb = emb * mask.unsqueeze(-1).float()
        thk = thickness * mask.unsqueeze(-1).float()
        x = torch.cat([emb, thk], dim=-1)
        x = x.view(x.size(0), -1)
        out = self.net(x)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out

# -------------------------
# Enhanced Loss Functions
# -------------------------

def physics_constrained_loss(pred, target, spectrum_dim, lambda_conservation=0.1, lambda_negative=0.05):
    """
    物理约束损失
    - MSE基础损失
    - 能量守恒约束 (T+R<=1)
    - 非负约束 (T>=0, R>=0)
    """
    # 基础MSE损失
    mse_loss = F.mse_loss(pred, target)
    
    # 分离T和R
    pred_T = pred[:, :spectrum_dim]
    pred_R = pred[:, spectrum_dim:]
    
    # 能量守恒约束
    energy_sum = pred_T + pred_R
    violation = F.relu(energy_sum - 1.0)
    conservation_loss = violation.mean()
    
    # 非负约束
    negative_T = F.relu(-pred_T).mean()
    negative_R = F.relu(-pred_R).mean()
    negative_loss = negative_T + negative_R
    
    # 组合损失
    total_loss = mse_loss + lambda_conservation * conservation_loss + lambda_negative * negative_loss
    
    return total_loss, mse_loss, conservation_loss, negative_loss

def weighted_loss(pred, target, spectrum_dim, weight_T=1.0, weight_R=2.0):
    """
    加权损失 - R更难预测，给更高权重
    """
    pred_T = pred[:, :spectrum_dim]
    pred_R = pred[:, spectrum_dim:]
    target_T = target[:, :spectrum_dim]
    target_R = target[:, spectrum_dim:]
    
    loss_T = F.mse_loss(pred_T, target_T)
    loss_R = F.mse_loss(pred_R, target_R)
    
    total_loss = weight_T * loss_T + weight_R * loss_R
    return total_loss, loss_T, loss_R

# -------------------------
# Learning Rate Scheduler
# -------------------------

class WarmupCosineScheduler:
    """学习率Warmup + Cosine退火"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Warmup阶段
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine退火
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

# -------------------------
# Training
# -------------------------

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Device: {device}")
    set_seed(args.seed)

    # 加载数据集
    assert os.path.exists(args.data), f"Data file not found: {args.data}"
    ds = MultilayerDataset(args.data)
    vocab = ds.material_vocab
    max_layers = ds.max_layers
    spectrum_dim = ds.S

    print(f"Dataset samples: {len(ds)}, max_layers: {max_layers}, vocab_size: {len(vocab)}, spectrum_dim: {spectrum_dim}")
    print(f"Materials: {vocab}")

    # 数据集划分
    val_split = int(len(ds) * args.val_ratio)
    train_len = len(ds) - val_split
    train_ds, val_ds = random_split(ds, [train_len, val_split])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, 
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, 
                            num_workers=args.workers, pin_memory=True)

    # 创建模型
    if args.model_type == 'transformer':
        model = PNNTransformer(
            vocab_size=len(vocab), max_layers=max_layers,
            spectrum_dim=spectrum_dim, emb_dim=args.emb_dim,
            n_heads=args.n_heads, n_encoder_layers=args.n_layers,
            mlp_hidden=args.mlp_hidden, dropout=args.dropout,
            use_sigmoid=args.use_sigmoid,
            use_position_encoding=args.use_position_encoding,
            separate_TR=args.separate_TR
        )
    else:
        model = PNNMLP(
            vocab_size=len(vocab), max_layers=max_layers,
            spectrum_dim=spectrum_dim, emb_dim=args.emb_dim,
            hidden=[args.mlp_hidden, args.mlp_hidden//2], 
            dropout=args.dropout,
            use_sigmoid=args.use_sigmoid
        )

    model = model.to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    if args.use_warmup:
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs, base_lr=args.lr, min_lr=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )

    best_val = float('inf')
    train_losses = []
    val_losses = []
    learning_rates = []

    print(f"\nTraining configuration:")
    print(f"  Loss type: {args.loss_type}")
    print(f"  Use warmup: {args.use_warmup}")
    print(f"  Use position encoding: {args.use_position_encoding}")
    print(f"  Separate T/R: {args.separate_TR}")
    print(f"  Use sigmoid: {args.use_sigmoid}")
    print("")

    for epoch in range(1, args.epochs + 1):
        # 训练阶段
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        running_loss = 0.0
        running_mse = 0.0
        
        for batch in pbar:
            mat_idx, thickness, mask, target = batch
            mat_idx = mat_idx.to(device)
            thickness = thickness.to(device)
            mask = mask.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            out = model(mat_idx, thickness, mask)
            
            # 计算损失（不再使用clip！）
            if args.loss_type == 'physics':
                loss, mse_loss, cons_loss, neg_loss = physics_constrained_loss(
                    out, target, spectrum_dim,
                    lambda_conservation=args.lambda_conservation,
                    lambda_negative=args.lambda_negative
                )
                running_mse += mse_loss.item() * mat_idx.size(0)
            elif args.loss_type == 'weighted':
                loss, loss_T, loss_R = weighted_loss(
                    out, target, spectrum_dim,
                    weight_T=args.weight_T, weight_R=args.weight_R
                )
                running_mse += loss.item() * mat_idx.size(0)
            else:  # 'mse'
                loss = F.mse_loss(out, target)
                running_mse += loss.item() * mat_idx.size(0)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running_loss += loss.item() * mat_idx.size(0)
            pbar.set_postfix({'loss': loss.item()})

        epoch_train_loss = running_loss / train_len
        train_losses.append(epoch_train_loss)
        
        # 验证阶段
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for batch in val_loader:
                mat_idx, thickness, mask, target = batch
                mat_idx = mat_idx.to(device)
                thickness = thickness.to(device)
                mask = mask.to(device)
                target = target.to(device)
                out = model(mat_idx, thickness, mask)
                
                # 验证时使用简单MSE
                val_loss = F.mse_loss(out, target)
                val_running += val_loss.item() * mat_idx.size(0)
        
        epoch_val_loss = val_running / val_split if val_split > 0 else val_running / len(val_ds)
        val_losses.append(epoch_val_loss)
        
        # 更新学习率
        if args.use_warmup:
            current_lr = scheduler.step(epoch - 1)
        else:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        print(f"Epoch {epoch} | Train loss: {epoch_train_loss:.6f} | Val loss: {epoch_val_loss:.6f} | LR: {current_lr:.2e}")

        # 保存最佳模型
        if epoch_val_loss < best_val:
            best_val = epoch_val_loss
            torch.save({
                'model_state': model.state_dict(),
                'vocab': vocab,
                'max_layers': max_layers,
                'spectrum_dim': spectrum_dim,
                'thk_min': ds.thk_min,
                'thk_max': ds.thk_max,
                'use_sigmoid': args.use_sigmoid,
                'use_position_encoding': args.use_position_encoding,
                'separate_TR': args.separate_TR,
                'epoch': epoch,
                'val_loss': best_val
            }, args.save_best)
            print(f"  ✓ Saved best model (val {best_val:.6f})")

    # 保存最终模型
    torch.save({
        'model_state': model.state_dict(),
        'vocab': vocab,
        'max_layers': max_layers,
        'spectrum_dim': spectrum_dim,
        'thk_min': ds.thk_min,
        'thk_max': ds.thk_max,
        'use_sigmoid': args.use_sigmoid,
        'use_position_encoding': args.use_position_encoding,
        'separate_TR': args.separate_TR,
        'epoch': args.epochs,
        'val_loss': epoch_val_loss
    }, args.save_final)

    # 绘制损失曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(range(1, len(train_losses)+1), train_losses, label='Train', linewidth=2)
    ax1.plot(range(1, len(val_losses)+1), val_losses, label='Val', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.plot(range(1, len(learning_rates)+1), learning_rates, linewidth=2, color='green')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(args.loss_plot, dpi=300, bbox_inches='tight')
    print(f"Saved loss curve to {args.loss_plot}")

    print(f"\nTraining finished!")
    print(f"Best validation loss: {best_val:.6f}")
    print(f"Final validation loss: {epoch_val_loss:.6f}")
    
    return best_val

# -------------------------
# Preset Configurations
# -------------------------

def get_preset_config(config_name):
    """预设配置"""
    configs = {
        'basic': {
            'epochs': 100,
            'batch': 512,
            'lr': 5e-4,
            'emb_dim': 192,
            'n_layers': 4,
            'mlp_hidden': 384,
            'dropout': 0.15,
            'val_ratio': 0.1,
            'use_warmup': True,
            'warmup_epochs': 10,
            'use_position_encoding': True,
            'separate_TR': False,
            'loss_type': 'mse'
        },
        'large': {
            'epochs': 200,
            'batch': 512,
            'lr': 3e-4,
            'emb_dim': 256,
            'n_layers': 6,
            'mlp_hidden': 512,
            'dropout': 0.15,
            'val_ratio': 0.1,
            'use_warmup': True,
            'warmup_epochs': 15,
            'use_position_encoding': True,
            'separate_TR': True,
            'loss_type': 'physics',
            'lambda_conservation': 0.1,
            'lambda_negative': 0.05
        },
        'sigmoid': {
            'epochs': 150,
            'batch': 512,
            'lr': 5e-4,
            'emb_dim': 256,
            'n_layers': 5,
            'mlp_hidden': 512,
            'dropout': 0.15,
            'val_ratio': 0.1,
            'use_warmup': True,
            'warmup_epochs': 10,
            'use_position_encoding': True,
            'separate_TR': False,
            'use_sigmoid': True,
            'loss_type': 'weighted',
            'weight_T': 1.0,
            'weight_R': 2.0
        }
    }
    return configs.get(config_name, {})

# -------------------------
# CLI args
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Enhanced PNN Training')
    
    # 预设配置
    p.add_argument('--config', type=str, choices=['basic', 'large', 'sigmoid'],
                   help='Use preset configuration (basic/large/sigmoid)')
    
    # 数据和基础参数
    p.add_argument('--data', type=str, required=True, help='Path to dataset npz file')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--val_ratio', type=float, default=0.05)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    
    # 模型参数
    p.add_argument('--model_type', type=str, choices=['transformer', 'mlp'], default='transformer')
    p.add_argument('--emb_dim', type=int, default=128)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--n_layers', type=int, default=3)
    p.add_argument('--mlp_hidden', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.1)
    
    # 增强功能
    p.add_argument('--use_sigmoid', action='store_true', help='Use sigmoid activation')
    p.add_argument('--use_position_encoding', action='store_true', default=True, 
                   help='Use positional encoding')
    p.add_argument('--separate_TR', action='store_true', 
                   help='Use separate heads for T and R')
    
    # 损失函数选择
    p.add_argument('--loss_type', type=str, choices=['mse', 'physics', 'weighted'], 
                   default='mse', help='Loss function type')
    p.add_argument('--lambda_conservation', type=float, default=0.1,
                   help='Weight for energy conservation loss')
    p.add_argument('--lambda_negative', type=float, default=0.05,
                   help='Weight for non-negative constraint')
    p.add_argument('--weight_T', type=float, default=1.0, help='Weight for T loss')
    p.add_argument('--weight_R', type=float, default=2.0, help='Weight for R loss')
    
    # 学习率调度
    p.add_argument('--use_warmup', action='store_true', default=False,
                   help='Use learning rate warmup')
    p.add_argument('--warmup_epochs', type=int, default=10)
    
    # 优化参数
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--weight_decay', type=float, default=1e-6)
    
    # 保存路径
    p.add_argument('--save_best', type=str, default='pnn_best.pt')
    p.add_argument('--save_final', type=str, default='pnn_final.pt')
    p.add_argument('--loss_plot', type=str, default='loss_curve.png')
    
    p.add_argument('--cpu', action='store_true', help='Force CPU')
    
    args = p.parse_args()
    
    # 应用预设配置
    if args.config:
        preset = get_preset_config(args.config)
        for key, value in preset.items():
            if not hasattr(args, key) or getattr(args, key) == p.get_default(key):
                setattr(args, key, value)
        print(f"Using preset config: {args.config}")
        print(f"Config details: {preset}")
    
    return args

if __name__ == '__main__':
    args = parse_args()
    print("="*60)
    print("Enhanced PNN Training")
    print("="*60)
    train_model(args)
