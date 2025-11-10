#!/usr/bin/env python3
# enhanced_diffusion_model_no_pnn.py
"""
Diffusion-based inverse design (no PNN surrogate)
- This script implements the diffusion model training and sampling pipeline
  WITHOUT a PNN surrogate. Conditioning on target spectra is done via embedding.
- Designed for dataset with ~300k samples and NVIDIA 3090 GPU.
- Save as enhanced_diffusion_model_no_pnn.py and run:
    python enhanced_diffusion_model_no_pnn.py --data optimized_dataset/optimized_multilayer_dataset.npz --epochs 100 --batch 128

Notes & recommendations:
- If you run out of GPU memory on 3090 (24GB), reduce --batch to 64 or 32.
- To speed up experiments, reduce scheduler T to 200-400 (but final sampling quality may drop).
- This script focuses on stability: uses LayerNorm, SiLU, AdamW, EMA, gradient clipping.
"""

import os
import argparse
import math
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm

# -----------------------------
# Dataset
# -----------------------------
class MultilayerDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.structures = list(data['structures'])
        self.wavelengths = data['wavelengths']
        self.T = np.array(data['transmission'])
        self.R = np.array(data['reflection'])
        self.N = len(self.structures)
        # infer max layers and material vocab
        max_layers = max(len(s) for s in self.structures)
        self.max_layers = max_layers
        mats = set()
        for s in self.structures:
            for pair in s:
                mat = pair[0] if len(pair) > 0 else ''
                if isinstance(mat, str) and mat != '':
                    mats.add(mat)
        mats = sorted(list(mats))
        if 'VOID' not in mats:
            mats = ['VOID'] + mats
        self.material_vocab = mats
        self.mat2idx = {m:i for i,m in enumerate(self.material_vocab)}
        self.vocab_size = len(self.material_vocab)
        # preprocess into mat_idx, thickness, mask array
        self.mat_idx = np.zeros((self.N, self.max_layers), dtype=np.int64)
        self.thickness = np.zeros((self.N, self.max_layers, 1), dtype=np.float32)
        self.mask = np.zeros((self.N, self.max_layers), dtype=np.bool_)
        for i,s in enumerate(self.structures):
            for j in range(self.max_layers):
                if j < len(s):
                    mat = s[j][0] if len(s[j])>0 else ''
                    thk = float(s[j][1]) if len(s[j])>1 else 0.0
                    if not isinstance(mat, str) or mat == '':
                        idx = self.mat2idx['VOID']
                        self.mat_idx[i,j] = idx
                        self.thickness[i,j,0] = 0.0
                        self.mask[i,j] = False
                    else:
                        idx = self.mat2idx.get(mat, self.mat2idx['VOID'])
                        self.mat_idx[i,j] = idx
                        self.thickness[i,j,0] = thk
                        self.mask[i,j] = True
                else:
                    self.mat_idx[i,j] = self.mat2idx['VOID']
                    self.thickness[i,j,0] = 0.0
                    self.mask[i,j] = False
        # normalize thickness
        all_thk = self.thickness[self.mask]
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
        mask = torch.tensor(self.mask[idx], dtype=torch.bool)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return mat_idx, thickness, mask, target

# -----------------------------
# Noise scheduler (simple DDPM scheduler)
# -----------------------------
class NoiseScheduler:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.timesteps = T
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        betas = torch.cat([torch.tensor([0.0], device=device), betas])  # pad index 0
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod

    def q_sample(self, x0: torch.Tensor, eps: torch.Tensor, t: torch.LongTensor):
        # t in 1..T
        assert t.min().item() >= 1 and t.max().item() <= self.timesteps
        device = x0.device
        alpha_bar_t = self.alphas_cumprod[t].to(device)
        shape = [x0.shape[0]] + [1] * (x0.dim() - 1)
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t).view(*shape)
        sqrt_1m = torch.sqrt(1.0 - alpha_bar_t).view(*shape)
        return sqrt_alpha_bar * x0 + sqrt_1m * eps

    def posterior_variance(self, t: int) -> float:
        if t <= 0:
            return 0.0
        beta_t = float(self.betas[t].item())
        alpha_bar_t = float(self.alphas_cumprod[t].item())
        alpha_bar_prev = float(self.alphas_cumprod[t-1].item()) if t > 1 else 1.0
        var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        return max(var, 1e-12)

# -----------------------------
# Model blocks
# -----------------------------
def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int = 128):
    half = dim // 2
    device = timesteps.device
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb

class ResBlock(nn.Module):
    def __init__(self, dim, time_emb_dim, cond_dim=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(time_emb_dim, dim)
        self.cond_proj = nn.Linear(cond_dim, dim) if cond_dim is not None else None
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
    def forward(self, x, t_emb, cond_emb=None):
        h = self.norm1(x)
        h = self.act(self.fc1(h))
        bias = self.time_proj(t_emb).unsqueeze(1) if t_emb is not None else 0.0
        if cond_emb is not None:
            bias = bias + self.cond_proj(cond_emb).unsqueeze(1)
        h = h + bias
        h = self.norm2(h)
        h = self.act(self.fc2(h))
        return x + h

class NoAttentionTransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim*4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim*4, dim)
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
        h = self.norm1(x)
        h = self.act(self.fc1(h))
        h = self.fc2(h)
        h = self.norm2(h)
        if mask is not None:
            h = h * mask.unsqueeze(-1).float()
        return x + h

# -----------------------------
# UNet-like model for sequence
# -----------------------------
class EnhancedDiffusionUNet(nn.Module):
    def __init__(self, layer_count, vocab_size, hidden_dim=320, time_emb_dim=128, cond_dim=128):
        super().__init__()
        self.layer_count = layer_count
        self.vocab_size = vocab_size
        self.input_dim = vocab_size + 1  # materials one-hot (V) + thickness (1)
        
        # 适度的模型容量增加
        self.initial = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # 适度的网络深度（3层ResBlock + 1层Transformer）
        self.res1 = ResBlock(hidden_dim, time_emb_dim, cond_dim)
        self.res2 = ResBlock(hidden_dim, time_emb_dim, cond_dim)
        self.trans = NoAttentionTransformerBlock(hidden_dim)
        self.res3 = ResBlock(hidden_dim, time_emb_dim, cond_dim)
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # 简化的预测头
        self.material_head = nn.Linear(hidden_dim, vocab_size)
        self.thickness_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, mat_noisy, thk_noisy, timesteps, cond_emb, layer_mask, drop_mask=None):
        # mat_noisy: (B, L, V)    thk_noisy: (B, L, 1)
        B,L,V = mat_noisy.shape
        t_emb = sinusoidal_time_embedding(timesteps, dim=128).to(mat_noisy.device)
        x = torch.cat([mat_noisy, thk_noisy], dim=-1)
        x = self.initial(x)
        x = self.res1(x, t_emb, cond_emb)
        x = self.res2(x, t_emb, cond_emb)
        x = self.trans(x, mask=layer_mask)
        x = self.res3(x, t_emb, cond_emb)
        x = self.final_norm(x)
        mat_noise = self.material_head(x)
        thk_noise = self.thickness_head(x)
        return mat_noise, thk_noise

# -----------------------------
# Diffusion model wrapper (no PNN)
# -----------------------------
class EnhancedDiffusionModelNoPNN:
    def __init__(self, device='cuda', data_path=None):
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith('cuda') else 'cpu')
        print("[device]", self.device)
        assert data_path is not None and os.path.exists(data_path), "Provide valid data file."
        ds = MultilayerDataset(data_path)
        self.ds = ds
        # 改进噪声调度
        self.scheduler = NoiseScheduler(T=1000, beta_start=1e-4, beta_end=0.015, device=self.device)
        self.layer_count = ds.max_layers
        self.vocab_size = ds.vocab_size
        self.spectrum_dim = ds.S
        # 适度的模型容量
        self.model = EnhancedDiffusionUNet(self.layer_count, self.vocab_size, hidden_dim=320).to(self.device)
        # 保守的优化器设置
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=1.5e-4,  # 适中的学习率
            weight_decay=1e-6,
            betas=(0.9, 0.999)
        )
        # EMA state
        self.ema = {n: p.detach().clone().cpu() for n,p in self.model.named_parameters() if p.requires_grad}
        self.ema_decay = 0.9999
        # hyperparams
        self.p_uncond = 0.1      # classifier-free guidance prob during training
        self.guidance_w = 6.0    # CFG weight used at sampling
        # spec encoder (condition)
        self._spec_encoder = None
        # 添加学习率调度器
        self.scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

    def _encode_spectrum(self, spectra):
        # 适度的MLP条件编码器
        B = spectra.shape[0]
        if self._spec_encoder is None:
            self._spec_encoder = nn.Sequential(
                nn.LayerNorm(spectra.shape[-1]),
                nn.Linear(spectra.shape[-1], 256),
                nn.SiLU(),
                nn.Linear(256, 128)
            ).to(self.device)
        return self._spec_encoder(spectra.to(self.device))

    def _update_ema(self):
        for n,p in self.model.named_parameters():
            if p.requires_grad:
                # keep ema on CPU to save gpu memory
                self.ema[n].mul_(self.ema_decay).add_(p.detach().cpu(), alpha=1.0 - self.ema_decay)

    def _apply_ema(self):
        self._backup = {n: p.detach().clone() for n,p in self.model.named_parameters() if p.requires_grad}
        for n,p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.ema[n].to(p.device))

    def _restore_from_ema(self):
        for n,p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self._backup[n].to(p.device))
        del self._backup

    def _reconstruct_x0(self, x_t_mat, x_t_thk, pred_mat_noise, pred_thk_noise, timesteps):
        B = x_t_mat.shape[0]
        device = x_t_mat.device
        alpha_bar_t = self.scheduler.alphas_cumprod[timesteps].to(device)
        shape = [B, 1, 1]
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t).view(*shape)
        sqrt_1m = torch.sqrt(1.0 - alpha_bar_t).view(*shape)
        x0_mat = (x_t_mat - sqrt_1m * pred_mat_noise) / (sqrt_alpha_bar + 1e-12)
        x0_thk = (x_t_thk - sqrt_1m * pred_thk_noise) / (sqrt_alpha_bar + 1e-12)
        return x0_mat, x0_thk

    def train_step(self, batch, grad_debug=False):
        mat_idx, thickness_norm, mask, target = batch
        mat_idx = mat_idx.to(self.device)
        thickness_norm = thickness_norm.to(self.device)
        mask = mask.to(self.device)
        target = target.to(self.device)

        B = mat_idx.shape[0]
        materials_onehot = F.one_hot(mat_idx, num_classes=self.vocab_size).float().to(self.device)  # (B,L,V)
        thickness = thickness_norm.to(self.device)  # (B,L,1)

        timesteps = torch.randint(1, self.scheduler.timesteps + 1, (B,), dtype=torch.long, device=self.device)

        eps_mat = torch.randn_like(materials_onehot)
        eps_thk = torch.randn_like(thickness)

        x_t_mat = self.scheduler.q_sample(materials_onehot, eps_mat, timesteps)
        x_t_thk = self.scheduler.q_sample(thickness, eps_thk, timesteps)

        # classifier-free training mask (per-sample)
        drop_mask = (torch.rand(B, device=self.device) < self.p_uncond)

        cond_emb = self._encode_spectrum(target)  # (B, cond_dim)

        pred_mat_noise, pred_thk_noise = self.model(x_t_mat, x_t_thk, timesteps, cond_emb, mask, drop_mask)

        # 简化的损失计算
        loss_mat = F.mse_loss(pred_mat_noise, eps_mat)
        loss_thk = F.mse_loss(pred_thk_noise, eps_thk)
        
        # 简单的权重平衡
        total_loss = loss_mat + 0.8 * loss_thk
        
        # 计算详细指标（用于监控）
        with torch.no_grad():
            alpha_bar = self.scheduler.alphas_cumprod[timesteps].view(-1, 1, 1)
            x0_mat_hat = (x_t_mat - torch.sqrt(1 - alpha_bar) * pred_mat_noise) / torch.sqrt(alpha_bar)
            x0_thk_hat = (x_t_thk - torch.sqrt(1 - alpha_bar) * pred_thk_noise) / torch.sqrt(alpha_bar)
            
            mat_acc = (torch.argmax(x0_mat_hat, dim=-1) == mat_idx).float().mean()
            mae_thk = torch.abs(x0_thk_hat.squeeze(-1) - thickness.squeeze(-1)).mean()

        if grad_debug:
            print("DEBUG grad flags:")
            print("  loss_noise.requires_grad:", total_loss.requires_grad)
            print("  materials_onehot.requires_grad:", materials_onehot.requires_grad)
            print("  x_t_mat.requires_grad:", x_t_mat.requires_grad)

        # backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # ema update
        self._update_ema()

        metrics = {
            'loss': float(total_loss.detach().cpu()),
            'loss_mat': float(loss_mat.detach().cpu()),
            'loss_thk': float(loss_thk.detach().cpu()),
            'mat_acc': float(mat_acc.detach().cpu()),
            'mae_thk': float(mae_thk.detach().cpu())
        }
        return metrics


    @torch.no_grad()
    def sample(self, cond_spectra, num_samples=1, use_ema=True, guidance_w=None):
        guidance_w = self.guidance_w if guidance_w is None else guidance_w
        if use_ema:
            self._apply_ema()
        B = num_samples
        device = self.device
        mat_shape = (B, self.layer_count, self.vocab_size)
        thk_shape = (B, self.layer_count, 1)
        x = torch.randn(mat_shape, device=device)
        x_thk = torch.randn(thk_shape, device=device)
        layer_mask = torch.ones(B, self.layer_count, dtype=torch.bool, device=device)
        cond_emb = self._encode_spectrum(cond_spectra.to(device))

        for t in range(self.scheduler.timesteps, 0, -1):
            t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
            eps_cond_mat, eps_cond_thk = self.model(x, x_thk, t_tensor, cond_emb, layer_mask, drop_mask=None)
            eps_uncond_mat, eps_uncond_thk = self.model(x, x_thk, t_tensor, cond_emb*0.0, layer_mask, drop_mask=None)
            eps_mat = (1.0 + guidance_w) * eps_cond_mat - guidance_w * eps_uncond_mat
            eps_thk = (1.0 + guidance_w) * eps_cond_thk - guidance_w * eps_uncond_thk

            beta_t = float(self.scheduler.betas[t].item())
            alpha_t = float(self.scheduler.alphas[t].item())
            alpha_bar_t = float(self.scheduler.alphas_cumprod[t].item())
            coef1 = 1.0 / math.sqrt(alpha_t)
            coef2 = beta_t / math.sqrt(1.0 - alpha_bar_t)
            mu_mat = coef1 * (x - coef2 * eps_mat)
            mu_thk = coef1 * (x_thk - coef2 * eps_thk)
            if t > 1:
                sigma = math.sqrt(self.scheduler.posterior_variance(t))
                x = mu_mat + sigma * torch.randn_like(x)
                x_thk = mu_thk + sigma * torch.randn_like(x_thk)
            else:
                x = mu_mat
                x_thk = mu_thk

        x0_mat = x
        x0_thk = x_thk
        materials_probs = F.softmax(x0_mat, dim=-1)
        mats_idx = materials_probs.argmax(dim=-1).cpu().numpy()
        thks = x0_thk.squeeze(-1).cpu().numpy()
        results = []
        for i in range(B):
            layers = []
            for j in range(self.layer_count):
                midx = int(mats_idx[i,j])
                matname = self.ds.material_vocab[midx]
                thk_nm = float(thks[i,j]*(self.ds.thk_max - self.ds.thk_min) + self.ds.thk_min)
                layers.append((matname, thk_nm))
            results.append(layers)
        if use_ema:
            self._restore_from_ema()
        return results

# -----------------------------
# Training CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='path to optimized_dataset .npz')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--grad_debug', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--quick_check', action='store_true', help='run a single-batch quick sanity check then exit')
    return p.parse_args()

def train_model_cli(args):
    device = args.device
    model = EnhancedDiffusionModelNoPNN(device=device, data_path=args.data)
    ds = model.ds
    val_split = int(len(ds) * 0.05)
    train_len = len(ds) - val_split
    train_ds, val_ds = random_split(ds, [train_len, val_split])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
    print("[data] training:", train_len, "validation:", val_split)
    history = []
    best_val = float('inf')
    for epoch in range(1, args.epochs+1):
        model.model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        epoch_metrics = {'loss': 0, 'loss_mat': 0, 'loss_thk': 0, 'mat_acc': 0, 'mae_thk': 0}
        
        for i, batch in enumerate(pbar):
            if args.grad_debug and epoch==1 and i==0:
                metrics = model.train_step(batch, grad_debug=True)
            else:
                metrics = model.train_step(batch, grad_debug=False)
            
            # 累积指标
            for key in epoch_metrics:
                epoch_metrics[key] += metrics.get(key, 0) * batch[0].shape[0]
            
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'mat_acc': f"{metrics.get('mat_acc', 0):.3f}",
                'mae_thk': f"{metrics.get('mae_thk', 0):.4f}"
            })
            
            # quick-check early exit to speed debug if requested
            if args.quick_check:
                print("Quick check done; exiting.")
                return model, [metrics['loss']]
        
        # 计算平均指标
        for key in epoch_metrics:
            epoch_metrics[key] /= train_len
        
        # simple validation (no PNN)
        model.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                mat_idx, thickness, mask, target = batch
                # We compute forward but do not step optimizer
                mat_idx = mat_idx.to(model.device)
                thickness = thickness.to(model.device)
                mask = mask.to(model.device)
                target = target.to(model.device)
                # Generate noisy sample and compute noise prediction loss as proxy
                B = mat_idx.shape[0]
                materials_onehot = F.one_hot(mat_idx, num_classes=model.vocab_size).float().to(model.device)
                thickness_t = thickness.to(model.device)
                timesteps = torch.randint(1, model.scheduler.timesteps + 1, (B,), dtype=torch.long, device=model.device)
                eps_mat = torch.randn_like(materials_onehot)
                eps_thk = torch.randn_like(thickness_t)
                x_t_mat = model.scheduler.q_sample(materials_onehot, eps_mat, timesteps)
                x_t_thk = model.scheduler.q_sample(thickness_t, eps_thk, timesteps)
                cond_emb = model._encode_spectrum(target)
                pred_mat_noise, pred_thk_noise = model.model(x_t_mat, x_t_thk, timesteps, cond_emb, mask, drop_mask=None)
                loss_noise = F.mse_loss(pred_mat_noise, eps_mat) + F.mse_loss(pred_thk_noise, eps_thk)
                val_loss += loss_noise.item() * B
                break  # compute one batch only for quick approximate val
        val_loss = val_loss / (len(val_loader.dataset) if len(val_loader.dataset)>0 else 1.0)
        
        # 详细打印
        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"Train total: {epoch_metrics['loss']:.6f} | "
            f"Val proxy: {val_loss:.6f} | "
            f"mat_acc: {epoch_metrics['mat_acc']:.3f} | "
            f"mae_thk: {epoch_metrics['mae_thk']:.4f} | "
            f"loss_mat: {epoch_metrics['loss_mat']:.4f} | "
            f"loss_thk: {epoch_metrics['loss_thk']:.4f}"
        )

        history.append(epoch_metrics['loss'])
        
        # 学习率调度
        model.scheduler_lr.step()
    print("Training finished.")
    return model, history

def run_quick_check(data_path):
    # quick sanity check to ensure forward/backward works on a single batch
    ds = MultilayerDataset(data_path)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    model = EnhancedDiffusionModelNoPNN(device='cpu', data_path=data_path)
    metrics = model.train_step(batch, grad_debug=True)
    print("Quick check metrics:", metrics)

if __name__ == '__main__':
    args = parse_args()
    if args.quick_check:
        run_quick_check(args.data)
    else:
        model, history = train_model_cli(args)
        # save final model
        torch.save({'model_state': model.model.state_dict()}, 'enhanced_diffusion_model_no_pnn_final.pt')
        print("Saved final model: enhanced_diffusion_model_no_pnn_final.pt")
