#!/usr/bin/env python3
# pure_diffusion_model.py
"""
Pure Diffusion-based Inverse Design for Optical Multilayer Structures

This script implements a conditional diffusion model for inverse design WITHOUT PNN.
The model learns to generate material sequences and thicknesses conditioned on target spectra.

Key Design Principles:
1. Pure diffusion: Only denoising loss, no auxiliary physics loss
2. Conditional generation: Target spectra embedded as condition via trainable encoder
3. Classifier-free guidance: Enable flexible sampling control
4. Standard DDPM scheduler: Proven noise schedule

Architecture:
- Input: Noisy material probabilities + thicknesses + timestep + spectrum condition
- Output: Predicted noise (epsilon prediction)
- Loss: MSE between predicted and true noise

Usage:
    python pure_diffusion_model.py --data optimized_dataset/optimized_multilayer_dataset.npz --epochs 100 --batch 128
"""

import os
import argparse
import random
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm

# ---------------------------
# Placeholder adapters (replace with your actual implementations if available)
# ---------------------------

class NoiseScheduler:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.timesteps = T
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        betas = torch.cat([torch.tensor([0.0], device=device), betas])  # index 0 unused / alpha_bar_0 = 1
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod

    def q_sample(self, x0: torch.Tensor, eps: torch.Tensor, t: torch.LongTensor):
        # t should be in 1..T
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

# Minimal dataset loader compatible with your optimized dataset
class MultilayerDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        # expected keys: structures, wavelengths, transmission, reflection
        self.structures = list(data['structures'])
        self.wavelengths = data['wavelengths']
        self.T = np.array(data['transmission'])
        self.R = np.array(data['reflection'])
        self.N = len(self.structures)
        # infer max_layers and material vocab
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

# ---------------------------
# Model building blocks (compact)
# ---------------------------

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
        # Apply time embedding
        bias = self.time_proj(t_emb).unsqueeze(1)
        # Apply condition embedding if present
        if cond_emb is not None and self.cond_proj is not None:
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
        # Apply mask to residual block output, not to input
        h = self.norm1(x)
        h = self.act(self.fc1(h))
        h = self.fc2(h)
        h = self.norm2(h)
        # Apply mask only to the new residual contribution
        if mask is not None:
            h = h * mask.unsqueeze(-1).float()
        return x + h

# ---------------------------
# Diffusion UNet-like network for sequence (layers)
# ---------------------------

class EnhancedDiffusionUNet(nn.Module):
    def __init__(self, layer_count, vocab_size, hidden_dim=256, time_emb_dim=128, cond_dim=128):
        super().__init__()
        self.layer_count = layer_count
        self.vocab_size = vocab_size
        self.input_dim = vocab_size + 1
        self.initial = nn.Linear(self.input_dim, hidden_dim)
        self.res1 = ResBlock(hidden_dim, time_emb_dim, cond_dim)
        self.trans = NoAttentionTransformerBlock(hidden_dim)
        self.res2 = ResBlock(hidden_dim, time_emb_dim, cond_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.material_head = nn.Linear(hidden_dim, vocab_size)
        self.thickness_head = nn.Linear(hidden_dim, 1)
    def forward(self, mat_noisy, thk_noisy, timesteps, cond_emb, layer_mask, drop_mask=None):
        B,L,V = mat_noisy.shape
        # cond per-sample mask: True means drop condition, False means keep condition
        # If drop_mask[i] is True, set cond_emb[i] to zeros for that sample
        t_emb = sinusoidal_time_embedding(timesteps, dim=128).to(mat_noisy.device)
        
        # Apply per-sample condition dropping for classifier-free guidance
        cond_emb_masked = cond_emb.clone() if cond_emb is not None else None
        if drop_mask is not None and cond_emb_masked is not None:
            # drop_mask: (B,) boolean tensor, True means drop condition
            cond_emb_masked = cond_emb_masked * (~drop_mask).float().unsqueeze(-1)
        
        x = torch.cat([mat_noisy, thk_noisy], dim=-1)
        x = self.initial(x)
        x = self.res1(x, t_emb, cond_emb_masked)
        x = self.trans(x, mask=layer_mask)
        x = self.res2(x, t_emb, cond_emb_masked)
        x = self.final_norm(x)
        mat_noise = self.material_head(x)
        thk_noise = self.thickness_head(x)
        return mat_noise, thk_noise

# ---------------------------
# Pure Diffusion Model - No PNN required
# ---------------------------
# ---------------------------
# Full Diffusion model wrapper
# ---------------------------

class PureDiffusionModel:
    def __init__(self, device='cuda', data_path=None, hidden_dim=256, lr=1e-4):
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith('cuda') else 'cpu')
        print(f"[Device] {self.device}")
        
        # Load dataset
        assert data_path is not None and os.path.exists(data_path), "Provide valid data file."
        self.ds = MultilayerDataset(data_path)
        
        # Initialize scheduler
        self.scheduler = NoiseScheduler(T=1000, beta_start=1e-4, beta_end=0.02, device=self.device)
        
        # Model parameters
        self.layer_count = self.ds.max_layers
        self.vocab_size = self.ds.vocab_size
        self.spectrum_dim = self.ds.S
        
        # Build model
        self.model = EnhancedDiffusionUNet(
            layer_count=self.layer_count,
            vocab_size=self.vocab_size,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # Spectrum encoder (trainable)
        self.spec_encoder = nn.Sequential(
            nn.LayerNorm(self.ds.S * 2),
            nn.Linear(self.ds.S * 2, 256),
            nn.SiLU(),
            nn.Linear(256, 128)
        ).to(self.device)
        
        # Optimizer (includes both model and encoder)
        all_params = list(self.model.parameters()) + list(self.spec_encoder.parameters())
        self.optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-6)
        
        # EMA for stable sampling
        self.ema = {n: p.detach().clone().cpu() for n,p in self.model.named_parameters() if p.requires_grad}
        self.ema_decay = 0.9999
        
        # Hyperparameters
        self.p_uncond = 0.1  # Probability of dropping condition during training
        self.guidance_w = 6.0  # Classifier-free guidance weight for sampling
        
        # Learning rate scheduler
        self.scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

    def _encode_spectrum(self, spectra):
        # Encode spectrum using trainable spectrum encoder
        # Expect spectra shape (B, 2S) where first S is transmission, second S is reflection
        return self.spec_encoder(spectra.to(self.device))

    def _update_ema(self):
        """Update EMA parameters (kept on CPU to save memory)."""
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.ema[n].mul_(self.ema_decay).add_(p.detach().cpu(), alpha=1.0 - self.ema_decay)

    def _apply_ema(self):
        # save current and apply ema
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
        # timesteps: (B,) long in 1..T
        B = x_t_mat.shape[0]
        device = x_t_mat.device
        alpha_bar_t = self.scheduler.alphas_cumprod[timesteps].to(device)  # shape (B,)
        shape = [B, 1, 1]
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t).view(*shape)
        sqrt_1m = torch.sqrt(1.0 - alpha_bar_t).view(*shape)
        x0_mat = (x_t_mat - sqrt_1m * pred_mat_noise) / (sqrt_alpha_bar + 1e-12)
        x0_thk = (x_t_thk - sqrt_1m * pred_thk_noise) / (sqrt_alpha_bar + 1e-12)
        return x0_mat, x0_thk

    def train_step(self, batch, grad_debug=False):
        mat_idx, thickness_norm, layer_mask, target = batch
        mat_idx = mat_idx.to(self.device)
        thickness_norm = thickness_norm.to(self.device)
        layer_mask = layer_mask.to(self.device)
        target = target.to(self.device)  # (B, 2S)

        B = mat_idx.shape[0]
        # prepare 'clean' x0 in model space: we operate in logits-like continuous space for materials.
        # Represent materials as one-hot floats for forward q_sample
        materials_onehot = F.one_hot(mat_idx, num_classes=self.vocab_size).float().to(self.device)  # (B,L,V)
        thickness = thickness_norm.to(self.device)  # (B,L,1)

        # sample timesteps per-sample: 1..T inclusive
        timesteps = torch.randint(1, self.scheduler.timesteps + 1, (B,), dtype=torch.long, device=self.device)

        # sample gaussian noise
        eps_mat = torch.randn_like(materials_onehot)
        eps_thk = torch.randn_like(thickness)

        # build x_t via closed form
        x_t_mat = self.scheduler.q_sample(materials_onehot, eps_mat, timesteps)
        x_t_thk = self.scheduler.q_sample(thickness, eps_thk, timesteps)

        # per-sample classifier-free mask
        drop_mask = (torch.rand(B, device=self.device) < self.p_uncond)

        # condition embedding: we use target spectra as condition for training
        cond_emb = self._encode_spectrum(target)  # (B, cond_dim)

        # model predicts noise
        pred_mat_noise, pred_thk_noise = self.model(x_t_mat, x_t_thk, timesteps, cond_emb, layer_mask, drop_mask)

        # primary noise loss
        loss_noise = F.mse_loss(pred_mat_noise, eps_mat) + F.mse_loss(pred_thk_noise, eps_thk)

        # reconstruct x0 hat from predicted noise
        x0_mat_hat, x0_thk_hat = self._reconstruct_x0(x_t_mat, x_t_thk, pred_mat_noise, pred_thk_noise, timesteps)

        # materials probabilities via softmax (differentiable)
        materials_probs_hat = F.softmax(x0_mat_hat, dim=-1)  # (B,L,V)
        thickness_hat = torch.clamp(x0_thk_hat, 0.0, 1.0)  # (B,L,1) stay within normalized range

        # Pure diffusion: only noise loss, no PNN or physics loss
        loss_mat = F.mse_loss(pred_mat_noise, eps_mat)
        loss_thk = F.mse_loss(pred_thk_noise, eps_thk)
        total_loss = loss_mat + loss_thk
        
        # Compute metrics for monitoring (no gradient)
        with torch.no_grad():
            # Material prediction accuracy
            mat_pred = torch.argmax(x0_mat_hat, dim=-1)
            mat_acc = (mat_pred == mat_idx).float().mean()
            
            # Thickness MAE
            thk_mae = torch.abs(x0_thk_hat.squeeze(-1) - thickness.squeeze(-1)).mean()

        if grad_debug:
            print("=" * 60)
            print("GRADIENT DEBUG INFO")
            print("=" * 60)
            print(f"  total_loss.requires_grad: {total_loss.requires_grad}")
            print(f"  loss_mat.requires_grad: {loss_mat.requires_grad}")
            print(f"  loss_thk.requires_grad: {loss_thk.requires_grad}")
            print(f"  pred_mat_noise.requires_grad: {pred_mat_noise.requires_grad}")
            print(f"  x_t_mat.requires_grad: {x_t_mat.requires_grad}")
            print(f"  cond_emb.requires_grad: {cond_emb.requires_grad}")
            print("=" * 60)

        # backward & step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Update EMA
        self._update_ema()
        
        metrics = {
            'loss': float(total_loss.detach().cpu()),
            'loss_mat': float(loss_mat.detach().cpu()),
            'loss_thk': float(loss_thk.detach().cpu()),
            'mat_acc': float(mat_acc.cpu()),
            'mae_thk': float(thk_mae.cpu())
        }
        return metrics

    @torch.no_grad()
    def eval_step(self, batch):
        """Validation forward without weight updates."""
        mat_idx, thickness_norm, layer_mask, target = batch
        mat_idx = mat_idx.to(self.device)
        thickness_norm = thickness_norm.to(self.device)
        layer_mask = layer_mask.to(self.device)
        target = target.to(self.device)

        B = mat_idx.shape[0]
        materials_onehot = F.one_hot(mat_idx, num_classes=self.vocab_size).float().to(self.device)
        thickness = thickness_norm

        timesteps = torch.randint(1, self.scheduler.timesteps + 1, (B,), dtype=torch.long, device=self.device)
        eps_mat = torch.randn_like(materials_onehot)
        eps_thk = torch.randn_like(thickness)
        x_t_mat = self.scheduler.q_sample(materials_onehot, eps_mat, timesteps)
        x_t_thk = self.scheduler.q_sample(thickness, eps_thk, timesteps)

        drop_mask = (torch.rand(B, device=self.device) < self.p_uncond)
        cond_emb = self._encode_spectrum(target)
        pred_mat_noise, pred_thk_noise = self.model(x_t_mat, x_t_thk, timesteps, cond_emb, layer_mask, drop_mask)

        # Compute validation loss
        loss_mat = F.mse_loss(pred_mat_noise, eps_mat)
        loss_thk = F.mse_loss(pred_thk_noise, eps_thk)
        total_loss = loss_mat + loss_thk

        # Reconstruct for metrics
        x0_mat_hat, x0_thk_hat = self._reconstruct_x0(x_t_mat, x_t_thk, pred_mat_noise, pred_thk_noise, timesteps)
        mat_pred = torch.argmax(x0_mat_hat, dim=-1)
        mat_acc = (mat_pred == mat_idx).float().mean()
        thk_mae = torch.abs(x0_thk_hat.squeeze(-1) - thickness.squeeze(-1)).mean()

        return {
            'loss': float(total_loss.detach().cpu()),
            'loss_mat': float(loss_mat.detach().cpu()),
            'loss_thk': float(loss_thk.detach().cpu()),
            'mat_acc': float(mat_acc.cpu()),
            'mae_thk': float(thk_mae.cpu())
        }

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
            # For CFG: get predictions with condition
            drop_mask_cond = torch.zeros(B, dtype=torch.bool, device=device)
            eps_cond_mat, eps_cond_thk = self.model(x, x_thk, t_tensor, cond_emb, layer_mask, drop_mask=drop_mask_cond)
            # For CFG: get predictions without condition (drop_mask=True drops the condition)
            drop_mask_uncond = torch.ones(B, dtype=torch.bool, device=device)
            eps_uncond_mat, eps_uncond_thk = self.model(x, x_thk, t_tensor, cond_emb, layer_mask, drop_mask=drop_mask_uncond)
            # Apply classifier-free guidance
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

        # final decode
        x0_mat = x
        x0_thk = x_thk
        materials_probs = F.softmax(x0_mat, dim=-1)
        # NOTE: we return CPU-side physical structures via a simple decoder (argmax)
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

# ---------------------------
# CLI and training runner
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='path to optimized_dataset .npz')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--guidance', type=float, default=6.0)
    p.add_argument('--grad_debug', action='store_true', help='print grad debug info for first batch')
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()

def train_model_cli(args):
    device = args.device
    model = PureDiffusionModel(device=device, data_path=args.data, lr=args.lr)
    model.guidance_w = args.guidance
    # prepare dataloader
    ds = model.ds
    val_split = int(len(ds) * 0.05)
    train_len = len(ds) - val_split
    train_ds, val_ds = random_split(ds, [train_len, val_split])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
    print("[data] training:", train_len, "validation:", val_split)
    best_val = float('inf')
    history = []
    for epoch in range(1, args.epochs+1):
        model.model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        epoch_loss = 0.0
        for i, batch in enumerate(pbar):
            if args.grad_debug and epoch==1 and i==0:
                metrics = model.train_step(batch, grad_debug=True)
            else:
                metrics = model.train_step(batch, grad_debug=False)
            epoch_loss += metrics['loss'] * batch[0].shape[0]
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'mat_acc': f"{metrics['mat_acc']:.3f}",
                'thk_mae': f"{metrics['mae_thk']:.4f}"
            })
        epoch_loss = epoch_loss / len(train_ds)
        
        # proper validation (no weight updates)
        model.model.eval()
        val_loss = 0.0
        val_mat_acc = 0.0
        val_thk_mae = 0.0
        n_val = 0
        for batch in val_loader:
            ev = model.eval_step(batch)
            bs = batch[0].shape[0]
            val_loss += ev['loss'] * bs
            val_mat_acc += ev['mat_acc'] * bs
            val_thk_mae += ev['mae_thk'] * bs
            n_val += bs
        if n_val > 0:
            val_loss /= n_val
            val_mat_acc /= n_val
            val_thk_mae /= n_val
        print(f"Epoch {epoch} finished. train loss: {epoch_loss:.6f} | val loss: {val_loss:.6f} | val mat_acc: {val_mat_acc:.3f} | val thk_mae: {val_thk_mae:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model_state': model.model.state_dict()}, 'pure_diffusion_best.pt')
            print("Saved best diffusion model: pure_diffusion_best.pt")
        history.append(epoch_loss)
    print("Training finished.")
    return model, history

if __name__ == '__main__':
    args = parse_args()
    model, history = train_model_cli(args)
    # Optionally save final model
    torch.save({'model_state': model.model.state_dict()}, 'pure_diffusion_model_final.pt')
    print("Saved final model: pure_diffusion_model_final.pt")
