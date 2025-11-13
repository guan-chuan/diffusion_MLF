#!/usr/bin/env python3
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
    python train_diffusion.py --data optimized_dataset/optimized_multilayer_dataset.npz --epochs 100 --batch 128
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
    """Dataset loader for multilayer structures with optical spectra."""
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.structures = list(data['structures'])
        self.wavelengths = data['wavelengths']
        self.T = np.array(data['transmission'])
        self.R = np.array(data['reflection'])
        self.N = len(self.structures)
        
        # Infer max layers and material vocab
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
        
        # Preprocess into mat_idx, thickness, mask arrays
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
        
        # Normalize thickness
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
# DDPM Noise Scheduler
# -----------------------------
class DDPMScheduler:
    """Standard DDPM noise scheduler with linear beta schedule."""
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.timesteps = T
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        # Pad index 0 (unused, for convenience)
        betas = torch.cat([torch.tensor([0.0], device=device), betas])
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod

    def q_sample(self, x0: torch.Tensor, eps: torch.Tensor, t: torch.LongTensor):
        """Forward diffusion: sample x_t from x_0 and noise."""
        # t should be in 1..T
        assert t.min().item() >= 1 and t.max().item() <= self.timesteps
        device = x0.device
        alpha_bar_t = self.alphas_cumprod[t].to(device)
        shape = [x0.shape[0]] + [1] * (x0.dim() - 1)
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t).view(*shape)
        sqrt_1m = torch.sqrt(1.0 - alpha_bar_t).view(*shape)
        return sqrt_alpha_bar * x0 + sqrt_1m * eps

    def posterior_variance(self, t: int) -> float:
        """Compute variance for reverse diffusion step."""
        if t <= 0:
            return 0.0
        beta_t = float(self.betas[t].item())
        alpha_bar_t = float(self.alphas_cumprod[t].item())
        alpha_bar_prev = float(self.alphas_cumprod[t-1].item()) if t > 1 else 1.0
        var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        return max(var, 1e-12)

# -----------------------------
# Model Components
# -----------------------------
def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int = 128):
    """Sinusoidal positional embeddings for timesteps."""
    half = dim // 2
    device = timesteps.device
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb

class ResidualBlock(nn.Module):
    """Residual block with time and condition embedding injection."""
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
        
        # Time modulation
        bias = self.time_proj(t_emb).unsqueeze(1)
        
        # Condition modulation (for classifier-free guidance)
        if cond_emb is not None and self.cond_proj is not None:
            bias = bias + self.cond_proj(cond_emb).unsqueeze(1)
        
        h = h + bias
        h = self.norm2(h)
        h = self.act(self.fc2(h))
        return x + h

class FeedForwardBlock(nn.Module):
    """Simple feedforward block with layer masking support."""
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x, mask=None):
        h = self.norm1(x)
        h = self.act(self.fc1(h))
        h = self.fc2(h)
        h = self.norm2(h)
        
        # Apply mask to residual contribution only
        if mask is not None:
            h = h * mask.unsqueeze(-1).float()
        return x + h

# -----------------------------
# Main Diffusion Model
# -----------------------------
class DiffusionUNet(nn.Module):
    """
    UNet-style architecture for sequence denoising.
    
    Input: noisy material one-hot + noisy thickness + timestep + condition
    Output: predicted noise for materials and thickness
    """
    def __init__(self, layer_count, vocab_size, hidden_dim=256, time_emb_dim=128, cond_dim=128):
        super().__init__()
        self.layer_count = layer_count
        self.vocab_size = vocab_size
        self.input_dim = vocab_size + 1  # materials one-hot (V) + thickness (1)
        
        # Input projection
        self.initial = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # Core processing blocks
        self.res1 = ResidualBlock(hidden_dim, time_emb_dim, cond_dim)
        self.res2 = ResidualBlock(hidden_dim, time_emb_dim, cond_dim)
        self.ffn = FeedForwardBlock(hidden_dim)
        self.res3 = ResidualBlock(hidden_dim, time_emb_dim, cond_dim)
        
        # Output heads
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.material_head = nn.Linear(hidden_dim, vocab_size)
        self.thickness_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, mat_noisy, thk_noisy, timesteps, cond_emb, layer_mask, drop_mask=None):
        """
        Args:
            mat_noisy: (B, L, V) noisy material one-hot
            thk_noisy: (B, L, 1) noisy thickness
            timesteps: (B,) timestep indices
            cond_emb: (B, cond_dim) condition embedding (spectrum)
            layer_mask: (B, L) valid layer mask
            drop_mask: (B,) boolean mask for classifier-free guidance (True = drop condition)
        
        Returns:
            mat_noise: (B, L, V) predicted material noise
            thk_noise: (B, L, 1) predicted thickness noise
        """
        # Time embedding
        t_emb = sinusoidal_time_embedding(timesteps, dim=time_emb_dim)
        
        # Apply classifier-free guidance mask to condition
        if drop_mask is not None and cond_emb is not None:
            # Zero out condition for samples where drop_mask is True
            cond_emb = cond_emb * (~drop_mask).float().unsqueeze(-1)
        
        # Concatenate material and thickness
        x = torch.cat([mat_noisy, thk_noisy], dim=-1)  # (B, L, V+1)
        
        # Process through network
        x = self.initial(x)
        x = self.res1(x, t_emb, cond_emb)
        x = self.res2(x, t_emb, cond_emb)
        x = self.ffn(x, mask=layer_mask)
        x = self.res3(x, t_emb, cond_emb)
        x = self.final_norm(x)
        
        # Predict noise
        mat_noise = self.material_head(x)
        thk_noise = self.thickness_head(x)
        
        return mat_noise, thk_noise

# -----------------------------
# Training Wrapper
# -----------------------------
class DiffusionModel:
    """Complete diffusion model with training and sampling logic."""
    
    def __init__(self, device='cuda', data_path=None, hidden_dim=256, lr=1e-4):
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith('cuda') else 'cpu')
        print(f"[Device] {self.device}")
        
        # Load dataset
        assert data_path is not None and os.path.exists(data_path), "Provide valid data file."
        self.ds = MultilayerDataset(data_path)
        
        # Initialize scheduler
        self.scheduler = DDPMScheduler(T=1000, beta_start=1e-4, beta_end=0.02, device=self.device)
        
        # Model parameters
        self.layer_count = self.ds.max_layers
        self.vocab_size = self.ds.vocab_size
        self.spectrum_dim = self.ds.S
        
        # Build model
        self.model = DiffusionUNet(
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
        """Encode target spectrum into condition embedding."""
        return self.spec_encoder(spectra.to(self.device))

    def _update_ema(self):
        """Update EMA parameters (kept on CPU to save memory)."""
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.ema[n].mul_(self.ema_decay).add_(p.detach().cpu(), alpha=1.0 - self.ema_decay)

    def _apply_ema(self):
        """Temporarily apply EMA parameters to model."""
        self._backup = {n: p.detach().clone() for n,p in self.model.named_parameters() if p.requires_grad}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.ema[n].to(p.device))

    def _restore_from_ema(self):
        """Restore original parameters after EMA application."""
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self._backup[n].to(p.device))
        del self._backup

    def train_step(self, batch, grad_debug=False):
        """
        Single training step.
        
        Returns:
            metrics: dict with loss, material accuracy, thickness MAE
        """
        mat_idx, thickness_norm, mask, target = batch
        mat_idx = mat_idx.to(self.device)
        thickness_norm = thickness_norm.to(self.device)
        mask = mask.to(self.device)
        target = target.to(self.device)
        
        B = mat_idx.shape[0]
        
        # Convert to one-hot (continuous space for diffusion)
        materials_onehot = F.one_hot(mat_idx, num_classes=self.vocab_size).float()
        thickness = thickness_norm
        
        # Sample timesteps (1..T inclusive)
        timesteps = torch.randint(1, self.scheduler.timesteps + 1, (B,), dtype=torch.long, device=self.device)
        
        # Sample noise
        eps_mat = torch.randn_like(materials_onehot)
        eps_thk = torch.randn_like(thickness)
        
        # Forward diffusion (add noise)
        x_t_mat = self.scheduler.q_sample(materials_onehot, eps_mat, timesteps)
        x_t_thk = self.scheduler.q_sample(thickness, eps_thk, timesteps)
        
        # Classifier-free guidance: randomly drop condition
        drop_mask = (torch.rand(B, device=self.device) < self.p_uncond)
        
        # Encode condition
        cond_emb = self._encode_spectrum(target)
        
        # Predict noise
        pred_mat_noise, pred_thk_noise = self.model(
            x_t_mat, x_t_thk, timesteps, cond_emb, mask, drop_mask
        )
        
        # Compute loss (simple MSE between predicted and true noise)
        loss_mat = F.mse_loss(pred_mat_noise, eps_mat)
        loss_thk = F.mse_loss(pred_thk_noise, eps_thk)
        total_loss = loss_mat + loss_thk
        
        # Compute metrics for monitoring (no gradient)
        with torch.no_grad():
            # Reconstruct x0 from x_t and predicted noise
            alpha_bar = self.scheduler.alphas_cumprod[timesteps].view(-1, 1, 1).to(self.device)
            sqrt_alpha_bar = torch.sqrt(alpha_bar)
            sqrt_1m_alpha_bar = torch.sqrt(1.0 - alpha_bar)
            
            x0_mat_hat = (x_t_mat - sqrt_1m_alpha_bar * pred_mat_noise) / (sqrt_alpha_bar + 1e-12)
            x0_thk_hat = (x_t_thk - sqrt_1m_alpha_bar * pred_thk_noise) / (sqrt_alpha_bar + 1e-12)
            
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
        
        # Backward pass
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
    def sample(self, cond_spectra, num_samples=1, use_ema=True, guidance_w=None):
        """
        Sample structures conditioned on target spectra using DDPM reverse process.
        
        Args:
            cond_spectra: (num_samples, 2*S) target spectra [T, R]
            num_samples: number of samples to generate
            use_ema: whether to use EMA parameters
            guidance_w: classifier-free guidance weight
        
        Returns:
            results: list of generated structures (material, thickness) tuples
        """
        guidance_w = self.guidance_w if guidance_w is None else guidance_w
        
        if use_ema:
            self._apply_ema()
        
        self.model.eval()
        
        B = num_samples
        device = self.device
        
        # Start from pure noise
        x = torch.randn(B, self.layer_count, self.vocab_size, device=device)
        x_thk = torch.randn(B, self.layer_count, 1, device=device)
        layer_mask = torch.ones(B, self.layer_count, dtype=torch.bool, device=device)
        
        # Encode condition
        cond_emb = self._encode_spectrum(cond_spectra.to(device))
        
        # Reverse diffusion (from T to 1)
        for t in tqdm(range(self.scheduler.timesteps, 0, -1), desc="Sampling", leave=False):
            t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
            
            # Predict noise with condition
            eps_cond_mat, eps_cond_thk = self.model(x, x_thk, t_tensor, cond_emb, layer_mask, drop_mask=None)
            
            # Predict noise without condition (for CFG)
            eps_uncond_mat, eps_uncond_thk = self.model(x, x_thk, t_tensor, cond_emb*0.0, layer_mask, drop_mask=None)
            
            # Classifier-free guidance
            eps_mat = (1.0 + guidance_w) * eps_cond_mat - guidance_w * eps_uncond_mat
            eps_thk = (1.0 + guidance_w) * eps_cond_thk - guidance_w * eps_uncond_thk
            
            # Compute reverse step parameters
            beta_t = float(self.scheduler.betas[t].item())
            alpha_t = float(self.scheduler.alphas[t].item())
            alpha_bar_t = float(self.scheduler.alphas_cumprod[t].item())
            
            coef1 = 1.0 / math.sqrt(alpha_t)
            coef2 = beta_t / math.sqrt(1.0 - alpha_bar_t)
            
            # Mean of reverse distribution
            mu_mat = coef1 * (x - coef2 * eps_mat)
            mu_thk = coef1 * (x_thk - coef2 * eps_thk)
            
            # Add noise (except at last step)
            if t > 1:
                sigma = math.sqrt(self.scheduler.posterior_variance(t))
                x = mu_mat + sigma * torch.randn_like(x)
                x_thk = mu_thk + sigma * torch.randn_like(x_thk)
            else:
                x = mu_mat
                x_thk = mu_thk
        
        # Convert to discrete materials
        materials_probs = F.softmax(x, dim=-1)
        mats_idx = materials_probs.argmax(dim=-1).cpu().numpy()
        
        # Denormalize thickness
        thks = x_thk.squeeze(-1).cpu().numpy()
        thks = thks * (self.ds.thk_max - self.ds.thk_min) + self.ds.thk_min
        
        # Format results
        results = []
        for i in range(B):
            layers = []
            for j in range(self.layer_count):
                midx = int(mats_idx[i, j])
                matname = self.ds.material_vocab[midx]
                thk_nm = float(thks[i, j])
                layers.append((matname, thk_nm))
            results.append(layers)
        
        if use_ema:
            self._restore_from_ema()
        
        self.model.train()
        
        return results

# -----------------------------
# Training Script
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train pure diffusion model for optical inverse design")
    p.add_argument('--data', required=True, help='path to dataset .npz file')
    p.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    p.add_argument('--batch', type=int, default=128, help='batch size')
    p.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    p.add_argument('--hidden_dim', type=int, default=256, help='model hidden dimension')
    p.add_argument('--device', default='cuda:0', help='device to use')
    p.add_argument('--grad_debug', action='store_true', help='print gradient debug info')
    p.add_argument('--save_every', type=int, default=10, help='save checkpoint every N epochs')
    return p.parse_args()

def train(args):
    """Main training loop."""
    # Initialize model
    model = DiffusionModel(
        device=args.device,
        data_path=args.data,
        hidden_dim=args.hidden_dim,
        lr=args.lr
    )
    
    # Split dataset
    ds = model.ds
    val_split = int(len(ds) * 0.05)
    train_len = len(ds) - val_split
    train_ds, val_ds = random_split(ds, [train_len, val_split])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"[Dataset] Train: {train_len}, Val: {val_split}")
    print(f"[Vocab] Materials: {model.vocab_size}, Max layers: {model.layer_count}")
    print(f"[Model] Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    
    history = []
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.model.train()
        
        # Training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        epoch_metrics = {'loss': 0, 'loss_mat': 0, 'loss_thk': 0, 'mat_acc': 0, 'mae_thk': 0}
        
        for i, batch in enumerate(pbar):
            # Debug gradients on first batch of first epoch
            debug = (args.grad_debug and epoch == 1 and i == 0)
            metrics = model.train_step(batch, grad_debug=debug)
            
            # Accumulate metrics
            batch_size = batch[0].shape[0]
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key] * batch_size
            
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'mat_acc': f"{metrics['mat_acc']:.3f}",
                'thk_mae': f"{metrics['mae_thk']:.4f}"
            })
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= train_len
        
        # Validation
        model.model.eval()
        val_metrics = {'loss': 0, 'loss_mat': 0, 'loss_thk': 0}
        
        with torch.no_grad():
            for batch in val_loader:
                mat_idx, thickness, mask, target = batch
                mat_idx = mat_idx.to(model.device)
                thickness = thickness.to(model.device)
                mask = mask.to(model.device)
                target = target.to(model.device)
                
                B = mat_idx.shape[0]
                materials_onehot = F.one_hot(mat_idx, num_classes=model.vocab_size).float()
                
                timesteps = torch.randint(1, model.scheduler.timesteps + 1, (B,), dtype=torch.long, device=model.device)
                
                eps_mat = torch.randn_like(materials_onehot)
                eps_thk = torch.randn_like(thickness)
                
                x_t_mat = model.scheduler.q_sample(materials_onehot, eps_mat, timesteps)
                x_t_thk = model.scheduler.q_sample(thickness, eps_thk, timesteps)
                
                cond_emb = model._encode_spectrum(target)
                pred_mat_noise, pred_thk_noise = model.model(x_t_mat, x_t_thk, timesteps, cond_emb, mask, drop_mask=None)
                
                loss_mat = F.mse_loss(pred_mat_noise, eps_mat)
                loss_thk = F.mse_loss(pred_thk_noise, eps_thk)
                total_loss = loss_mat + loss_thk
                
                val_metrics['loss'] += total_loss.item() * B
                val_metrics['loss_mat'] += loss_mat.item() * B
                val_metrics['loss_thk'] += loss_thk.item() * B
        
        for key in val_metrics:
            val_metrics[key] /= val_split
        
        # Print epoch summary
        print(
            f"[Epoch {epoch:3d}/{args.epochs}] "
            f"Train Loss: {epoch_metrics['loss']:.6f} | "
            f"Val Loss: {val_metrics['loss']:.6f} | "
            f"Mat Acc: {epoch_metrics['mat_acc']:.3f} | "
            f"Thk MAE: {epoch_metrics['mae_thk']:.4f}"
        )
        
        history.append({
            'epoch': epoch,
            'train': epoch_metrics,
            'val': val_metrics
        })
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint = {
                'epoch': epoch,
                'model_state': model.model.state_dict(),
                'spec_encoder_state': model.spec_encoder.state_dict(),
                'optimizer_state': model.optimizer.state_dict(),
                'ema': model.ema,
                'val_loss': best_val_loss,
                'config': {
                    'vocab_size': model.vocab_size,
                    'layer_count': model.layer_count,
                    'spectrum_dim': model.spectrum_dim,
                    'hidden_dim': args.hidden_dim,
                    'material_vocab': model.ds.material_vocab,
                    'thk_min': model.ds.thk_min,
                    'thk_max': model.ds.thk_max
                }
            }
            torch.save(checkpoint, 'best_diffusion_model.pt')
            print(f"  → Saved best model (val_loss: {best_val_loss:.6f})")
        
        # Periodic checkpoint
        if epoch % args.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state': model.model.state_dict(),
                'spec_encoder_state': model.spec_encoder.state_dict(),
                'optimizer_state': model.optimizer.state_dict(),
                'ema': model.ema,
                'config': {
                    'vocab_size': model.vocab_size,
                    'layer_count': model.layer_count,
                    'spectrum_dim': model.spectrum_dim,
                    'hidden_dim': args.hidden_dim,
                    'material_vocab': model.ds.material_vocab,
                    'thk_min': model.ds.thk_min,
                    'thk_max': model.ds.thk_max
                }
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
        
        # Learning rate decay
        model.scheduler_lr.step()
    
    print("Training complete!")
    return model, history

if __name__ == '__main__':
    args = parse_args()
    model, history = train(args)
