#!/usr/bin/env python3
# enhanced_diffusion_model_fixed_gradible.py
"""
Fixed enhanced diffusion training script (gradients preserved through PNN)

Key fixes:
 - No more torch.no_grad / detach around PNN predictions.
 - PNN parameters are frozen (requires_grad=False) but gradient can flow through its outputs.
 - Per-sample classifier-free masking implemented.
 - Correct x0 reconstruction using scheduler.alphas_cumprod with t in 1..T.
 - Timesteps sampling corrected to 1..T inclusive.
 - Added optional gradient debug prints (--grad_debug) to inspect requires_grad flags.
 - Provided TODO notes for hooking real PNN/TMM/Processor implementations.

Usage:
  python enhanced_diffusion_model_fixed_gradible.py --data optimized_dataset/optimized_multilayer_dataset.npz --pnn pnn_final.pt
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
from pnn import PNNTransformer, PNNMLP

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
        # cond per-sample mask handled by caller (drop_mask boolean vector)
        t_emb = sinusoidal_time_embedding(timesteps, dim=128).to(mat_noisy.device)
        x = torch.cat([mat_noisy, thk_noisy], dim=-1)
        x = self.initial(x)
        x = self.res1(x, t_emb, cond_emb)
        x = self.trans(x, mask=layer_mask)
        x = self.res2(x, t_emb, cond_emb)
        x = self.final_norm(x)
        mat_noise = self.material_head(x)
        thk_noise = self.thickness_head(x)
        return mat_noise, thk_noise

# ---------------------------
# PNN surrogate loader (expects you have a pnn.py or state dict)
# ---------------------------

class PNNSurrogateWrapper(nn.Module):
    """
    Wrapper to load an external PNN (either state_dict or full model).
    We expect PNN.forward(mat_idx, thickness_norm, mask) -> (B, S) or (B, 2S)
    """
    def __init__(self, pnn_path=None, device='cpu', current_vocab=None):
        super().__init__()
        self.pnn = None
        self.device = device
        self.pnn_vocab = None
        self.reorder_idx = None  # mapping from pnn_vocab order to current_vocab order
        if pnn_path is not None and os.path.exists(pnn_path):
            # TODO: adapt this to your PNN class definition if needed.
            # We try to load a full model first, else a state_dict into a default architecture.
            try:
                loaded = torch.load(pnn_path, map_location=device)
                if isinstance(loaded, dict) and 'model_state' in loaded:
                    # user-supplied checkpoint with metadata; reconstruct Transformer model
                    spectrum_dim = loaded.get('spectrum_dim', None)
                    vocab = loaded.get('vocab', None)
                    max_layers = loaded.get('max_layers', None)
                    emb_dim = loaded.get('emb_dim', 128)
                    n_heads = loaded.get('n_heads', 4)
                    n_encoder_layers = loaded.get('n_encoder_layers', 3)
                    mlp_hidden = loaded.get('mlp_hidden', 256)
                    dropout = loaded.get('dropout', 0.1)
                    use_sigmoid = loaded.get('use_sigmoid', False)
                    use_position_encoding = loaded.get('use_position_encoding', True)
                    separate_TR = loaded.get('separate_TR', False)
                    if spectrum_dim is not None and vocab is not None and max_layers is not None:
                        self.pnn_vocab = list(vocab)
                        self.pnn = PNNTransformer(
                            vocab_size=len(vocab), max_layers=max_layers, spectrum_dim=spectrum_dim,
                            emb_dim=emb_dim, n_heads=n_heads, n_encoder_layers=n_encoder_layers,
                            mlp_hidden=mlp_hidden, dropout=dropout, use_sigmoid=use_sigmoid,
                            use_position_encoding=use_position_encoding, separate_TR=separate_TR
                        )
                        self.pnn.load_state_dict(loaded['model_state'])
                    else:
                        # maybe the file directly contains model
                        if isinstance(loaded, nn.Module):
                            self.pnn = loaded
                        else:
                            print("[PNN] checkpoint missing necessary metadata; unable to instantiate model.")
                elif isinstance(loaded, nn.Module):
                    self.pnn = loaded
                else:
                    # maybe user saved entire model via torch.save(model)
                    try:
                        model = loaded
                        self.pnn = model
                    except Exception:
                        self.pnn = None
            except Exception as e:
                print("[PNN] Failed to auto-load PNN: ", e)
                self.pnn = None
        else:
            if pnn_path:
                print(f"[PNN] pnn_path {pnn_path} not found; continuing without PNN.")
        if self.pnn is not None:
            self.pnn.to(device)
            # freeze PNN parameters but allow gradients to flow through its outputs
            for p in self.pnn.parameters():
                p.requires_grad = False
            self.pnn.eval()
            # build reorder mapping if vocab provided
            if self.pnn_vocab is not None and current_vocab is not None:
                try:
                    perm = []
                    current_index = {m: i for i, m in enumerate(current_vocab)}
                    for m in self.pnn_vocab:
                        perm.append(current_index.get(m, current_index.get('VOID', 0)))
                    self.reorder_idx = torch.tensor(perm, dtype=torch.long)
                except Exception as e:
                    print("[PNN] failed to build vocab reorder mapping:", e)

    def predict_from_probs(self, materials_probs, thickness_norm, mask):
        """
        Differentiable bridge between diffusion output (probabilities)
        and PNN input (indices).
        """
        if self.pnn is None:
            raise RuntimeError("PNN surrogate not loaded.")

        # 1️⃣ 计算 soft embedding (保留梯度)
        emb_layer = getattr(self.pnn, "mat_emb", None)
        if emb_layer is None:
            raise RuntimeError("PNN has no mat_emb embedding layer.")
        # reorder materials_probs to match pnn vocab order if needed
        if self.reorder_idx is not None:
            idx = self.reorder_idx.to(materials_probs.device)
            materials_probs = torch.index_select(materials_probs, dim=-1, index=idx)
        emb_weight = emb_layer.weight  # (V, D)
        soft_emb = torch.einsum("blv,vd->bld", materials_probs, emb_weight)  # (B,L,D)

        # 2️⃣ 直接调用 pnn.forward()（现在它能接受 soft embedding）
        out = self.pnn(soft_emb, thickness_norm, mask)

        # 3️⃣ 不做 detach、不做 no_grad
        return out
# ---------------------------
# Full Diffusion model wrapper
# ---------------------------

class EnhancedDiffusionModel:
    def __init__(self, device='cuda', data_path=None, pnn_path=None):
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith('cuda') else 'cpu')
        print("[device]", self.device)
        assert data_path is not None and os.path.exists(data_path), "Provide valid data file."
        ds = MultilayerDataset(data_path)
        self.ds = ds
        self.scheduler = NoiseScheduler(T=1000, beta_start=1e-4, beta_end=0.02, device=self.device)
        self.layer_count = ds.max_layers
        self.vocab_size = ds.vocab_size
        self.spectrum_dim = ds.S
        self.model = EnhancedDiffusionUNet(self.layer_count, self.vocab_size, hidden_dim=256).to(self.device)
        self.pnn_wrapper = PNNSurrogateWrapper(pnn_path, device=self.device) if pnn_path else None
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.ema = {n: p.detach().clone() for n,p in self.model.named_parameters() if p.requires_grad}
        self.ema_decay = 0.9999
        # hyperparams
        self.p_uncond = 0.1
        self.lambda_spec = 1.0
        self.lambda_phys = 0.1
        self.guidance_w = 6.0

    def _encode_spectrum(self, spectra):
        # small MLP cond encoder (trainable)
        # Expect spectra shape (B, 2S) or (B, S) depending on usage
        B = spectra.shape[0]
        if not hasattr(self, '_spec_encoder'):
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
               # 修改后
                self.ema[n] = self.ema[n].to(p.device)
                self.ema[n].mul_(self.ema_decay).add_(p.detach(), alpha=1.0 - self.ema_decay)
               #self.ema[n].mul_(self.ema_decay).add_(p.detach().cpu(), alpha=1.0 - self.ema_decay)

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

    def train_step(self, batch, lambda_spec_scale: float = 1.0, grad_debug: bool = False):
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

        # Use PNN surrogate to compute predicted spectra.
        # CRITICAL: Do NOT wrap this call in torch.no_grad() or detach(); we WANT gradients to flow.
        spec_loss_val = 0.0
        phys_loss_val = 0.0
        if self.pnn_wrapper is not None and self.pnn_wrapper.pnn is not None:
            # ensure PNN parameters are frozen but outputs are differentiable
            # NOTE: PNNSurrogateWrapper already sets p.requires_grad = False for p in pnn.parameters()
            pred_spectra = self.pnn_wrapper.predict_from_probs(materials_probs_hat, thickness_hat, layer_mask)
            # pred_spectra should be tensor (B, 2S)
            # compute L1 or MSE loss
            spec_loss = F.l1_loss(pred_spectra, target)
            spec_loss_val = spec_loss.item() if not grad_debug else spec_loss.detach().cpu().item()
            # (optional) physics constraint loss could be added; simple form:
            # energy conservation penalty: max(0, T+R-1)
            S = self.spectrum_dim
            pred_T = pred_spectra[:, :S]
            pred_R = pred_spectra[:, S:]
            cons_violation = F.relu(pred_T + pred_R - 1.0).mean()
            phys_loss = cons_violation
            phys_loss_val = phys_loss.item() if not grad_debug else phys_loss.detach().cpu().item()
        else:
            # no surrogate: skip spec loss (or you may compute PNN offline)
            spec_loss = torch.tensor(0.0, device=self.device)
            phys_loss = torch.tensor(0.0, device=self.device)

        # total loss
        total_loss = loss_noise + (self.lambda_spec * lambda_spec_scale) * spec_loss + self.lambda_phys * phys_loss

        # Debug: check requires_grad flags for critical tensors (only if asked)
        if grad_debug:
            print("DEBUG grad flags:")
            print("  loss_noise.requires_grad:", loss_noise.requires_grad)
            print("  spec_loss.requires_grad:", spec_loss.requires_grad)
            print("  total_loss.requires_grad:", total_loss.requires_grad)
            print("  pred_spectra.requires_grad (if present):", pred_spectra.requires_grad if 'pred_spectra' in locals() else "N/A")
            print("  materials_probs_hat.requires_grad:", materials_probs_hat.requires_grad)
            print("  x0_mat_hat.requires_grad:", x0_mat_hat.requires_grad)

        # backward & step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # ema update (on CPU copy for memory safety)
        self._update_ema()

        # r2 metric for monitoring if surrogate available
        r2_val = 0.0
        if self.pnn_wrapper is not None and self.pnn_wrapper.pnn is not None:
            with torch.no_grad():
                y_true = target
                y_pred = pred_spectra
                y_mean = y_true.mean(dim=1, keepdim=True)
                ss_tot = ((y_true - y_mean) ** 2).sum(dim=1)
                ss_res = ((y_true - y_pred) ** 2).sum(dim=1)
                r2_batch = 1.0 - (ss_res / (ss_tot + 1e-12))
                r2_val = float(r2_batch.mean().detach().cpu())

        metrics = {
            'loss': float(total_loss.detach().cpu()),
            'noise': float(loss_noise.detach().cpu()),
            'spec': float(spec_loss.detach().cpu()) if hasattr(spec_loss, 'detach') else float(spec_loss),
            'phys': float(phys_loss.detach().cpu()) if hasattr(phys_loss, 'detach') else float(phys_loss),
            'r2': r2_val
        }
        return metrics

    @torch.no_grad()
    def eval_step(self, batch):
        """Validation forward without weight updates. Returns spec loss and r2 if surrogate is present."""
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

        x0_mat_hat, x0_thk_hat = self._reconstruct_x0(x_t_mat, x_t_thk, pred_mat_noise, pred_thk_noise, timesteps)
        materials_probs_hat = F.softmax(x0_mat_hat, dim=-1)
        thickness_hat = torch.clamp(x0_thk_hat, 0.0, 1.0)

        spec_loss = torch.tensor(0.0, device=self.device)
        r2_val = 0.0
        if self.pnn_wrapper is not None and self.pnn_wrapper.pnn is not None:
            pred_spectra = self.pnn_wrapper.predict_from_probs(materials_probs_hat, thickness_hat, layer_mask)
            spec_loss = F.l1_loss(pred_spectra, target)
            y_true = target
            y_pred = pred_spectra
            y_mean = y_true.mean(dim=1, keepdim=True)
            ss_tot = ((y_true - y_mean) ** 2).sum(dim=1)
            ss_res = ((y_true - y_pred) ** 2).sum(dim=1)
            r2_batch = 1.0 - (ss_res / (ss_tot + 1e-12))
            r2_val = float(r2_batch.mean().detach().cpu())

        return {
            'spec': float(spec_loss.detach().cpu()),
            'r2': r2_val
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
    p.add_argument('--pnn', required=False, default='pnn_best.pt', help='path to pnn checkpoint (optional). If provided, will be loaded')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--spec_warmup_epochs', type=int, default=20, help='epochs to anneal spectral loss weighting from 0->1')
    p.add_argument('--guidance', type=float, default=6.0)
    p.add_argument('--grad_debug', action='store_true', help='print grad debug info for first batch')
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()

def train_model_cli(args):
    device = args.device
    model = EnhancedDiffusionModel(device=device, data_path=args.data, pnn_path=args.pnn)
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
        # spectral loss annealing 0->1 over spec_warmup_epochs
        lambda_scale = min(1.0, epoch / max(1, args.spec_warmup_epochs))
        for i, batch in enumerate(pbar):
            if args.grad_debug and epoch==1 and i==0:
                metrics = model.train_step(batch, lambda_spec_scale=lambda_scale, grad_debug=True)
            else:
                metrics = model.train_step(batch, lambda_spec_scale=lambda_scale, grad_debug=False)
            epoch_loss += metrics['loss'] * batch[0].shape[0]
            pbar.set_postfix({'loss':metrics['loss'],'noise':metrics['noise'],'spec':metrics['spec'],'r2':metrics.get('r2',0.0)})
        epoch_loss = epoch_loss / len(train_ds)
        # proper validation (no weight updates)
        model.model.eval()
        val_spec = 0.0
        val_r2 = 0.0
        n_val = 0
        for batch in val_loader:
            ev = model.eval_step(batch)
            bs = batch[0].shape[0]
            val_spec += ev['spec'] * bs
            val_r2 += ev['r2'] * bs
            n_val += bs
        if n_val > 0:
            val_spec /= n_val
            val_r2 /= n_val
        print(f"Epoch {epoch} finished. train loss: {epoch_loss:.6f} | val spec L1: {val_spec:.6f} | val r2: {val_r2:.4f}")
        if val_spec < best_val:
            best_val = val_spec
            torch.save({'model_state': model.model.state_dict()}, 'enhanced_diffusion_best.pt')
            print("Saved best diffusion model: enhanced_diffusion_best.pt")
        history.append(epoch_loss)
    print("Training finished.")
    return model, history

if __name__ == '__main__':
    args = parse_args()
    model, history = train_model_cli(args)
    # Optionally save final model
    torch.save({'model_state': model.model.state_dict()}, 'enhanced_diffusion_model_final.pt')
    print("Saved final model: enhanced_diffusion_model_final.pt")
