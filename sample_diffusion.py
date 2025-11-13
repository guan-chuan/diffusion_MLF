#!/usr/bin/env python3
"""
Sampling script for trained diffusion model.

This script loads a trained diffusion model and generates structures
for given target spectra.

Usage:
    python sample_diffusion.py --checkpoint best_diffusion_model.pt --target_spectra test_spectra.npy --num_samples 10
"""

import argparse
import torch
import numpy as np
from train_diffusion import DiffusionModel, DiffusionUNet
import torch.nn as nn

def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    config = checkpoint['config']
    
    # Create model architecture
    model = DiffusionUNet(
        layer_count=config['layer_count'],
        vocab_size=config['vocab_size'],
        hidden_dim=config.get('hidden_dim', 256)
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state'])
    
    # Create spectrum encoder
    spec_encoder = nn.Sequential(
        nn.LayerNorm(config['spectrum_dim'] * 2),
        nn.Linear(config['spectrum_dim'] * 2, 256),
        nn.SiLU(),
        nn.Linear(256, 128)
    ).to(device)
    spec_encoder.load_state_dict(checkpoint['spec_encoder_state'])
    
    # Load EMA if available
    ema = checkpoint.get('ema', None)
    
    print(f"[Model] Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"[Model] Vocab size: {config['vocab_size']}, Layers: {config['layer_count']}")
    
    return model, spec_encoder, ema, config

def sample_from_target(model, spec_encoder, target_spectra, config, device='cuda', 
                       num_samples=1, guidance_w=6.0, use_ema=True, ema_state=None):
    """
    Sample structures from target spectra.
    
    Args:
        model: DiffusionUNet model
        spec_encoder: spectrum encoder
        target_spectra: (N, 2S) array of target spectra
        config: model config dict
        device: compute device
        num_samples: number of samples per target
        guidance_w: classifier-free guidance weight
        use_ema: whether to use EMA weights
        ema_state: EMA state dict
    
    Returns:
        results: list of generated structures
    """
    from train_diffusion import DDPMScheduler, sinusoidal_time_embedding
    import math
    import torch.nn.functional as F
    from tqdm import tqdm
    
    model.eval()
    
    # Apply EMA if requested
    if use_ema and ema_state is not None:
        backup = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        for n, p in model.named_parameters():
            if p.requires_grad and n in ema_state:
                p.data.copy_(ema_state[n].to(p.device))
    
    # Initialize scheduler
    scheduler = DDPMScheduler(T=1000, beta_start=1e-4, beta_end=0.02, device=device)
    
    layer_count = config['layer_count']
    vocab_size = config['vocab_size']
    material_vocab = config['material_vocab']
    thk_min = config['thk_min']
    thk_max = config['thk_max']
    
    all_results = []
    
    # Process each target spectrum
    for target_idx, target in enumerate(target_spectra):
        target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0).repeat(num_samples, 1).to(device)
        
        # Encode condition
        cond_emb = spec_encoder(target_tensor)
        
        # Start from pure noise
        x = torch.randn(num_samples, layer_count, vocab_size, device=device)
        x_thk = torch.randn(num_samples, layer_count, 1, device=device)
        layer_mask = torch.ones(num_samples, layer_count, dtype=torch.bool, device=device)
        
        # Reverse diffusion
        with torch.no_grad():
            for t in tqdm(range(scheduler.timesteps, 0, -1), 
                         desc=f"Sampling target {target_idx+1}/{len(target_spectra)}", 
                         leave=False):
                t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=device)
                
                # Predict with condition
                eps_cond_mat, eps_cond_thk = model(x, x_thk, t_tensor, cond_emb, layer_mask, drop_mask=None)
                
                # Predict without condition
                eps_uncond_mat, eps_uncond_thk = model(x, x_thk, t_tensor, cond_emb*0.0, layer_mask, drop_mask=None)
                
                # Classifier-free guidance
                eps_mat = (1.0 + guidance_w) * eps_cond_mat - guidance_w * eps_uncond_mat
                eps_thk = (1.0 + guidance_w) * eps_cond_thk - guidance_w * eps_uncond_thk
                
                # Reverse step
                beta_t = float(scheduler.betas[t].item())
                alpha_t = float(scheduler.alphas[t].item())
                alpha_bar_t = float(scheduler.alphas_cumprod[t].item())
                
                coef1 = 1.0 / math.sqrt(alpha_t)
                coef2 = beta_t / math.sqrt(1.0 - alpha_bar_t)
                
                mu_mat = coef1 * (x - coef2 * eps_mat)
                mu_thk = coef1 * (x_thk - coef2 * eps_thk)
                
                if t > 1:
                    sigma = math.sqrt(scheduler.posterior_variance(t))
                    x = mu_mat + sigma * torch.randn_like(x)
                    x_thk = mu_thk + sigma * torch.randn_like(x_thk)
                else:
                    x = mu_mat
                    x_thk = mu_thk
        
        # Convert to discrete structures
        materials_probs = F.softmax(x, dim=-1)
        mats_idx = materials_probs.argmax(dim=-1).cpu().numpy()
        thks = x_thk.squeeze(-1).cpu().numpy()
        thks = thks * (thk_max - thk_min) + thk_min
        
        # Format results
        target_results = []
        for i in range(num_samples):
            layers = []
            for j in range(layer_count):
                midx = int(mats_idx[i, j])
                matname = material_vocab[midx]
                thk_nm = float(thks[i, j])
                layers.append((matname, thk_nm))
            target_results.append(layers)
        
        all_results.append(target_results)
    
    # Restore original weights if EMA was used
    if use_ema and ema_state is not None:
        for n, p in model.named_parameters():
            if p.requires_grad and n in backup:
                p.data.copy_(backup[n].to(p.device))
    
    return all_results

def parse_args():
    p = argparse.ArgumentParser(description="Sample from trained diffusion model")
    p.add_argument('--checkpoint', required=True, help='path to model checkpoint')
    p.add_argument('--target_spectra', required=True, help='path to target spectra .npy file (N, 2S)')
    p.add_argument('--num_samples', type=int, default=10, help='number of samples per target')
    p.add_argument('--guidance_w', type=float, default=6.0, help='classifier-free guidance weight')
    p.add_argument('--device', default='cuda:0', help='compute device')
    p.add_argument('--output', default='sampled_structures.npy', help='output file path')
    p.add_argument('--use_ema', action='store_true', default=True, help='use EMA weights')
    return p.parse_args()

def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    print(f"[Device] {device}")
    
    # Load model
    print(f"[Loading] Checkpoint: {args.checkpoint}")
    model, spec_encoder, ema_state, config = load_model(args.checkpoint, device)
    
    # Load target spectra
    print(f"[Loading] Target spectra: {args.target_spectra}")
    target_spectra = np.load(args.target_spectra)
    print(f"[Target] Shape: {target_spectra.shape}")
    
    # Sample
    print(f"[Sampling] Generating {args.num_samples} samples per target...")
    results = sample_from_target(
        model=model,
        spec_encoder=spec_encoder,
        target_spectra=target_spectra,
        config=config,
        device=device,
        num_samples=args.num_samples,
        guidance_w=args.guidance_w,
        use_ema=args.use_ema,
        ema_state=ema_state
    )
    
    # Save results
    print(f"[Saving] Output: {args.output}")
    np.save(args.output, results, allow_pickle=True)
    
    # Print sample
    print("\n" + "="*60)
    print("Sample result (first target, first sample):")
    print("="*60)
    for i, (mat, thk) in enumerate(results[0][0]):
        print(f"  Layer {i+1}: {mat:12s} {thk:8.2f} nm")
    print("="*60)

if __name__ == '__main__':
    main()
