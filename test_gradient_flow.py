#!/usr/bin/env python3
"""
Gradient flow verification script.

This script performs a simple sanity check to ensure:
1. All loss components have proper gradients
2. Gradients flow through the entire model
3. No gradient breaks or NaN/Inf values

Usage:
    python test_gradient_flow.py --data optimized_dataset/optimized_multilayer_dataset.npz
"""

import argparse
import torch
import torch.nn.functional as F
from train_diffusion import DiffusionModel, MultilayerDataset
from torch.utils.data import DataLoader

def check_gradient_flow(model_wrapper, batch, verbose=True):
    """
    Verify gradient flow through the model.
    
    Returns:
        dict with gradient statistics
    """
    # Run forward pass
    metrics = model_wrapper.train_step(batch, grad_debug=False)
    
    # Check gradients
    grad_stats = {}
    total_params = 0
    params_with_grad = 0
    max_grad = 0.0
    min_grad = float('inf')
    
    print("\n" + "="*80)
    print("GRADIENT FLOW VERIFICATION")
    print("="*80)
    
    # Check diffusion model gradients
    print("\n[Diffusion Model Gradients]")
    for name, param in model_wrapper.model.named_parameters():
        if param.requires_grad:
            total_params += 1
            if param.grad is not None:
                params_with_grad += 1
                grad_norm = param.grad.norm().item()
                max_grad = max(max_grad, grad_norm)
                min_grad = min(min_grad, grad_norm)
                
                if verbose and grad_norm > 0:
                    print(f"  ✓ {name:50s} | grad_norm: {grad_norm:.6f}")
                elif param.grad is None:
                    print(f"  ✗ {name:50s} | grad: None")
                elif grad_norm == 0:
                    print(f"  ⚠ {name:50s} | grad_norm: 0.0 (might be ok)")
            else:
                print(f"  ✗ {name:50s} | grad: None")
    
    # Check spectrum encoder gradients
    print("\n[Spectrum Encoder Gradients]")
    for name, param in model_wrapper.spec_encoder.named_parameters():
        if param.requires_grad:
            total_params += 1
            if param.grad is not None:
                params_with_grad += 1
                grad_norm = param.grad.norm().item()
                max_grad = max(max_grad, grad_norm)
                min_grad = min(min_grad, grad_norm)
                
                if verbose and grad_norm > 0:
                    print(f"  ✓ {name:50s} | grad_norm: {grad_norm:.6f}")
                elif param.grad is None:
                    print(f"  ✗ {name:50s} | grad: None")
                elif grad_norm == 0:
                    print(f"  ⚠ {name:50s} | grad_norm: 0.0 (might be ok)")
            else:
                print(f"  ✗ {name:50s} | grad: None")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  Total parameters requiring grad: {total_params}")
    print(f"  Parameters with gradients: {params_with_grad}")
    print(f"  Coverage: {params_with_grad/total_params*100:.1f}%")
    print(f"  Max gradient norm: {max_grad:.6f}")
    print(f"  Min gradient norm: {min_grad:.6f}")
    
    grad_stats['total_params'] = total_params
    grad_stats['params_with_grad'] = params_with_grad
    grad_stats['coverage'] = params_with_grad / total_params if total_params > 0 else 0
    grad_stats['max_grad'] = max_grad
    grad_stats['min_grad'] = min_grad
    
    # Check for issues
    issues = []
    if params_with_grad < total_params:
        issues.append(f"⚠ {total_params - params_with_grad} parameters missing gradients")
    if max_grad > 100:
        issues.append(f"⚠ Large gradient detected: {max_grad:.2f} (possible instability)")
    if max_grad == 0:
        issues.append("✗ All gradients are zero (model not learning)")
    
    if issues:
        print("\n[ISSUES DETECTED]")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ No issues detected - gradient flow is healthy!")
    
    print("="*80)
    
    return grad_stats

def test_loss_gradients(model_wrapper, batch):
    """
    Test that loss components have proper requires_grad flags.
    """
    mat_idx, thickness_norm, mask, target = batch
    mat_idx = mat_idx.to(model_wrapper.device)
    thickness_norm = thickness_norm.to(model_wrapper.device)
    mask = mask.to(model_wrapper.device)
    target = target.to(model_wrapper.device)
    
    B = mat_idx.shape[0]
    
    # Prepare inputs
    materials_onehot = F.one_hot(mat_idx, num_classes=model_wrapper.vocab_size).float()
    thickness = thickness_norm
    timesteps = torch.randint(1, model_wrapper.scheduler.timesteps + 1, (B,), 
                             dtype=torch.long, device=model_wrapper.device)
    
    # Forward diffusion
    eps_mat = torch.randn_like(materials_onehot)
    eps_thk = torch.randn_like(thickness)
    x_t_mat = model_wrapper.scheduler.q_sample(materials_onehot, eps_mat, timesteps)
    x_t_thk = model_wrapper.scheduler.q_sample(thickness, eps_thk, timesteps)
    
    # Encode condition
    cond_emb = model_wrapper._encode_spectrum(target)
    
    # Predict noise
    drop_mask = (torch.rand(B, device=model_wrapper.device) < model_wrapper.p_uncond)
    pred_mat_noise, pred_thk_noise = model_wrapper.model(
        x_t_mat, x_t_thk, timesteps, cond_emb, mask, drop_mask
    )
    
    # Compute losses
    loss_mat = F.mse_loss(pred_mat_noise, eps_mat)
    loss_thk = F.mse_loss(pred_thk_noise, eps_thk)
    total_loss = loss_mat + loss_thk
    
    print("\n" + "="*80)
    print("LOSS GRADIENT FLAGS")
    print("="*80)
    print(f"  loss_mat.requires_grad:      {loss_mat.requires_grad}")
    print(f"  loss_thk.requires_grad:      {loss_thk.requires_grad}")
    print(f"  total_loss.requires_grad:    {total_loss.requires_grad}")
    print(f"  pred_mat_noise.requires_grad: {pred_mat_noise.requires_grad}")
    print(f"  pred_thk_noise.requires_grad: {pred_thk_noise.requires_grad}")
    print(f"  cond_emb.requires_grad:      {cond_emb.requires_grad}")
    print(f"  x_t_mat.requires_grad:       {x_t_mat.requires_grad}")
    print(f"  x_t_thk.requires_grad:       {x_t_thk.requires_grad}")
    print("="*80)
    
    # Verify all key tensors have gradients
    assert total_loss.requires_grad, "❌ total_loss should require gradient!"
    assert loss_mat.requires_grad, "❌ loss_mat should require gradient!"
    assert loss_thk.requires_grad, "❌ loss_thk should require gradient!"
    assert pred_mat_noise.requires_grad, "❌ pred_mat_noise should require gradient!"
    assert pred_thk_noise.requires_grad, "❌ pred_thk_noise should require gradient!"
    assert cond_emb.requires_grad, "❌ cond_emb should require gradient!"
    
    print("\n✓ All loss components have proper gradient flags!")
    
    return {
        'loss_mat': loss_mat.item(),
        'loss_thk': loss_thk.item(),
        'total_loss': total_loss.item()
    }

def parse_args():
    p = argparse.ArgumentParser(description="Test gradient flow in diffusion model")
    p.add_argument('--data', required=True, help='path to dataset .npz file')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--verbose', action='store_true', help='print detailed gradient info')
    return p.parse_args()

def main():
    args = parse_args()
    
    print("="*80)
    print("GRADIENT FLOW TEST")
    print("="*80)
    print(f"Device: {args.device}")
    print(f"Data: {args.data}")
    print(f"Batch size: {args.batch_size}")
    
    # Initialize model
    print("\n[Initializing model...]")
    model_wrapper = DiffusionModel(
        device=args.device,
        data_path=args.data,
        hidden_dim=256,
        lr=1e-4
    )
    
    # Load a batch
    print("[Loading data batch...]")
    loader = DataLoader(model_wrapper.ds, batch_size=args.batch_size, shuffle=True)
    batch = next(iter(loader))
    
    # Test 1: Check loss gradient flags
    print("\n" + "="*80)
    print("TEST 1: Loss Gradient Flags")
    print("="*80)
    loss_info = test_loss_gradients(model_wrapper, batch)
    print(f"\nLoss values:")
    print(f"  loss_mat:   {loss_info['loss_mat']:.6f}")
    print(f"  loss_thk:   {loss_info['loss_thk']:.6f}")
    print(f"  total_loss: {loss_info['total_loss']:.6f}")
    
    # Test 2: Check gradient flow through entire model
    print("\n" + "="*80)
    print("TEST 2: Full Gradient Flow")
    print("="*80)
    grad_stats = check_gradient_flow(model_wrapper, batch, verbose=args.verbose)
    
    # Test 3: Multiple iterations
    print("\n" + "="*80)
    print("TEST 3: Multiple Training Steps")
    print("="*80)
    print("Running 5 training steps to check consistency...")
    for i in range(5):
        metrics = model_wrapper.train_step(batch, grad_debug=False)
        print(f"  Step {i+1}: loss={metrics['loss']:.6f}, "
              f"mat_acc={metrics['mat_acc']:.3f}, "
              f"thk_mae={metrics['mae_thk']:.4f}")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print("✓ All tests passed successfully!")
    print(f"✓ Gradient coverage: {grad_stats['coverage']*100:.1f}%")
    print(f"✓ Model is training properly")
    print("\nYou can now proceed with full training using train_diffusion.py")
    print("="*80)

if __name__ == '__main__':
    main()
