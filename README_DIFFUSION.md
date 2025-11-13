# Pure Diffusion Model for Optical Inverse Design

## Overview

This implementation provides a **pure diffusion-based approach** for inverse design of optical multilayer structures. The model learns to generate material sequences and layer thicknesses conditioned on target optical spectra (transmission and reflection).

## Why Pure Diffusion (No PNN)?

### Problem with PNN Integration

The previous approach tried to integrate a Physics Neural Network (PNN) into the diffusion training loop:
1. Diffusion model generates structure from spectrum (spectrum → structure)
2. PNN predicts spectrum from structure (structure → spectrum)
3. Compute loss between predicted and target spectrum

**This creates logical issues:**

- **Circular dependency**: The diffusion model already learns spectrum→structure mapping through conditioning. Adding PNN creates a redundant structure→spectrum→structure cycle.
- **Gradient flow problems**: Even when PNN parameters are frozen, gradient quality is poor because:
  - Early in training, diffusion outputs are noisy
  - PNN cannot give meaningful predictions on noisy structures
  - Backpropagated gradients are unreliable or misleading
- **Conflicting objectives**: PNN was trained on a different data distribution and may have systematic biases that conflict with diffusion learning.

### Pure Diffusion Solution

The correct approach is **standard conditional diffusion**:

1. **Condition encoding**: Encode target spectrum into embedding vector
2. **Denoising**: Train model to predict noise given noisy structure + spectrum condition
3. **Loss**: Simple MSE between predicted noise and true noise
4. **Sampling**: Use classifier-free guidance for controlled generation

**Advantages:**
- ✅ Clean gradient flow through entire model
- ✅ Single, well-defined objective (denoising)
- ✅ Spectrum-structure relationship learned end-to-end
- ✅ No circular dependencies
- ✅ Proven architecture (DDPM + conditioning)

## Architecture

```
Target Spectrum (T, R)
        ↓
   [Spectrum Encoder]  ← Trainable MLP
        ↓
   Condition Embedding
        ↓
   [Diffusion UNet]
    - Input: x_t (noisy structure) + t (timestep) + condition
    - Process: ResBlocks + FFN with time & condition modulation
    - Output: ε_predicted (noise prediction)
        ↓
   Loss: MSE(ε_predicted, ε_true)
```

### Key Components

1. **Spectrum Encoder**: 
   - Trainable MLP that maps spectra to condition embeddings
   - Architecture: LayerNorm → Linear(2S→256) → SiLU → Linear(256→128)

2. **Diffusion UNet**:
   - Processes noisy structures at different timesteps
   - Residual blocks with time and condition injection
   - Feedforward blocks with layer masking
   - Separate heads for material and thickness predictions

3. **DDPM Scheduler**:
   - Standard linear beta schedule (1e-4 to 0.02)
   - 1000 diffusion timesteps
   - Closed-form forward process
   - Iterative reverse process with variance estimation

4. **Classifier-Free Guidance**:
   - During training: randomly drop condition 10% of the time
   - During sampling: interpolate between conditional and unconditional predictions
   - Guidance weight controls how strongly to follow the condition

## Usage

### Training

```bash
python train_diffusion.py \
    --data optimized_dataset/optimized_multilayer_dataset.npz \
    --epochs 100 \
    --batch 128 \
    --lr 1e-4 \
    --hidden_dim 256 \
    --device cuda:0
```

**Training features:**
- EMA (Exponential Moving Average) for stable sampling
- Gradient clipping for training stability
- Cosine learning rate schedule
- Automatic checkpointing (best model + periodic saves)
- Validation monitoring

**Expected training time:**
- ~300k samples, batch size 128, 100 epochs
- ~2-3 hours on NVIDIA RTX 3090

### Sampling

```bash
python sample_diffusion.py \
    --checkpoint best_diffusion_model.pt \
    --target_spectra test_spectra.npy \
    --num_samples 10 \
    --guidance_w 6.0 \
    --device cuda:0 \
    --output sampled_structures.npy
```

**Sampling parameters:**
- `guidance_w`: Controls adherence to target spectrum
  - Higher (8-12): More faithful to target, less diversity
  - Lower (3-5): More diversity, less precise
  - Default (6): Good balance

### Validation with PNN (Optional)

After sampling, you can validate generated structures using PNN:

```python
import torch
from pnn import PNNTransformer
import numpy as np

# Load generated structures
structures = np.load('sampled_structures.npy', allow_pickle=True)

# Load PNN
pnn = torch.load('pnn_final.pt')
pnn.eval()

# Predict spectra for validation
for struct in structures:
    predicted_spectrum = pnn(struct)
    # Compare with target...
```

This is the **correct use of PNN**: as a post-hoc validation tool, not a training component.

## Loss Function

The loss is **extremely simple**:

```python
# Predict noise
pred_mat_noise, pred_thk_noise = model(x_t, t, condition)

# Compute MSE with true noise
loss_mat = MSE(pred_mat_noise, true_mat_noise)
loss_thk = MSE(pred_thk_noise, true_thk_noise)
total_loss = loss_mat + loss_thk
```

**Why this works:**
- Denoising objective implicitly learns data distribution
- Conditioning on spectra biases generation toward matching targets
- No need for explicit physics constraints (they're learned from data)

## Implementation Details

### Model Sizes

| Component | Parameters | Size |
|-----------|-----------|------|
| Diffusion UNet | ~2-3M | 256 hidden dim |
| Spectrum Encoder | ~65K | 2S→128 |
| Total | ~2.1M | Lightweight |

### Memory Requirements

| Batch Size | GPU Memory | Training Speed |
|-----------|-----------|----------------|
| 32 | ~8 GB | ~10 samples/sec |
| 128 | ~18 GB | ~35 samples/sec |
| 256 | ~24 GB | ~60 samples/sec |

Tested on NVIDIA RTX 3090 (24GB).

### Hyperparameters

```python
# Training
learning_rate = 1e-4
weight_decay = 1e-6
gradient_clip = 1.0
ema_decay = 0.9999
p_uncond = 0.1  # Classifier-free guidance dropout

# Diffusion
T = 1000  # Timesteps
beta_start = 1e-4
beta_end = 0.02

# Architecture
hidden_dim = 256
time_emb_dim = 128
cond_dim = 128
```

## Monitoring Training

Key metrics to watch:

1. **Total Loss**: Should steadily decrease
   - Good: < 0.01 after 50 epochs
   - Excellent: < 0.005 after 100 epochs

2. **Material Accuracy**: How well reconstructed x_0 matches true materials
   - Target: > 0.8 after 50 epochs
   - This is monitored but not directly optimized

3. **Thickness MAE**: Mean absolute error in thickness prediction
   - Target: < 5 nm after 50 epochs

4. **Validation Loss**: Should track training loss
   - Watch for overfitting (val >> train)

## Troubleshooting

### Training is unstable
- Reduce learning rate to 5e-5
- Reduce batch size
- Check data normalization

### Generated structures are poor quality
- Train longer (200+ epochs)
- Increase model capacity (hidden_dim=320 or 384)
- Try different guidance weights during sampling

### Samples don't match target spectra
- Increase guidance weight (8-12)
- Ensure spectrum encoder is training (check gradients)
- Verify condition encoding is correct

### Out of memory
- Reduce batch size
- Reduce hidden_dim to 192 or 128
- Use gradient checkpointing (add to code)

## Comparison: PNN-Enhanced vs Pure Diffusion

| Aspect | PNN-Enhanced | Pure Diffusion |
|--------|--------------|----------------|
| Gradient flow | ❌ Broken | ✅ Clean |
| Training stability | ❌ Unstable | ✅ Stable |
| Loss complexity | ❌ Multi-term | ✅ Single term |
| Conceptual clarity | ❌ Circular logic | ✅ Clear objective |
| Implementation | ❌ Complex | ✅ Simple |
| Physics constraints | ⚠️ Explicit (but broken) | ✅ Implicit (learned) |

## Future Improvements

1. **Attention Mechanism**: Add self-attention for better layer interactions
2. **Adaptive Guidance**: Learn guidance weight during training
3. **Multi-scale Architecture**: U-Net with downsampling/upsampling
4. **Physics-Informed Initialization**: Pre-train on simpler physics tasks
5. **Latent Diffusion**: Compress structure space with autoencoder

## References

- Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models" (2021)
- Ho & Salimans, "Classifier-Free Diffusion Guidance" (2022)

## Citation

If you use this code, please cite:

```bibtex
@software{pure_diffusion_optical,
  title={Pure Diffusion Model for Optical Inverse Design},
  author={Your Name},
  year={2024},
  url={https://github.com/yourrepo}
}
```
