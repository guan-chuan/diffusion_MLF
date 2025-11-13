# Quick Start Guide

## 30-Second Summary

This project uses a **pure diffusion model** to generate optical multilayer structures from target spectra. No PNN integration, no gradient breaks, just clean conditional generation.

## Installation

```bash
pip install torch numpy tqdm
```

## Verify Installation

```bash
./quick_test.sh
```

This will:
- ✓ Check dataset exists
- ✓ Verify gradient flow
- ✓ Confirm everything works

## Train Model

```bash
python train_diffusion.py \
    --data optimized_dataset/optimized_multilayer_dataset.npz \
    --epochs 100 \
    --batch 128 \
    --device cuda:0
```

**Expected time**: 2-3 hours on RTX 3090 for 300k samples

## Generate Structures

```bash
python sample_diffusion.py \
    --checkpoint best_diffusion_model.pt \
    --target_spectra my_target.npy \
    --num_samples 10 \
    --output results.npy
```

## Validate Results (Optional)

```python
import torch
import numpy as np

# Load results
structures = np.load('results.npy', allow_pickle=True)

# Load PNN for validation only (not training!)
pnn = torch.load('pnn_final.pt')
pnn.eval()

# Check if structures match target spectra
for struct in structures:
    predicted = pnn(struct)
    # Compare with target...
```

## Key Points

✅ **Do this**:
- Use `train_diffusion.py` for training
- Use PNN only for post-hoc validation
- Monitor gradient flow with `test_gradient_flow.py`

❌ **Don't do this**:
- Don't use `enhanced_diffusion_model_fixed.py` (deprecated, broken gradients)
- Don't integrate PNN into training loop
- Don't create multi-term losses with physics constraints

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python train_diffusion.py --batch 64  # or 32
```

### Training Unstable
```bash
# Reduce learning rate
python train_diffusion.py --lr 5e-5
```

### Poor Sample Quality
- Train longer (200+ epochs)
- Increase model capacity: `--hidden_dim 320`
- Adjust guidance weight: `--guidance_w 8.0`

## What Changed?

**Old (broken)**:
```
Diffusion + Frozen PNN → Multi-term loss → Gradient breaks 💔
```

**New (working)**:
```
Pure Diffusion + Trainable Encoder → Simple loss → Clean gradients ✅
```

## Documentation

- **Quick overview**: [README.md](README.md)
- **Technical details**: [README_DIFFUSION.md](README_DIFFUSION.md)
- **Why redesign?**: [ARCHITECTURE_REDESIGN.md](ARCHITECTURE_REDESIGN.md)
- **Changes**: [CHANGELOG.md](CHANGELOG.md)

## Support

If something doesn't work:

1. Check `test_gradient_flow.py` output
2. Read [README_DIFFUSION.md](README_DIFFUSION.md) troubleshooting section
3. Verify you're using `train_diffusion.py`, not old scripts

## Example Workflow

```bash
# 1. Verify setup
./quick_test.sh

# 2. Train (takes ~2-3 hours)
python train_diffusion.py \
    --data optimized_dataset/optimized_multilayer_dataset.npz \
    --epochs 100 \
    --batch 128

# 3. Sample
python sample_diffusion.py \
    --checkpoint best_diffusion_model.pt \
    --target_spectra test_target.npy \
    --num_samples 20

# 4. Validate with PNN (optional)
python validate_with_pnn.py  # (you'd need to write this)
```

## Performance Expectations

After 100 epochs:
- Loss: < 0.005
- Material accuracy: > 0.80
- Thickness MAE: < 5 nm

These are monitored automatically during training.

---

That's it! Read [README.md](README.md) for more details.
