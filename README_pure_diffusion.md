# Pure Diffusion Model for Optical Multilayer Inverse Design

## Overview

This is a pure diffusion-based model for inverse design of optical multilayer structures, **without** PNN (Physics Neural Network) components. The model was created by modifying `enhanced_diffusion_model_fixed.py` to remove all PNN-related code and use only diffusion loss.

## Key Changes from Original

### Removed Components
- **PNN Surrogate**: Completely removed `PNNSurrogateWrapper` class and all PNN-related imports
- **Physics Loss**: Removed physical constraint losses (energy conservation, etc.)
- **Spectral Loss**: Removed L1/MSE spectral matching losses that required PNN predictions
- **PNN Dependencies**: Removed `from pnn import PNNTransformer, PNNMLP` import

### Simplified Architecture
- **Pure Diffusion Loss**: Only MSE noise prediction loss for materials and thicknesses
- **Conditional Generation**: Uses target spectra as conditioning via trainable encoder
- **Classifier-Free Guidance**: Maintained for flexible sampling control
- **EMA Updates**: Kept for stable sampling

### Training Modifications
- **Loss Function**: Simple `loss_mat + loss_thk` (MSE between predicted and true noise)
- **Metrics**: Material accuracy and thickness MAE for monitoring
- **Validation**: Same metrics as training (no PNN-based R² or spectral loss)
- **Learning Rate**: Added cosine annealing scheduler

## Model Architecture

```
Input: (noisy_material_probs, noisy_thickness, timestep, spectrum_condition)
  ↓
EnhancedDiffusionUNet
  ↓
Output: (predicted_material_noise, predicted_thickness_noise)
```

### Components
1. **NoiseScheduler**: Standard DDPM linear noise schedule
2. **EnhancedDiffusionUNet**: 
   - Input projection (vocab_size + 1 → hidden_dim)
   - Residual blocks with time/condition embeddings
   - Transformer block for layer interactions
   - Separate heads for material and thickness noise prediction
3. **Spectrum Encoder**: Trainable encoder for target spectra conditioning
4. **EMA**: Exponential moving average for stable sampling

## Usage

### Basic Training
```bash
python pure_diffusion_model.py \
    --data optimized_dataset/optimized_multilayer_dataset.npz \
    --epochs 200 \
    --batch 128 \
    --lr 1e-4 \
    --device cuda:0
```

### With Gradient Debugging
```bash
python pure_diffusion_model.py \
    --data optimized_dataset/optimized_multilayer_dataset.npz \
    --epochs 10 \
    --batch 32 \
    --grad_debug \
    --device cpu
```

### Sampling from Trained Model
```python
from pure_diffusion_model import PureDiffusionModel

# Load trained model
model = PureDiffusionModel(device='cuda', data_path='path/to/dataset.npz')
model.model.load_state_dict(torch.load('pure_diffusion_best.pt')['model_state'])

# Generate structures
target_spectra = torch.randn(5, 2 * model.spectrum_dim)  # 5 samples
results = model.sample(target_spectra, num_samples=5, use_ema=True)
```

## Key Features

1. **Pure Diffusion**: No auxiliary losses, only noise prediction
2. **Conditional Generation**: Target spectra guide the generation process
3. **Classifier-Free Guidance**: Enables control over generation quality/diversity
4. **Stable Training**: EMA and gradient clipping for robust optimization
5. **Flexible Architecture**: Supports variable number of layers and materials

## File Structure

- `pure_diffusion_model.py`: Main training script
- `test_pure_diffusion.py`: Basic functionality test
- `pure_diffusion_best.pt`: Best model checkpoint (saved during training)
- `pure_diffusion_model_final.pt`: Final model checkpoint

## Advantages of Pure Diffusion Approach

1. **Simpler Training**: No need for pre-trained PNN or complex loss balancing
2. **End-to-End**: Directly learns the mapping from spectra to structures
3. **More General**: Not constrained by PNN's approximation capabilities
4. **Faster**: No forward pass through PNN during training
5. **Better Scaling**: Can potentially learn more complex design spaces

## Notes

- The model operates on normalized thickness values (0-1 range)
- Materials are represented as one-hot vectors during diffusion
- The denoising process uses standard DDPM reverse diffusion
- Sampling uses classifier-free guidance for better quality

## Testing

Run the test script to verify installation:
```bash
python test_pure_diffusion.py
```

This will create a dummy dataset and test model initialization, training step, and sampling.