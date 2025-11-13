#!/bin/bash
# Quick test script to verify installation and gradient flow

set -e  # Exit on error

echo "=========================================="
echo "Quick Test Script"
echo "=========================================="
echo ""

# Check if data file exists
DATA_FILE="optimized_dataset/optimized_multilayer_dataset.npz"
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Error: Dataset not found at $DATA_FILE"
    echo "Please ensure the dataset is available before running tests."
    exit 1
fi

echo "✓ Dataset found: $DATA_FILE"
echo ""

# Test 1: Gradient flow verification
echo "=========================================="
echo "Test 1: Gradient Flow Verification"
echo "=========================================="
python test_gradient_flow.py \
    --data "$DATA_FILE" \
    --batch_size 4 \
    --device cpu

if [ $? -eq 0 ]; then
    echo "✓ Gradient flow test passed!"
else
    echo "❌ Gradient flow test failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "All tests passed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Train model: python train_diffusion.py --data $DATA_FILE --epochs 100 --batch 128"
echo "  2. Sample: python sample_diffusion.py --checkpoint best_diffusion_model.pt --target_spectra <target>.npy"
echo ""
