"""Evaluate diffusion-based inverse design by comparing generated spectra to target spectra.

This script loads a trained diffusion model and evaluates it on a held-out test set of target spectra.
For each target, the script samples several candidate multilayer stacks, computes their spectra via the
exact TMM implementation, and records accuracy metrics such as r2, MSE, and MAE between the generated
spectrum and the target spectrum.
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from enhanced_diffusion_model_fixed import EnhancedDiffusionModel
from optimized_multilayer_generator import OptimizedMaterialDatabase, OptimizedTMMCalculator


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_layers(layers: List[Tuple[str, float]], min_thickness: float = 1.0) -> List[Tuple[str, float]]:
    cleaned = []
    for mat, thk in layers:
        mat = str(mat)
        try:
            thk_val = float(thk)
        except Exception:
            thk_val = 0.0
        if mat.upper() == 'VOID' or thk_val <= min_thickness:
            continue
        cleaned.append((mat, max(thk_val, min_thickness)))
    return cleaned


def compute_metrics_np(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2) + 1e-12)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return {'r2': r2, 'mse': mse, 'mae': mae}


@dataclass
class DatasetSplit:
    structures: List[List[Tuple[str, float]]]
    targets: np.ndarray  # shape (N, 2S)
    wavelengths: np.ndarray  # shape (S,)


def load_dataset_split(npz_path: str, train_ratio: float = 0.8) -> DatasetSplit:
    data = np.load(npz_path, allow_pickle=True)
    structures_raw = data['structures']
    transmission = data['transmission']
    reflection = data['reflection']
    wavelengths = data['wavelengths']

    num_samples = structures_raw.shape[0]
    train_size = int(num_samples * train_ratio)
    test_slice = slice(train_size, num_samples)

    structures = [list(struct) for struct in structures_raw[test_slice]]
    targets = np.concatenate([transmission[test_slice], reflection[test_slice]], axis=1)

    return DatasetSplit(structures=structures, targets=targets, wavelengths=wavelengths)


# -----------------------------------------------------------------------------
# Spectra predictors
# -----------------------------------------------------------------------------


class TMMPredictor:
    def __init__(self, wavelengths: np.ndarray):
        self.material_db = OptimizedMaterialDatabase()
        self.tmm = OptimizedTMMCalculator(self.material_db)
        self.wavelengths = wavelengths

    def predict(self, layers: List[Tuple[str, float]]) -> np.ndarray:
        cleaned = clean_layers(layers, min_thickness=self.tmm.min_thickness)
        if not cleaned:
            cleaned = [('SiO2', max(self.tmm.min_thickness, 50.0))]
        try:
            T, R, _ = self.tmm.calculate_spectrum_with_validation(cleaned, self.wavelengths)
        except Exception:
            T = np.full_like(self.wavelengths, 0.5, dtype=np.float64)
            R = np.full_like(self.wavelengths, 0.3, dtype=np.float64)
        return np.concatenate([np.asarray(T, dtype=np.float64), np.asarray(R, dtype=np.float64)])


# -----------------------------------------------------------------------------
# Evaluation workflow
# -----------------------------------------------------------------------------


def evaluate(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    diff_model = EnhancedDiffusionModel(device=args.device, data_path=args.data, pnn_path=None)
    diff_model.guidance_w = args.guidance
    if args.diffusion_ckpt and os.path.exists(args.diffusion_ckpt):
        ckpt = torch.load(args.diffusion_ckpt, map_location=device)
        state = ckpt.get('model_state', ckpt)
        diff_model.model.load_state_dict(state)
        print(f"Loaded diffusion weights from {args.diffusion_ckpt}")
    else:
        print("[warn] Diffusion checkpoint not found; using randomly initialized weights.")
    diff_model.model.eval()

    ds = diff_model.ds

    predictor = TMMPredictor(wavelengths=ds.wavelengths)

    split = load_dataset_split(args.data, train_ratio=args.train_ratio)
    num_targets = len(split.targets)
    print(f"Loaded test split with {num_targets} targets (train_ratio={args.train_ratio})")

    indices = list(range(num_targets))
    random.shuffle(indices)
    if args.num_targets > 0:
        indices = indices[:min(args.num_targets, num_targets)]

    metrics_list: List[Dict[str, float]] = []
    best_structures: List[Dict] = []

    for local_idx in tqdm(indices, desc='Evaluating targets'):
        target_spec = split.targets[local_idx]
        cond = torch.tensor(target_spec, dtype=torch.float32, device=device).unsqueeze(0)

        best_metric = {'r2': -math.inf, 'mse': math.inf, 'mae': math.inf}
        best_layers: Optional[List[Tuple[str, float]]] = None
        best_pred: Optional[np.ndarray] = None

        for _ in range(args.samples_per_target):
            sampled = diff_model.sample(cond, num_samples=1, use_ema=True)[0]
            predicted_spec = predictor.predict(sampled)
            metrics = compute_metrics_np(target_spec, predicted_spec)
            if metrics['r2'] > best_metric['r2']:
                best_metric = metrics
                best_layers = clean_layers(sampled)
                best_pred = predicted_spec

        metrics_list.append(best_metric)
        if args.topk_results > 0:
            best_structures.append({
                'index': int(local_idx),
                'r2': best_metric['r2'],
                'mse': best_metric['mse'],
                'mae': best_metric['mae'],
                'target_spectrum': target_spec.tolist(),
                'pred_spectrum': best_pred.tolist() if best_pred is not None else None,
                'layers': [(mat, float(thk)) for mat, thk in (best_layers or [])]
            })

    r2_vals = np.array([m['r2'] for m in metrics_list])
    mse_vals = np.array([m['mse'] for m in metrics_list])
    mae_vals = np.array([m['mae'] for m in metrics_list])

    print("\n=== Evaluation Summary ===")
    print(f"Samples evaluated: {len(metrics_list)}")
    print(f"r2 -> mean: {r2_vals.mean():.4f}, median: {np.median(r2_vals):.4f}, >=0.9: {(r2_vals >= 0.9).mean():.3f}")
    print(f"MSE -> mean: {mse_vals.mean():.6f}, median: {np.median(mse_vals):.6f}")
    print(f"MAE -> mean: {mae_vals.mean():.6f}, median: {np.median(mae_vals):.6f}")

    if args.save_hist:
        try:
            plt.figure(figsize=(6, 4))
            plt.hist(r2_vals, bins=40, alpha=0.8)
            plt.xlabel('r2')
            plt.ylabel('count')
            plt.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig(args.save_hist, dpi=200)
            print(f"Saved r2 histogram to {args.save_hist}")
        except Exception as e:
            print(f"Failed to save histogram: {e}")

    if args.save_results and args.topk_results > 0:
        best_structures = sorted(best_structures, key=lambda x: x['r2'], reverse=True)[:args.topk_results]
        with open(args.save_results, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'num_evaluated': len(metrics_list),
                    'r2_mean': r2_vals.mean(),
                    'r2_median': float(np.median(r2_vals)),
                    'mse_mean': mse_vals.mean(),
                    'mae_mean': mae_vals.mean()
                },
                'top_results': best_structures
            }, f, ensure_ascii=False, indent=2)
        print(f"Saved detailed results to {args.save_results}")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate inverse-designed multilayer structures.')
    parser.add_argument('--data', type=str, default='optimized_dataset/optimized_multilayer_dataset.npz', help='Dataset NPZ path')
    parser.add_argument('--diffusion_ckpt', type=str, default='enhanced_diffusion_best.pt', help='Trained diffusion checkpoint')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Portion of dataset considered training; remainder used for evaluation')
    parser.add_argument('--num_targets', type=int, default=200, help='Number of test targets to evaluate (0 => all)')
    parser.add_argument('--samples_per_target', type=int, default=4, help='Samples drawn per target; best r2 kept')
    parser.add_argument('--guidance', type=float, default=6.0, help='CFG strength during sampling')
    parser.add_argument('--predictor', type=str, default='tmm', choices=['tmm'], help='Forward model used for spectra prediction (TMM only)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_hist', type=str, default='r2_hist.png')
    parser.add_argument('--save_results', type=str, default='eval_results.json', help='Optional JSON to store top samples')
    parser.add_argument('--topk_results', type=int, default=10, help='Number of top samples to keep in JSON (0 disables)')
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()


