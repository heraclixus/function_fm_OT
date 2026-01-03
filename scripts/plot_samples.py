#!/usr/bin/env python3
"""
Generate 3x2 sample visualization figure comparing different methods.

Layout:
    Ground Truth    |    Independent
    Euclidean       |    RBF
    Signature       |    DDPM

Usage:
    python plot_samples.py --dataset AEMET
    python plot_samples.py --dataset Heston --sample-idx 5
    python plot_samples.py --dataset navier_stokes --is-2d
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Tuple


# Dataset configurations
DATASETS = {
    'AEMET': {
        'dir': 'AEMET_ot_comprehensive',
        'title': 'AEMET Temperature',
        'is_2d': False,
    },
    'expr_genes': {
        'dir': 'expr_genes_ot_comprehensive',
        'title': 'Gene Expression',
        'is_2d': False,
    },
    'econ1': {
        'dir': 'econ_ot_comprehensive',
        'title': 'Economy',
        'is_2d': False,
        'subdir': 'econ1_population',
    },
    'Heston': {
        'dir': 'Heston_ot_kappa1.0',
        'title': 'Heston Model',
        'is_2d': False,
    },
    'rBergomi': {
        'dir': 'rBergomi_ot_H0p10',
        'title': 'Rough Bergomi',
        'is_2d': False,
    },
    'kdv': {
        'dir': 'kdv_ot',
        'title': 'KdV Equation',
        'is_2d': False,
    },
    'navier_stokes': {
        'dir': 'navier_stokes_ot',
        'title': 'Navier-Stokes',
        'is_2d': True,
    },
    'stochastic_kdv': {
        'dir': 'stochastic_kdv_ot',
        'title': 'Stochastic KdV',
        'is_2d': False,
    },
    'stochastic_ns': {
        'dir': 'stochastic_ns_ot',
        'title': 'Stochastic NS',
        'is_2d': True,
    },
}

# Method configurations - which config folder to use for each method
METHOD_CONFIGS = {
    'Ground Truth': None,  # Special case - load from ground_truth.pt
    'Independent': ['independent'],
    'Euclidean': ['euclidean_exact', 'euclidean_sinkhorn_reg0.1', 'euclidean_sinkhorn_reg0.5'],
    'RBF': ['rbf_exact', 'rbf_sinkhorn_reg0.1', 'rbf_sinkhorn_reg0.5'],
    'Signature': ['signature_sinkhorn_reg0.1', 'signature_sinkhorn_reg0.5', 'signature_sinkhorn_reg1.0'],
    'DDPM': ['DDPM'],
}

# Plot order for 3x2 grid
PLOT_ORDER = ['Ground Truth', 'Independent', 'Euclidean', 'RBF', 'Signature', 'DDPM']


def load_samples(
    dataset_dir: Path,
    method_name: str,
    config_options: List[str],
    seed: int = 1
) -> Optional[torch.Tensor]:
    """Load samples for a given method."""
    if config_options is None:
        # Ground truth
        gt_path = dataset_dir / 'ground_truth.pt'
        if gt_path.exists():
            return torch.load(gt_path, weights_only=True)
        # Try alternative locations
        for alt in ['ground_truth_rescaled.pt', 'ground_truth_original.pt']:
            alt_path = dataset_dir / alt
            if alt_path.exists():
                return torch.load(alt_path, weights_only=True)
        return None
    
    # Try each config option
    for config_name in config_options:
        samples_path = dataset_dir / config_name / f'seed_{seed}' / 'samples.pt'
        if samples_path.exists():
            return torch.load(samples_path, weights_only=True)
    
    return None


def plot_1d_sample(ax, sample: torch.Tensor, title: str, color: str = 'blue'):
    """Plot a 1D time series sample."""
    sample = sample.cpu().numpy()
    if sample.ndim > 1:
        sample = sample.squeeze()
    
    x = np.linspace(0, 1, len(sample))
    ax.plot(x, sample, color=color, linewidth=1.5, alpha=0.9)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('t', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_2d_sample(ax, sample: torch.Tensor, title: str, cmap: str = 'RdBu_r'):
    """Plot a 2D field sample."""
    sample = sample.cpu().numpy()
    if sample.ndim > 2:
        sample = sample.squeeze()
    if sample.ndim > 2:
        sample = sample[0]  # Take first channel if still 3D
    
    vmax = np.abs(sample).max()
    im = ax.imshow(sample, cmap=cmap, vmin=-vmax, vmax=vmax, origin='lower')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def create_sample_comparison_figure(
    dataset_key: str,
    outputs_dir: Path,
    sample_idx: int = 0,
    seed: int = 1,
    output_path: Optional[Path] = None,
    is_2d: Optional[bool] = None,
) -> plt.Figure:
    """
    Create 3x2 sample comparison figure.
    
    Parameters
    ----------
    dataset_key : str
        Key of the dataset in DATASETS
    outputs_dir : Path
        Path to outputs directory
    sample_idx : int
        Index of sample to visualize
    seed : int
        Random seed directory to use
    output_path : Path, optional
        Where to save the figure
    is_2d : bool, optional
        Whether data is 2D (overrides dataset config)
    """
    if dataset_key not in DATASETS:
        print(f"Unknown dataset: {dataset_key}")
        print(f"Available: {list(DATASETS.keys())}")
        return None
    
    dataset_info = DATASETS[dataset_key]
    dataset_dir = outputs_dir / dataset_info['dir']
    
    # Handle subdirectory (e.g., econ1_population)
    if 'subdir' in dataset_info:
        dataset_dir = dataset_dir / dataset_info['subdir']
    
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return None
    
    # Determine if 2D
    data_is_2d = is_2d if is_2d is not None else dataset_info.get('is_2d', False)
    
    # Load samples for each method
    samples = {}
    for method_name in PLOT_ORDER:
        config_options = METHOD_CONFIGS[method_name]
        sample_data = load_samples(dataset_dir, method_name, config_options, seed)
        if sample_data is not None:
            if sample_idx < len(sample_data):
                samples[method_name] = sample_data[sample_idx]
            else:
                print(f"Warning: sample_idx {sample_idx} out of range for {method_name}")
                samples[method_name] = sample_data[0]
        else:
            print(f"Warning: Could not load samples for {method_name}")
            samples[method_name] = None
    
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    
    # Colors for 1D plots
    colors = {
        'Ground Truth': '#2E7D32',  # Green
        'Independent': '#1565C0',   # Blue
        'Euclidean': '#E65100',     # Orange
        'RBF': '#7B1FA2',           # Purple
        'Signature': '#C62828',     # Red
        'DDPM': '#00838F',          # Teal
    }
    
    # Plot each method
    for idx, method_name in enumerate(PLOT_ORDER):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        sample = samples.get(method_name)
        if sample is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(method_name, fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        if data_is_2d:
            im = plot_2d_sample(ax, sample, method_name)
        else:
            plot_1d_sample(ax, sample, method_name, color=colors.get(method_name, 'blue'))
    
    # Add colorbar for 2D plots
    if data_is_2d:
        # Find a valid image for colorbar
        for idx, method_name in enumerate(PLOT_ORDER):
            if samples.get(method_name) is not None:
                row, col = idx // 2, idx % 2
                fig.colorbar(axes[row, col].images[0], ax=axes, shrink=0.6, aspect=30,
                           pad=0.02, label='Value')
                break
    
    plt.suptitle(f'{dataset_info["title"]} - Sample Comparison', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.95 if data_is_2d else 1, 0.96])
    
    # Save
    if output_path is None:
        output_path = outputs_dir / f'{dataset_key}_sample_comparison.png'
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Generate 3x2 sample comparison figure'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help=f'Dataset name. Available: {", ".join(DATASETS.keys())}'
    )
    parser.add_argument(
        '--outputs-dir', '-o',
        type=str,
        default=None,
        help='Path to outputs directory'
    )
    parser.add_argument(
        '--sample-idx', '-s',
        type=int,
        default=0,
        help='Index of sample to visualize (default: 0)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed directory to use (default: 1)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path'
    )
    parser.add_argument(
        '--is-2d',
        action='store_true',
        help='Force 2D plotting mode'
    )
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List available datasets and exit'
    )
    
    args = parser.parse_args()
    
    if args.list_datasets:
        print("Available datasets:")
        for key, info in DATASETS.items():
            print(f"  {key}: {info['title']} ({'2D' if info.get('is_2d') else '1D'})")
        return
    
    # Determine outputs directory
    if args.outputs_dir:
        outputs_dir = Path(args.outputs_dir)
    else:
        script_dir = Path(__file__).parent
        outputs_dir = script_dir.parent / 'outputs'
    
    output_path = Path(args.output) if args.output else None
    
    fig = create_sample_comparison_figure(
        dataset_key=args.dataset,
        outputs_dir=outputs_dir,
        sample_idx=args.sample_idx,
        seed=args.seed,
        output_path=output_path,
        is_2d=args.is_2d if args.is_2d else None,
    )
    
    if fig:
        plt.close(fig)
        print("Done!")


if __name__ == '__main__':
    main()

