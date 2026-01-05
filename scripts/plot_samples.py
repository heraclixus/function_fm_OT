#!/usr/bin/env python3
"""
Generate 3x2 sample visualization figure comparing different methods.

Layout for 1D data:
    Ground Truth    |    Independent
    Euclidean       |    RBF
    Signature       |    DDPM

Layout for 2D data (no Signature kernel):
    Ground Truth    |    Independent
    Euclidean       |    RBF
    DDPM            |    NCSN

Usage:
    python plot_samples.py --dataset AEMET
    python plot_samples.py --dataset Heston --sample-idx 5
    python plot_samples.py --dataset navier_stokes
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List


# Dataset configurations
DATASETS = {
    'AEMET': {
        'dir': 'AEMET_ot_comprehensive',
        'title': 'AEMET Temperature',
        'is_2d': False,
        # Ground truth may be in different location
        'ground_truth_fallback': 'data/aemet.csv',  # Original CSV data
    },
    'expr_genes': {
        'dir': 'expr_genes_ot_comprehensive',
        'title': 'Gene Expression',
        'is_2d': False,
        'samples_file': 'samples_original.pt',  # Different filename
    },
    'econ1': {
        'dir': 'econ_ot_comprehensive',
        'title': 'Economy',
        'is_2d': False,
        'subdir': 'econ1_population',
        'samples_file': 'samples_original.pt',  # Different filename
    },
    'Heston': {
        'dir': 'Heston_ot_kappa1.0',
        'title': 'Heston Model',
        'is_2d': False,
        # Ground truth generated on-the-fly from this data file
        'data_file': 'data/Heston_kappa1.0_sigma0.3_n5000.pt',
    },
    'rBergomi': {
        'dir': 'rBergomi_ot_H0p10',
        'title': 'Rough Bergomi',
        'is_2d': False,
        # Ground truth generated on-the-fly from this data file
        'data_file': 'data/rBergomi_H0p10_n5000.pt',
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
    'Ground Truth': None,  # Special case - load from ground_truth.pt or data file
    'Independent': ['independent'],
    'Euclidean': ['euclidean_exact', 'euclidean_sinkhorn_reg0.1', 'euclidean_sinkhorn_reg0.5'],
    'RBF': ['rbf_exact', 'rbf_sinkhorn_reg0.1', 'rbf_sinkhorn_reg0.5'],
    'Signature': ['signature_sinkhorn_reg0.1', 'signature_sinkhorn_reg0.5', 'signature_sinkhorn_reg1.0'],
    'DDPM': ['DDPM'],
    'NCSN': ['NCSN'],
}

# Plot order for 3x2 grid - different for 1D and 2D
PLOT_ORDER_1D = ['Ground Truth', 'Independent', 'Euclidean', 'RBF', 'Signature', 'DDPM']
PLOT_ORDER_2D = ['Ground Truth', 'Independent', 'Euclidean', 'RBF', 'DDPM', 'NCSN']


def load_samples(
    dataset_dir: Path,
    method_name: str,
    config_options: List[str],
    seed: int = 1,
    data_file: Optional[str] = None,
    project_root: Optional[Path] = None,
    samples_file: str = 'samples.pt',
    ground_truth_fallback: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """Load samples for a given method.
    
    Parameters
    ----------
    dataset_dir : Path
        Directory containing experiment outputs
    method_name : str
        Name of the method
    config_options : List[str]
        List of config folder names to try
    seed : int
        Random seed directory
    data_file : str, optional
        Path to original data file (relative to project root) for on-the-fly datasets
    project_root : Path, optional
        Project root directory
    samples_file : str
        Name of the samples file (default: 'samples.pt', can be 'samples_original.pt')
    ground_truth_fallback : str, optional
        Fallback path for ground truth (can be CSV or PT file)
    """
    if config_options is None:
        # Ground truth - try multiple sources
        
        # 1. Try ground_truth.pt in dataset directory
        gt_path = dataset_dir / 'ground_truth.pt'
        if gt_path.exists():
            data = torch.load(gt_path, weights_only=False)
            if isinstance(data, dict):
                for key in ['log_V_normalized', 'log_S_normalized', 'data', 'samples', 'X', 'x']:
                    if key in data and isinstance(data[key], torch.Tensor):
                        return data[key]
                for v in data.values():
                    if isinstance(v, torch.Tensor) and v.ndim >= 2:
                        return v
            return data
        
        # 2. Try alternative locations in dataset directory
        for alt in ['ground_truth_rescaled.pt', 'ground_truth_original.pt']:
            alt_path = dataset_dir / alt
            if alt_path.exists():
                data = torch.load(alt_path, weights_only=False)
                if isinstance(data, dict):
                    for key in ['log_V_normalized', 'log_S_normalized', 'data', 'samples', 'X', 'x']:
                        if key in data and isinstance(data[key], torch.Tensor):
                            return data[key]
                    for v in data.values():
                        if isinstance(v, torch.Tensor) and v.ndim >= 2:
                            return v
                return data
        
        # 3. Try outputs-old directory as fallback
        if project_root:
            outputs_old_dir = project_root / 'outputs-old' / dataset_dir.name
            for alt in ['ground_truth_rescaled.pt', 'ground_truth_original.pt', 'ground_truth.pt']:
                alt_path = outputs_old_dir / alt
                if alt_path.exists():
                    print(f"  Loading ground truth from: {alt_path}")
                    data = torch.load(alt_path, weights_only=False)
                    if isinstance(data, dict):
                        for key in ['log_V_normalized', 'log_S_normalized', 'data', 'samples', 'X', 'x']:
                            if key in data and isinstance(data[key], torch.Tensor):
                                return data[key]
                        for v in data.values():
                            if isinstance(v, torch.Tensor) and v.ndim >= 2:
                                return v
                    return data
        
        # 4. Try loading from original data file (for Heston, rBergomi, etc.)
        if data_file and project_root:
            data_path = project_root / data_file
            if data_path.exists():
                data = torch.load(data_path, weights_only=False)
                # Handle different data formats
                if isinstance(data, dict):
                    # Heston format: has 'log_V_normalized', 'log_S_normalized', etc.
                    if 'log_V_normalized' in data:
                        return data['log_V_normalized']
                    # rBergomi format: has 'log_V_normalized' 
                    if 'log_S_normalized' in data:
                        return data['log_S_normalized']
                    # Generic: try common keys
                    for key in ['data', 'samples', 'X', 'x']:
                        if key in data:
                            return data[key]
                    # Return first tensor value found
                    for v in data.values():
                        if isinstance(v, torch.Tensor) and v.ndim >= 2:
                            return v
                return data
        
        # 5. Try ground_truth_fallback (can be CSV for AEMET)
        if ground_truth_fallback and project_root:
            fallback_path = project_root / ground_truth_fallback
            if fallback_path.exists():
                print(f"  Loading ground truth from fallback: {fallback_path}")
                if fallback_path.suffix == '.csv':
                    # Load CSV (e.g., AEMET data)
                    import pandas as pd
                    df = pd.read_csv(fallback_path, index_col=0)
                    data = torch.tensor(df.values, dtype=torch.float32)
                    print(f"  CSV data shape: {data.shape}")
                    return data
                else:
                    data = torch.load(fallback_path, weights_only=False)
                    if isinstance(data, dict):
                        for key in ['log_V_normalized', 'log_S_normalized', 'data', 'samples', 'X', 'x']:
                            if key in data and isinstance(data[key], torch.Tensor):
                                return data[key]
                        for v in data.values():
                            if isinstance(v, torch.Tensor) and v.ndim >= 2:
                                return v
                    return data
        
        return None
    
    # Try each config option
    for config_name in config_options:
        # Try with the specified samples file, then fallback to samples.pt
        for sf in [samples_file, 'samples.pt', 'samples_original.pt']:
            samples_path = dataset_dir / config_name / f'seed_{seed}' / sf
            if samples_path.exists():
                data = torch.load(samples_path, weights_only=False)
                # Handle dict format
                if isinstance(data, dict):
                    for key in ['samples', 'data', 'X', 'x']:
                        if key in data and isinstance(data[key], torch.Tensor):
                            return data[key]
                    # Return first tensor
                    for v in data.values():
                        if isinstance(v, torch.Tensor) and v.ndim >= 2:
                            return v
                return data
    
    return None


def plot_1d_sample(ax, sample: torch.Tensor, title: str, color: str = 'blue'):
    """Plot a 1D time series sample."""
    sample = sample.cpu().numpy()
    # Squeeze all extra dimensions: (1, T), (1, 1, T), etc. -> (T,)
    sample = sample.squeeze()
    
    x = np.linspace(0, 1, len(sample))
    ax.plot(x, sample, color=color, linewidth=1.5, alpha=0.9)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('t', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_1d_samples_multi(ax, samples: torch.Tensor, title: str, color: str = 'blue', 
                          n_samples: int = 500, alpha: float = 0.1):
    """Plot multiple 1D time series samples overlaid.
    
    Parameters
    ----------
    ax : matplotlib axis
    samples : torch.Tensor
        Tensor of shape (N, T) or (N, 1, T) where N is number of samples, T is time steps
    title : str
        Plot title
    color : str
        Line color
    n_samples : int
        Number of samples to plot (randomly selected if more available)
    alpha : float
        Transparency for individual lines
    """
    samples = samples.cpu().numpy()
    # Handle 3D tensors with channel dimension: (N, C, T) -> (N, T)
    if samples.ndim == 3:
        samples = samples.squeeze(1)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    
    n_available = samples.shape[0]
    n_plot = min(n_samples, n_available)
    
    # Randomly sample if we have more than needed
    if n_available > n_samples:
        indices = np.random.choice(n_available, n_samples, replace=False)
        samples = samples[indices]
    
    x = np.linspace(0, 1, samples.shape[1])
    
    # Plot all samples with transparency
    for i in range(n_plot):
        ax.plot(x, samples[i], color=color, linewidth=0.5, alpha=alpha)
    
    # Plot mean trajectory with thicker line
    mean_traj = np.mean(samples, axis=0)
    ax.plot(x, mean_traj, color=color, linewidth=2, alpha=0.9, label='Mean')
    
    ax.set_title(f'{title} (n={n_plot})', fontsize=12, fontweight='bold')
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
    
    # Get project root for loading data files
    project_root = outputs_dir.parent
    
    # Handle subdirectory (e.g., econ1_population)
    if 'subdir' in dataset_info:
        dataset_dir = dataset_dir / dataset_info['subdir']
    
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return None
    
    # Determine if 2D
    data_is_2d = is_2d if is_2d is not None else dataset_info.get('is_2d', False)
    
    # Select appropriate plot order based on data type
    plot_order = PLOT_ORDER_2D if data_is_2d else PLOT_ORDER_1D
    
    # Get data file path for on-the-fly datasets (Heston, rBergomi)
    data_file = dataset_info.get('data_file', None)
    # Get samples file name (default: samples.pt, can be samples_original.pt)
    samples_file = dataset_info.get('samples_file', 'samples.pt')
    # Get ground truth fallback path (for datasets like AEMET)
    ground_truth_fallback = dataset_info.get('ground_truth_fallback', None)
    
    # Load samples for each method
    samples = {}
    for method_name in plot_order:
        config_options = METHOD_CONFIGS[method_name]
        sample_data = load_samples(
            dataset_dir, method_name, config_options, seed,
            data_file=data_file, project_root=project_root,
            samples_file=samples_file,
            ground_truth_fallback=ground_truth_fallback
        )
        if sample_data is not None:
            # Handle case where sample_data might still be a dict
            if isinstance(sample_data, dict):
                # Try to extract the first tensor
                for key in ['log_V_normalized', 'log_S_normalized', 'data', 'samples', 'X', 'x']:
                    if key in sample_data and isinstance(sample_data[key], torch.Tensor):
                        sample_data = sample_data[key]
                        break
                else:
                    # Get first tensor in dict
                    for v in sample_data.values():
                        if isinstance(v, torch.Tensor) and v.ndim >= 2:
                            sample_data = v
                            break
            
            # Now index into the tensor
            if isinstance(sample_data, torch.Tensor):
                if sample_idx < len(sample_data):
                    samples[method_name] = sample_data[sample_idx]
                else:
                    print(f"Warning: sample_idx {sample_idx} out of range for {method_name}")
                    samples[method_name] = sample_data[0]
            else:
                print(f"Warning: Unexpected data type for {method_name}: {type(sample_data)}")
                samples[method_name] = None
        else:
            print(f"Warning: Could not load samples for {method_name}")
            samples[method_name] = None
    
    # Create figure - use 2x3 for 2D data, 3x2 for 1D data
    if data_is_2d:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        n_cols = 3
    else:
        fig, axes = plt.subplots(3, 2, figsize=(8, 10))
        n_cols = 2
    
    # Colors for plots
    colors = {
        'Ground Truth': '#2E7D32',  # Green
        'Independent': '#1565C0',   # Blue
        'Euclidean': '#E65100',     # Orange
        'RBF': '#7B1FA2',           # Purple
        'Signature': '#C62828',     # Red
        'DDPM': '#00838F',          # Teal
        'NCSN': '#6A1B9A',          # Deep Purple
    }
    
    # Plot each method
    for idx, method_name in enumerate(plot_order):
        row = idx // n_cols
        col = idx % n_cols
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
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    if output_path is None:
        output_path = outputs_dir / f'{dataset_key}_sample_comparison.png'
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    
    return fig


def create_multi_sample_comparison_figure(
    dataset_key: str,
    outputs_dir: Path,
    n_samples: int = 500,
    seed: int = 1,
    output_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """
    Create 3x2 multi-sample comparison figure showing 500 samples overlaid.
    Only works for 1D data.
    
    Parameters
    ----------
    dataset_key : str
        Key of the dataset in DATASETS
    outputs_dir : Path
        Path to outputs directory
    n_samples : int
        Number of samples to plot per method (default: 500)
    seed : int
        Random seed directory to use
    output_path : Path, optional
        Where to save the figure
    """
    if dataset_key not in DATASETS:
        print(f"Unknown dataset: {dataset_key}")
        print(f"Available: {list(DATASETS.keys())}")
        return None
    
    dataset_info = DATASETS[dataset_key]
    
    # Skip 2D datasets for multi-sample visualization
    if dataset_info.get('is_2d', False):
        print(f"Skipping multi-sample visualization for 2D dataset: {dataset_key}")
        return None
    
    dataset_dir = outputs_dir / dataset_info['dir']
    project_root = outputs_dir.parent
    
    # Handle subdirectory (e.g., econ1_population)
    if 'subdir' in dataset_info:
        dataset_dir = dataset_dir / dataset_info['subdir']
    
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return None
    
    plot_order = PLOT_ORDER_1D
    
    # Get data file and samples file
    data_file = dataset_info.get('data_file', None)
    samples_file = dataset_info.get('samples_file', 'samples.pt')
    ground_truth_fallback = dataset_info.get('ground_truth_fallback', None)
    
    # Load ALL samples for each method (not just one)
    all_samples = {}
    for method_name in plot_order:
        config_options = METHOD_CONFIGS[method_name]
        sample_data = load_samples(
            dataset_dir, method_name, config_options, seed,
            data_file=data_file, project_root=project_root,
            samples_file=samples_file,
            ground_truth_fallback=ground_truth_fallback
        )
        if sample_data is not None:
            # Handle case where sample_data might still be a dict
            if isinstance(sample_data, dict):
                for key in ['log_V_normalized', 'log_S_normalized', 'data', 'samples', 'X', 'x']:
                    if key in sample_data and isinstance(sample_data[key], torch.Tensor):
                        sample_data = sample_data[key]
                        break
                else:
                    for v in sample_data.values():
                        if isinstance(v, torch.Tensor) and v.ndim >= 2:
                            sample_data = v
                            break
            
            if isinstance(sample_data, torch.Tensor):
                all_samples[method_name] = sample_data
            else:
                print(f"Warning: Unexpected data type for {method_name}: {type(sample_data)}")
                all_samples[method_name] = None
        else:
            print(f"Warning: Could not load samples for {method_name}")
            all_samples[method_name] = None
    
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    
    # Colors for plots
    colors = {
        'Ground Truth': '#2E7D32',
        'Independent': '#1565C0',
        'Euclidean': '#E65100',
        'RBF': '#7B1FA2',
        'Signature': '#C62828',
        'DDPM': '#00838F',
        'NCSN': '#6A1B9A',
    }
    
    # Plot each method
    for idx, method_name in enumerate(plot_order):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        samples = all_samples.get(method_name)
        if samples is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(method_name, fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        plot_1d_samples_multi(ax, samples, method_name, 
                             color=colors.get(method_name, 'blue'),
                             n_samples=n_samples, alpha=0.05)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    if output_path is None:
        output_path = outputs_dir / f'{dataset_key}_multi_sample_comparison.png'
    
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
    parser.add_argument(
        '--multi-sample',
        action='store_true',
        help='Generate multi-sample comparison plot (500 samples overlaid)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=500,
        help='Number of samples for multi-sample plot (default: 500)'
    )
    parser.add_argument(
        '--both',
        action='store_true',
        help='Generate both single-sample and multi-sample plots'
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
    
    # Generate single-sample plot unless --multi-sample only
    if not args.multi_sample or args.both:
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
    
    # Generate multi-sample plot if requested
    if args.multi_sample or args.both:
        multi_output_path = None
        if output_path:
            # Modify output path for multi-sample version
            stem = output_path.stem
            multi_output_path = output_path.with_name(f'{stem}_multi{output_path.suffix}')
        
        fig = create_multi_sample_comparison_figure(
            dataset_key=args.dataset,
            outputs_dir=outputs_dir,
            n_samples=args.n_samples,
            seed=args.seed,
            output_path=multi_output_path,
        )
        if fig:
            plt.close(fig)
    
    print("Done!")


if __name__ == '__main__':
    main()

