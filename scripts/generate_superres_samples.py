#!/usr/bin/env python3
"""
Generate super-resolution samples from trained FFM models.

The Fourier Neural Operator (FNO) backbone allows generating samples at arbitrary
resolutions different from the training resolution. This script demonstrates this
capability by generating samples at 2x and 4x the original resolution.

For 1D sequences (time series):
  - Original: e.g., 64 or 365 time points
  - Super-res: e.g., 256, 512, or 1024 time points

For 2D grids (PDEs):
  - Original: e.g., 64x64 spatial grid
  - Super-res: e.g., 128x128, 256x256 spatial grid

NOTE: This script requires trained models (model.pt) to be saved. If models were
not saved during training, use --train-fresh to train a quick model on-the-fly.

Usage:
    # Generate for sequence dataset at 2x resolution
    python generate_superres_samples.py --dataset AEMET --scale 2
    
    # Generate for PDE dataset
    python generate_superres_samples.py --dataset navier_stokes --scale 4
    
    # Train fresh models and generate (if no saved models available)
    python generate_superres_samples.py --dataset AEMET --train-fresh --epochs 20
    
    # List available datasets
    python generate_superres_samples.py --list-datasets
"""

import sys
sys.path.append('../')

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json

from models.fno import FNO
from functional_fm_ot import FFMModelOT
from diffusion import DiffusionModel

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Dataset configurations
# =============================================================================

DATASETS = {
    # 1D Sequence datasets
    'AEMET': {
        'dir': 'AEMET_ot_comprehensive',
        'title': 'AEMET Temperature',
        'is_2d': False,
        'original_dims': (365,),  # Days in a year
        'n_channels': 1,
        # FNO model params
        'modes': 64,
        'width': 256,
        'mlp_width': 128,
        # GP params
        'kernel_length': 0.01,
        'kernel_variance': 0.1,
    },
    'expr_genes': {
        'dir': 'expr_genes_ot_comprehensive',
        'title': 'Gene Expression',
        'is_2d': False,
        'original_dims': (64,),
        'n_channels': 1,
        'modes': 32,
        'width': 256,
        'mlp_width': 128,
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
    },
    'econ1': {
        'dir': 'econ_ot_comprehensive',
        'title': 'Economy',
        'is_2d': False,
        'original_dims': (128,),
        'subdir': 'econ1_population',
        'n_channels': 1,
        'modes': 32,
        'width': 256,
        'mlp_width': 128,
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
    },
    'Heston': {
        'dir': 'Heston_ot_kappa1.0',
        'title': 'Heston Model',
        'is_2d': False,
        'original_dims': (64,),
        'n_channels': 1,
        'modes': 32,
        'width': 256,
        'mlp_width': 128,
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
        'data_file': 'data/Heston_kappa1.0_sigma0.3_n5000.pt',
    },
    'rBergomi': {
        'dir': 'rBergomi_ot_H0p10',
        'title': 'Rough Bergomi',
        'is_2d': False,
        'original_dims': (64,),
        'n_channels': 1,
        'modes': 32,
        'width': 256,
        'mlp_width': 128,
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
        'data_file': 'data/rBergomi_H0p10_n5000.pt',
    },
    'kdv': {
        'dir': 'kdv_ot',
        'title': 'KdV Equation',
        'is_2d': False,
        'original_dims': (64,),
        'n_channels': 1,
        'modes': 32,
        'width': 256,
        'mlp_width': 128,
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
    },
    'stochastic_kdv': {
        'dir': 'stochastic_kdv_ot',
        'title': 'Stochastic KdV',
        'is_2d': False,
        'original_dims': (256,),
        'n_channels': 1,
        'modes': 64,
        'width': 256,
        'mlp_width': 128,
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
    },
    'moGP': {
        'dir': 'moGP_ot',
        'title': 'Mixed Output GP',
        'is_2d': False,
        'original_dims': (64,),
        'n_channels': 1,
        'modes': 32,
        'width': 256,
        'mlp_width': 128,
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
    },
    
    # 2D PDE datasets
    'navier_stokes': {
        'dir': 'navier_stokes_ot',
        'title': 'Navier-Stokes',
        'is_2d': True,
        'original_dims': (64, 64),
        'n_channels': 1,
        # FNO model params for 2D
        'modes': 16,
        'hch': 32,
        'pch': 64,
        # GP params
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
    },
    'stochastic_ns': {
        'dir': 'stochastic_ns_ot',
        'title': 'Stochastic NS',
        'is_2d': True,
        'original_dims': (64, 64),
        'n_channels': 1,
        'modes': 16,
        'hch': 32,
        'pch': 64,
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
    },
}

# Kernel configurations to try loading (in order of preference)
METHOD_CONFIGS = {
    'Independent': ['independent'],
    'Euclidean': ['euclidean_exact', 'euclidean_sinkhorn_reg0.1', 'euclidean_sinkhorn_reg0.5'],
    'RBF': ['rbf_exact', 'rbf_sinkhorn_reg0.1', 'rbf_sinkhorn_reg0.5'],
    'Signature': ['signature_sinkhorn_reg0.1', 'signature_sinkhorn_reg0.5', 'signature_sinkhorn_reg1.0'],
    'DDPM': ['DDPM'],
    'NCSN': ['NCSN'],
}

# Baseline diffusion model configurations
BASELINE_CONFIGS = {
    "DDPM": {
        "method": "DDPM",
        "T": 1000,
        "beta_min": 1e-4,
        "beta_max": 0.02,
    },
    "NCSN": {
        "method": "NCSN",
        "T": 10,
        "sigma1": 1.0,
        "sigmaT": 0.01,
        "precondition": True,
    },
}

# Plot order - DDPM for 1D sequences, DDPM+NCSN for 2D PDEs
PLOT_ORDER_1D = ['Ground Truth', 'Independent', 'Euclidean', 'RBF', 'Signature', 'DDPM']
PLOT_ORDER_2D = ['Ground Truth', 'Independent', 'Euclidean', 'RBF', 'DDPM', 'NCSN']


# =============================================================================
# Model loading utilities
# =============================================================================

def create_model_1d(modes: int, width: int, mlp_width: int, device: str) -> FNO:
    """Create 1D FNO model."""
    return FNO(
        modes,
        vis_channels=1,
        hidden_channels=width,
        proj_channels=mlp_width,
        x_dim=1,
        t_scaling=1000.0,
    ).to(device)


def create_model_2d(modes: int, hch: int, pch: int, device: str) -> FNO:
    """Create 2D FNO model."""
    return FNO(
        modes,
        vis_channels=1,
        hidden_channels=hch,
        proj_channels=pch,
        x_dim=2,
        t_scaling=1000.0,
    ).to(device)


def load_model_and_ffm(
    dataset_key: str,
    method_name: str,
    dataset_dir: Path,
    config_options: List[str],
    device: str,
    seed: int = 1,
) -> Optional[Tuple[FNO, FFMModelOT]]:
    """Load a trained model and create FFM wrapper.
    
    Returns
    -------
    tuple : (model, ffm) or None if not found
    """
    ds_info = DATASETS[dataset_key]
    
    # Try each config option until we find a model.pt
    model_path = None
    for config_name in config_options:
        candidate = dataset_dir / config_name / f'seed_{seed}' / 'model.pt'
        if candidate.exists():
            model_path = candidate
            print(f"  Found model: {config_name}/seed_{seed}/model.pt")
            break
    
    if model_path is None:
        print(f"  No model found for {method_name}")
        return None
    
    # Create model architecture
    is_2d = ds_info['is_2d']
    if is_2d:
        model = create_model_2d(
            ds_info['modes'],
            ds_info['hch'],
            ds_info['pch'],
            device,
        )
    else:
        model = create_model_1d(
            ds_info['modes'],
            ds_info['width'],
            ds_info['mlp_width'],
            device,
        )
    
    # Load state dict (weights_only=False for compatibility with PyTorch 2.6+)
    loaded = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different save formats
    if isinstance(loaded, dict):
        # Check for wrapped formats first
        if 'model' in loaded and isinstance(loaded['model'], dict):
            state_dict = loaded['model']
        elif 'state_dict' in loaded and isinstance(loaded['state_dict'], dict):
            state_dict = loaded['state_dict']
        else:
            # Filter out internal PyTorch metadata keys (like '_metadata')
            # State dict keys are like 'model.layer.weight', so filter _metadata specifically
            state_dict = {k: v for k, v in loaded.items() if k != '_metadata'}
        
        model.load_state_dict(state_dict, strict=False)
    elif hasattr(loaded, 'state_dict'):
        # Full model object was saved
        model.load_state_dict(loaded.state_dict(), strict=False)
    else:
        # Try loading directly
        model.load_state_dict(loaded, strict=False)
    
    model.eval()
    
    # Create FFM wrapper (we don't need OT for sampling, just GP prior)
    ffm = FFMModelOT(
        model=model,
        kernel_length=ds_info['kernel_length'],
        kernel_variance=ds_info['kernel_variance'],
        sigma_min=1e-4,
        use_ot=False,  # Not needed for sampling
        device=device,
        dtype=torch.float32,
    )
    
    return model, ffm


def load_diffusion_model(
    dataset_key: str,
    method_name: str,
    dataset_dir: Path,
    config_options: List[str],
    device: str,
    seed: int = 1,
) -> Optional[Tuple[FNO, DiffusionModel]]:
    """Load a trained DDPM/NCSN model and create DiffusionModel wrapper.
    
    Returns
    -------
    tuple : (model, diffusion) or None if not found
    """
    ds_info = DATASETS[dataset_key]
    
    # Try each config option until we find a model.pt
    model_path = None
    for config_name in config_options:
        candidate = dataset_dir / config_name / f'seed_{seed}' / 'model.pt'
        if candidate.exists():
            model_path = candidate
            print(f"  Found model: {config_name}/seed_{seed}/model.pt")
            break
    
    if model_path is None:
        print(f"  No model found for {method_name}")
        return None
    
    # Create model architecture
    is_2d = ds_info['is_2d']
    if is_2d:
        model = create_model_2d(
            ds_info['modes'],
            ds_info['hch'],
            ds_info['pch'],
            device,
        )
    else:
        model = create_model_1d(
            ds_info['modes'],
            ds_info['width'],
            ds_info['mlp_width'],
            device,
        )
    
    # Load state dict
    loaded = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different save formats
    if isinstance(loaded, dict):
        if 'model' in loaded and isinstance(loaded['model'], dict):
            state_dict = loaded['model']
        elif 'state_dict' in loaded and isinstance(loaded['state_dict'], dict):
            state_dict = loaded['state_dict']
        else:
            state_dict = {k: v for k, v in loaded.items() if k != '_metadata'}
        
        model.load_state_dict(state_dict, strict=False)
    elif hasattr(loaded, 'state_dict'):
        model.load_state_dict(loaded.state_dict(), strict=False)
    else:
        model.load_state_dict(loaded, strict=False)
    
    model.eval()
    
    # Create DiffusionModel wrapper
    baseline_config = BASELINE_CONFIGS[method_name]
    method = baseline_config["method"]
    
    if method == "DDPM":
        diffusion = DiffusionModel(
            model, 
            method=method, 
            T=baseline_config["T"],
            device=device,
            kernel_length=ds_info['kernel_length'], 
            kernel_variance=ds_info['kernel_variance'],
            beta_min=baseline_config["beta_min"], 
            beta_max=baseline_config["beta_max"],
            dtype=torch.float32,
        )
    else:  # NCSN
        diffusion = DiffusionModel(
            model, 
            method=method, 
            T=baseline_config["T"],
            device=device,
            kernel_length=ds_info['kernel_length'], 
            kernel_variance=ds_info['kernel_variance'],
            sigma1=baseline_config["sigma1"], 
            sigmaT=baseline_config["sigmaT"], 
            precondition=baseline_config.get("precondition", True),
            dtype=torch.float32,
        )
    
    return model, diffusion


def is_diffusion_method(method_name: str) -> bool:
    """Check if a method is a diffusion baseline (DDPM/NCSN)."""
    return method_name in ['DDPM', 'NCSN']


def train_fresh_model(
    dataset_key: str,
    method_name: str,
    train_data: torch.Tensor,
    device: str,
    epochs: int = 30,
    batch_size: int = 64,
    use_ot: bool = False,
) -> Tuple[FNO, FFMModelOT]:
    """Train a fresh model for super-resolution generation.
    
    Parameters
    ----------
    dataset_key : str
        Dataset name
    method_name : str
        Method name (Independent, Euclidean, RBF, Signature)
    train_data : Tensor
        Training data
    device : str
        Device to use
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    use_ot : bool
        Whether to use OT pairing
        
    Returns
    -------
    tuple : (model, ffm)
    """
    ds_info = DATASETS[dataset_key]
    is_2d = ds_info['is_2d']
    
    # Create model
    if is_2d:
        model = create_model_2d(
            ds_info['modes'],
            ds_info['hch'],
            ds_info['pch'],
            device,
        )
    else:
        model = create_model_1d(
            ds_info['modes'],
            ds_info['width'],
            ds_info['mlp_width'],
            device,
        )
    
    # Determine OT config based on method
    ot_config = {}
    if method_name == 'Independent':
        ot_config = {'use_ot': False}
    elif method_name == 'Euclidean':
        ot_config = {
            'use_ot': True,
            'ot_method': 'sinkhorn',
            'ot_reg': 0.1,
            'ot_kernel': 'euclidean',
            'ot_coupling': 'sample',
        }
    elif method_name == 'RBF':
        ot_config = {
            'use_ot': True,
            'ot_method': 'sinkhorn',
            'ot_reg': 0.1,
            'ot_kernel': 'rbf',
            'ot_coupling': 'sample',
        }
    elif method_name == 'Signature':
        ot_config = {
            'use_ot': True,
            'ot_method': 'sinkhorn',
            'ot_reg': 0.5,
            'ot_kernel': 'signature',
            'ot_coupling': 'sample',
            'ot_kernel_params': {'depth': 3, 'normalize': True},
        }
    else:
        ot_config = {'use_ot': False}
    
    # Create FFM
    ffm = FFMModelOT(
        model=model,
        kernel_length=ds_info['kernel_length'],
        kernel_variance=ds_info['kernel_variance'],
        sigma_min=1e-4,
        device=device,
        dtype=torch.float32,
        **ot_config,
    )
    
    # Prepare data
    if train_data.ndim == 2:
        train_data = train_data.unsqueeze(1)  # Add channel dim
    
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    
    # Train
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"    Training {method_name} for {epochs} epochs...")
    ffm.train(
        train_loader,
        optimizer,
        epochs=epochs,
        device=device,
        save_path=None,
        evaluate=False,
        generate=False,
    )
    
    model.eval()
    return model, ffm


def load_ground_truth(
    dataset_key: str,
    dataset_dir: Path,
    project_root: Path,
    n_samples: int = 100,
) -> Optional[torch.Tensor]:
    """Load ground truth samples for comparison."""
    import pandas as pd
    
    ds_info = DATASETS[dataset_key]
    
    # Try ground_truth.pt in dataset directory
    gt_path = dataset_dir / 'ground_truth.pt'
    if gt_path.exists():
        data = torch.load(gt_path, weights_only=False)
        if isinstance(data, torch.Tensor):
            return data[:n_samples]
        if isinstance(data, dict):
            for key in ['log_V_normalized', 'log_S_normalized', 'data', 'samples', 'X', 'x']:
                if key in data and isinstance(data[key], torch.Tensor):
                    return data[key][:n_samples]
    
    # Try loading from data file
    data_file = ds_info.get('data_file')
    if data_file:
        data_path = project_root / data_file
        if data_path.exists():
            data = torch.load(data_path, weights_only=False)
            if isinstance(data, dict):
                if 'log_V_normalized' in data:
                    return data['log_V_normalized'][:n_samples]
                if 'log_S_normalized' in data:
                    return data['log_S_normalized'][:n_samples]
    
    # Special handling for AEMET (CSV file)
    if dataset_key == 'AEMET':
        aemet_path = project_root / 'data' / 'aemet.csv'
        if aemet_path.exists():
            data_raw = pd.read_csv(aemet_path, index_col=0)
            data = torch.tensor(data_raw.values, dtype=torch.float32)
            # Normalize like in AEMET_ot.py
            min_data = torch.min(data)
            max_data = torch.max(data)
            data = 6 * (data - min_data) / (max_data - min_data) - 3
            return data[:n_samples]
    
    # Try common data paths
    common_paths = [
        project_root / 'data' / f'{dataset_key}.pt',
        project_root / 'data' / f'{dataset_key.lower()}.pt',
        project_root / 'data' / 'economy' / 'econ1.pt',  # For economy dataset
    ]
    
    for path in common_paths:
        if path.exists():
            try:
                data = torch.load(path, weights_only=False)
                if isinstance(data, torch.Tensor):
                    if data.ndim == 2:
                        return data[:n_samples]
                    elif data.ndim == 3:
                        return data[:n_samples].squeeze(1)
                if isinstance(data, dict):
                    for key in ['data', 'samples', 'X', 'x', 'log_V_normalized']:
                        if key in data:
                            d = data[key]
                            if d.ndim == 2:
                                return d[:n_samples]
                            elif d.ndim == 3:
                                return d[:n_samples].squeeze(1)
            except Exception:
                continue
    
    return None


# =============================================================================
# Sample generation
# =============================================================================

def generate_superres_samples(
    model_wrapper,  # Can be FFMModelOT or DiffusionModel
    target_dims: tuple,
    n_samples: int = 100,
    n_channels: int = 1,
) -> torch.Tensor:
    """Generate samples at specified resolution.
    
    Parameters
    ----------
    model_wrapper : FFMModelOT or DiffusionModel
        The model wrapper (supports both FFM and diffusion models)
    target_dims : tuple
        Target spatial dimensions, e.g., (256,) for 1D or (128, 128) for 2D
    n_samples : int
        Number of samples to generate
    n_channels : int
        Number of channels
        
    Returns
    -------
    samples : Tensor, shape (n_samples, *target_dims) or (n_samples, n_channels, *target_dims)
    """
    with torch.no_grad():
        # Both FFMModelOT and DiffusionModel have compatible sample() methods
        if isinstance(model_wrapper, DiffusionModel):
            # DiffusionModel.sample() signature
            samples = model_wrapper.sample(
                dims=target_dims,
                n_channels=n_channels,
                n_samples=n_samples,
            )
        else:
            # FFMModelOT.sample() signature
            samples = model_wrapper.sample(
                dims=target_dims,
                n_channels=n_channels,
                n_samples=n_samples,
                n_eval=2,  # Just need start and end
            )
    
    # Squeeze channel dim if single channel
    if n_channels == 1:
        samples = samples.squeeze(1)
    
    return samples


# =============================================================================
# Plotting utilities
# =============================================================================

def plot_1d_samples_multi_resolution(
    ax,
    samples: torch.Tensor,
    title: str,
    n_plot: int = 50,
    alpha: float = 0.3,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
):
    """Plot multiple 1D samples with varying line colors (no single color per model).
    
    This matches the visualization style of the FFM paper where each trajectory
    is drawn in a different color.
    """
    samples = samples.cpu().numpy()
    if samples.ndim == 3:
        samples = samples.squeeze(1)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    
    n_available = samples.shape[0]
    n_to_plot = min(n_plot, n_available)
    
    # Randomly select samples if needed
    if n_available > n_plot:
        indices = np.random.choice(n_available, n_plot, replace=False)
        samples = samples[indices]
    
    # Time axis
    n_points = samples.shape[1]
    t = np.linspace(0, 1, n_points)
    
    # Use colormap for varied colors (similar to FFM paper style)
    # Use a diverse colormap for better visual distinction
    cmap = plt.cm.get_cmap('tab20')
    colors = [cmap(i / min(20, n_to_plot)) for i in range(min(20, n_to_plot))]
    
    # Plot each sample with cycling colors
    for i in range(n_to_plot):
        color = colors[i % len(colors)]
        ax.plot(t, samples[i], color=color, linewidth=0.5, alpha=alpha)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    if show_xlabel:
        ax.set_xlabel('t', fontsize=12)
    if show_ylabel:
        ax.set_ylabel('Value', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    # Remove grid for cleaner look (matching FFM paper style)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_2d_samples_grid(
    ax,
    samples: torch.Tensor,
    title: str,
    sample_idx: int = 0,
    cmap: str = 'RdBu_r',
):
    """Plot a single 2D sample."""
    sample = samples[sample_idx].cpu().numpy()
    if sample.ndim > 2:
        sample = sample.squeeze()
    if sample.ndim > 2:
        sample = sample[0]
    
    vmax = np.abs(sample).max()
    ax.imshow(sample, cmap=cmap, vmin=-vmax, vmax=vmax, origin='lower')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])


def load_training_data(
    dataset_key: str,
    project_root: Path,
) -> Optional[torch.Tensor]:
    """Load training data for a dataset."""
    ds_info = DATASETS[dataset_key]
    
    # Try common data file locations
    data_file = ds_info.get('data_file')
    if data_file:
        data_path = project_root / data_file
        if data_path.exists():
            data = torch.load(data_path, weights_only=False)
            if isinstance(data, dict):
                if 'log_V_normalized' in data:
                    return data['log_V_normalized']
                if 'log_S_normalized' in data:
                    return data['log_S_normalized']
            return data
    
    # Try standard data paths
    data_paths = [
        project_root / 'data' / f'{dataset_key}.pt',
        project_root / 'data' / f'{dataset_key.lower()}.pt',
    ]
    
    for path in data_paths:
        if path.exists():
            data = torch.load(path, weights_only=False)
            if isinstance(data, dict):
                for key in ['data', 'samples', 'X', 'x']:
                    if key in data:
                        return data[key]
            return data
    
    return None


def create_superres_comparison_1d(
    dataset_key: str,
    outputs_dir: Path,
    scale_factors: List[int] = [4],
    n_samples: int = 50,
    seed: int = 1,
    output_path: Optional[Path] = None,
    device: str = 'cuda',
    train_fresh: bool = False,
    epochs: int = 30,
) -> Optional[plt.Figure]:
    """
    Create super-resolution comparison figure for 1D data.
    
    Layout: 3x2 grid
        - Ground Truth: original resolution (1x)
        - All other methods: 4x super-resolution
    """
    if dataset_key not in DATASETS:
        print(f"Unknown dataset: {dataset_key}")
        return None
    
    ds_info = DATASETS[dataset_key]
    if ds_info['is_2d']:
        print(f"Use create_superres_comparison_2d for 2D datasets")
        return None
    
    dataset_dir = outputs_dir / ds_info['dir']
    if 'subdir' in ds_info:
        dataset_dir = dataset_dir / ds_info['subdir']
    
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return None
    
    project_root = outputs_dir.parent
    original_dims = ds_info['original_dims']
    
    # Use 4x super-resolution for models (or highest scale factor provided)
    superres_scale = max(scale_factors) if scale_factors else 4
    superres_dims = tuple(d * superres_scale for d in original_dims)
    
    # Methods to compare (6 methods for 3x2 grid)
    methods = PLOT_ORDER_1D  # ['Ground Truth', 'Independent', 'Euclidean', 'RBF', 'Signature', 'DDPM']
    
    # Load ground truth at original resolution
    gt_samples = load_ground_truth(dataset_key, dataset_dir, project_root, n_samples)
    
    # Create 3x2 figure
    n_rows = 3
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 9))
    axes = axes.flatten()
    
    print(f"\nGenerating super-resolution samples for {ds_info['title']}...")
    print(f"Ground Truth: {original_dims[0]} pts")
    print(f"Models: {superres_dims[0]} pts ({superres_scale}x)")
    
    for idx, method_name in enumerate(methods):
        ax = axes[idx]
        print(f"\nProcessing {method_name}...")
        
        if method_name == 'Ground Truth':
            # Plot ground truth at ORIGINAL resolution
            if gt_samples is not None:
                plot_1d_samples_multi_resolution(
                    ax, gt_samples, f'Ground Truth\n(Original: {original_dims[0]} pts)',
                    n_plot=n_samples, alpha=0.3,
                )
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=16)
                ax.set_title(f'Ground Truth\n(Original)', fontsize=14, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            # Load model and generate at super-resolution
            config_options = METHOD_CONFIGS.get(method_name, [])
            
            # Use different loader for diffusion models vs FFM
            if is_diffusion_method(method_name):
                result = load_diffusion_model(
                    dataset_key, method_name, dataset_dir, config_options, device, seed
                )
            else:
                result = load_model_and_ffm(
                    dataset_key, method_name, dataset_dir, config_options, device, seed
                )
            
            if result is None:
                if train_fresh and not is_diffusion_method(method_name):
                    # Train a fresh model (only for FFM methods, not DDPM/NCSN)
                    print(f"  Training fresh model for {method_name}...")
                    train_data = load_training_data(dataset_key, project_root)
                    if train_data is None:
                        print(f"  Could not load training data for {dataset_key}")
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                               transform=ax.transAxes, fontsize=16)
                        ax.set_title(f'{method_name}', fontsize=14, fontweight='bold')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue
                    
                    model, model_wrapper = train_fresh_model(
                        dataset_key, method_name, train_data, device,
                        epochs=epochs, batch_size=64,
                    )
                else:
                    ax.text(0.5, 0.5, 'No model', ha='center', va='center',
                           transform=ax.transAxes, fontsize=16)
                    ax.set_title(f'{method_name}', fontsize=14, fontweight='bold')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
            else:
                model, model_wrapper = result
            
            # Generate at super-resolution (4x)
            print(f"  Generating at {superres_dims[0]} points ({superres_scale}x)...")
            
            try:
                samples = generate_superres_samples(
                    model_wrapper, superres_dims, n_samples=n_samples, n_channels=1
                )
                plot_1d_samples_multi_resolution(
                    ax, samples, f'{method_name}\n({superres_scale}x: {superres_dims[0]} pts)',
                    n_plot=n_samples, alpha=0.3,
                )
            except Exception as e:
                print(f"    Error: {e}")
                ax.text(0.5, 0.5, 'Error', ha='center', va='center',
                       transform=ax.transAxes, fontsize=16)
                ax.set_title(f'{method_name}\n({superres_scale}x)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    if output_path is None:
        output_path = outputs_dir / f'{dataset_key}_superres_comparison.png'
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    
    return fig


def create_superres_comparison_2d(
    dataset_key: str,
    outputs_dir: Path,
    scale_factors: List[int] = [2],
    n_samples: int = 4,
    seed: int = 1,
    output_path: Optional[Path] = None,
    device: str = 'cuda',
) -> Optional[plt.Figure]:
    """
    Create super-resolution comparison figure for 2D data.
    
    Layout: 2x3 grid (6 methods)
        - Ground Truth: original resolution (64x64)
        - All other methods: 2x super-resolution (128x128)
    """
    if dataset_key not in DATASETS:
        print(f"Unknown dataset: {dataset_key}")
        return None
    
    ds_info = DATASETS[dataset_key]
    if not ds_info['is_2d']:
        print(f"Use create_superres_comparison_1d for 1D datasets")
        return None
    
    dataset_dir = outputs_dir / ds_info['dir']
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return None
    
    project_root = outputs_dir.parent
    original_dims = ds_info['original_dims']
    
    # Use 2x super-resolution for models (or highest scale factor provided)
    superres_scale = max(scale_factors) if scale_factors else 2
    superres_dims = tuple(d * superres_scale for d in original_dims)
    
    # Methods to compare (6 methods for 2x3 grid)
    methods = PLOT_ORDER_2D  # ['Ground Truth', 'Independent', 'Euclidean', 'RBF', 'DDPM', 'NCSN']
    
    # Load ground truth
    gt_samples = load_ground_truth(dataset_key, dataset_dir, project_root, n_samples)
    
    # Create 2x3 figure
    n_rows = 2
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
    axes = axes.flatten()
    
    print(f"\nGenerating super-resolution samples for {ds_info['title']}...")
    print(f"Ground Truth: {original_dims[0]}x{original_dims[1]}")
    print(f"Models: {superres_dims[0]}x{superres_dims[1]} ({superres_scale}x)")
    
    for idx, method_name in enumerate(methods):
        ax = axes[idx]
        print(f"\nProcessing {method_name}...")
        
        if method_name == 'Ground Truth':
            # Plot ground truth at ORIGINAL resolution
            if gt_samples is not None:
                plot_2d_samples_grid(
                    ax, gt_samples, f'Ground Truth\n(Original: {original_dims[0]}x{original_dims[1]})',
                    sample_idx=0,
                )
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                       transform=ax.transAxes, fontsize=16)
                ax.set_title(f'Ground Truth\n(Original)', fontsize=14, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            config_options = METHOD_CONFIGS.get(method_name, [])
            
            # Use different loader for diffusion models vs FFM
            if is_diffusion_method(method_name):
                result = load_diffusion_model(
                    dataset_key, method_name, dataset_dir, config_options, device, seed
                )
            else:
                result = load_model_and_ffm(
                    dataset_key, method_name, dataset_dir, config_options, device, seed
                )
            
            if result is None:
                ax.text(0.5, 0.5, 'No model', ha='center', va='center',
                       transform=ax.transAxes, fontsize=16)
                ax.set_title(f'{method_name}', fontsize=14, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            model, model_wrapper = result
            
            # Generate at super-resolution (2x)
            print(f"  Generating at {superres_dims[0]}x{superres_dims[1]} ({superres_scale}x)...")
            
            try:
                samples = generate_superres_samples(
                    model_wrapper, superres_dims, n_samples=n_samples, n_channels=1
                )
                plot_2d_samples_grid(
                    ax, samples, f'{method_name}\n({superres_scale}x: {superres_dims[0]}x{superres_dims[1]})',
                    sample_idx=0,
                )
            except Exception as e:
                print(f"    Error: {e}")
                ax.text(0.5, 0.5, 'Error', ha='center', va='center',
                       transform=ax.transAxes, fontsize=16)
                ax.set_title(f'{method_name}\n({superres_scale}x)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = outputs_dir / f'{dataset_key}_superres_comparison.png'
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    
    return fig


def create_ffm_style_superres_plot(
    dataset_key: str,
    outputs_dir: Path,
    resolutions: List[int],
    n_samples: int = 50,
    seed: int = 1,
    output_path: Optional[Path] = None,
    device: str = 'cuda',
    train_fresh: bool = False,
    epochs: int = 30,
) -> Optional[plt.Figure]:
    """
    Create FFM paper-style 3x2 grid showing Ground Truth and models at different resolutions.
    
    Layout (similar to FFM paper Figure):
        Ground truth    |    FFM - OT (ours)
        FFM - VP (ours) |    DDPM
        DDO             |    GANO
    
    But adapted for our OT kernel comparison:
        Ground truth (orig) | Ground truth (2x)
        Independent (orig)  | Independent (2x)
        Signature (orig)    | Signature (2x)
    
    Each panel shows multiple trajectories with different colors.
    """
    if dataset_key not in DATASETS:
        print(f"Unknown dataset: {dataset_key}")
        return None
    
    ds_info = DATASETS[dataset_key]
    if ds_info['is_2d']:
        print(f"Use 2D function for 2D datasets")
        return None
    
    dataset_dir = outputs_dir / ds_info['dir']
    if 'subdir' in ds_info:
        dataset_dir = dataset_dir / ds_info['subdir']
    
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return None
    
    project_root = outputs_dir.parent
    
    # Methods to show (include DDPM for 1D)
    methods = ['Ground Truth', 'Independent', 'Euclidean', 'RBF', 'Signature', 'DDPM']
    
    # Load ground truth
    gt_samples = load_ground_truth(dataset_key, dataset_dir, project_root, n_samples)
    
    # Create figure - rows = methods, cols = resolutions
    n_rows = len(methods)
    n_cols = len(resolutions)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows))
    
    print(f"\nCreating FFM-style super-resolution plot for {ds_info['title']}...")
    print(f"Resolutions: {resolutions}")
    
    # Load training data if needed
    train_data = None
    if train_fresh:
        train_data = load_training_data(dataset_key, project_root)
    
    for row, method_name in enumerate(methods):
        print(f"\nProcessing {method_name}...")
        
        if method_name == 'Ground Truth':
            for col, res in enumerate(resolutions):
                ax = axes[row, col]
                if gt_samples is not None and col == 0:
                    plot_1d_samples_multi_resolution(
                        ax, gt_samples, f'Ground truth (n={res})',
                        n_plot=n_samples, alpha=0.4,
                        show_xlabel=(row == n_rows - 1),
                        show_ylabel=(col == 0),
                    )
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                           transform=ax.transAxes, fontsize=14)
                    ax.set_title(f'Ground truth (n={res})', fontsize=12, fontweight='bold')
                    ax.set_xticks([])
                    ax.set_yticks([])
        else:
            # Load or train model
            config_options = METHOD_CONFIGS.get(method_name, [])
            
            # Use different loader for diffusion models vs FFM
            if is_diffusion_method(method_name):
                result = load_diffusion_model(
                    dataset_key, method_name, dataset_dir, config_options, device, seed
                )
            else:
                result = load_model_and_ffm(
                    dataset_key, method_name, dataset_dir, config_options, device, seed
                )
            
            if result is None and train_fresh and train_data is not None and not is_diffusion_method(method_name):
                print(f"  Training fresh model for {method_name}...")
                model, model_wrapper = train_fresh_model(
                    dataset_key, method_name, train_data, device,
                    epochs=epochs, batch_size=64,
                )
            elif result is not None:
                model, model_wrapper = result
            else:
                for col in range(n_cols):
                    ax = axes[row, col]
                    ax.text(0.5, 0.5, 'No model', ha='center', va='center',
                           transform=ax.transAxes, fontsize=14)
                    ax.set_title(f'{method_name}', fontsize=12, fontweight='bold')
                    ax.set_xticks([])
                    ax.set_yticks([])
                continue
            
            for col, res in enumerate(resolutions):
                ax = axes[row, col]
                dims = (res,)
                print(f"  Generating at {res} points...")
                
                try:
                    samples = generate_superres_samples(
                        model_wrapper, dims, n_samples=n_samples, n_channels=1
                    )
                    plot_1d_samples_multi_resolution(
                        ax, samples, f'{method_name} (n={res})',
                        n_plot=n_samples, alpha=0.4,
                        show_xlabel=(row == n_rows - 1),
                        show_ylabel=(col == 0),
                    )
                except Exception as e:
                    print(f"    Error: {e}")
                    ax.text(0.5, 0.5, 'Error', ha='center', va='center',
                           transform=ax.transAxes, fontsize=14)
                    ax.set_title(f'{method_name} (n={res})', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = outputs_dir / f'{dataset_key}_superres_ffm_style.png'
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    
    return fig


def create_single_method_multiresolution_1d(
    dataset_key: str,
    method_name: str,
    outputs_dir: Path,
    resolutions: List[int],
    n_samples: int = 50,
    seed: int = 1,
    output_path: Optional[Path] = None,
    device: str = 'cuda',
) -> Optional[plt.Figure]:
    """
    Create a 2x3 or similar grid showing samples from a single method at multiple resolutions.
    
    This creates a plot similar to the provided screenshot where each panel shows
    samples at different temporal resolutions (e.g., 32, 64, 128, 256 points).
    """
    if dataset_key not in DATASETS:
        print(f"Unknown dataset: {dataset_key}")
        return None
    
    ds_info = DATASETS[dataset_key]
    if ds_info['is_2d']:
        print(f"Use separate 2D function for 2D datasets")
        return None
    
    dataset_dir = outputs_dir / ds_info['dir']
    if 'subdir' in ds_info:
        dataset_dir = dataset_dir / ds_info['subdir']
    
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return None
    
    project_root = outputs_dir.parent
    
    # Load model (use different loader for diffusion models)
    config_options = METHOD_CONFIGS.get(method_name, [])
    
    if is_diffusion_method(method_name):
        result = load_diffusion_model(
            dataset_key, method_name, dataset_dir, config_options, device, seed
        )
    else:
        result = load_model_and_ffm(
            dataset_key, method_name, dataset_dir, config_options, device, seed
        )
    
    if result is None:
        print(f"Could not load model for {method_name}")
        return None
    
    model, model_wrapper = result
    
    # Determine grid layout
    n_plots = len(resolutions)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    print(f"\nGenerating samples for {method_name} at resolutions: {resolutions}")
    
    for idx, res in enumerate(resolutions):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        dims = (res,)
        print(f"  Generating at {res} points...")
        
        try:
            samples = generate_superres_samples(
                model_wrapper, dims, n_samples=n_samples, n_channels=1
            )
            plot_1d_samples_multi_resolution(
                ax, samples, f'{method_name} (n={res})',
                n_plot=n_samples, alpha=0.3,
            )
        except Exception as e:
            print(f"    Error: {e}")
            ax.text(0.5, 0.5, 'Error', ha='center', va='center',
                   transform=ax.transAxes, fontsize=16)
            ax.set_title(f'{method_name} (n={res})', fontsize=14, fontweight='bold')
    
    # Hide empty subplots
    for idx in range(n_plots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle(f'{ds_info["title"]} - {method_name} at Various Resolutions', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path is None:
        output_path = outputs_dir / f'{dataset_key}_{method_name.lower()}_multiresolution.png'
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate super-resolution samples from trained FFM models'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default=None,
        help=f'Dataset name. Available: {", ".join(DATASETS.keys())}'
    )
    parser.add_argument(
        '--outputs-dir', '-o',
        type=str,
        default=None,
        help='Path to outputs directory'
    )
    parser.add_argument(
        '--scale', '-s',
        type=int,
        nargs='+',
        default=[2, 4],
        help='Scale factors for super-resolution (default: 2 4)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=100,
        help='Number of samples to generate (default: 50)'
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
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (default: cuda if available)'
    )
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List available datasets and exit'
    )
    parser.add_argument(
        '--method',
        type=str,
        default=None,
        help='Single method to visualize at multiple resolutions'
    )
    parser.add_argument(
        '--resolutions',
        type=int,
        nargs='+',
        default=None,
        help='Custom resolutions for --method mode (e.g., 64 128 256 512)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate super-resolution plots for all datasets'
    )
    parser.add_argument(
        '--train-fresh',
        action='store_true',
        help='Train fresh models if no saved models are found'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of epochs for fresh training (default: 30)'
    )
    parser.add_argument(
        '--ffm-style',
        action='store_true',
        help='Create FFM paper-style plot (methods as rows, resolutions as columns)'
    )
    
    args = parser.parse_args()
    
    if args.list_datasets:
        print("Available datasets:")
        print("\n1D Sequence datasets:")
        for key, info in DATASETS.items():
            if not info['is_2d']:
                print(f"  {key}: {info['title']} (original: {info['original_dims'][0]} pts)")
        print("\n2D PDE datasets:")
        for key, info in DATASETS.items():
            if info['is_2d']:
                print(f"  {key}: {info['title']} (original: {info['original_dims'][0]}x{info['original_dims'][1]})")
        return
    
    # Determine outputs directory
    if args.outputs_dir:
        outputs_dir = Path(args.outputs_dir)
    else:
        script_dir = Path(__file__).parent
        outputs_dir = script_dir.parent / 'outputs'
    
    output_path = Path(args.output) if args.output else None
    
    print(f"Using device: {args.device}")
    
    # Process all datasets or single dataset
    if args.all:
        datasets_to_process = list(DATASETS.keys())
    elif args.dataset:
        datasets_to_process = [args.dataset]
    else:
        parser.print_help()
        return
    
    for dataset_key in datasets_to_process:
        if dataset_key not in DATASETS:
            print(f"Unknown dataset: {dataset_key}")
            continue
        
        ds_info = DATASETS[dataset_key]
        print(f"\n{'='*60}")
        print(f"Processing: {ds_info['title']}")
        print(f"{'='*60}")
        
        if args.ffm_style and not ds_info['is_2d']:
            # FFM paper-style plot
            if args.resolutions:
                resolutions = args.resolutions
            else:
                orig = ds_info['original_dims'][0]
                resolutions = [orig, orig * 2, orig * 4]
            
            fig = create_ffm_style_superres_plot(
                dataset_key=dataset_key,
                outputs_dir=outputs_dir,
                resolutions=resolutions,
                n_samples=args.n_samples,
                seed=args.seed,
                output_path=output_path,
                device=args.device,
                train_fresh=args.train_fresh,
                epochs=args.epochs,
            )
        elif args.method:
            # Single method at multiple resolutions
            if args.resolutions:
                resolutions = args.resolutions
            else:
                # Default resolutions based on original
                orig = ds_info['original_dims'][0]
                resolutions = [orig // 2, orig, orig * 2, orig * 4]
            
            fig = create_single_method_multiresolution_1d(
                dataset_key=dataset_key,
                method_name=args.method,
                outputs_dir=outputs_dir,
                resolutions=resolutions,
                n_samples=args.n_samples,
                seed=args.seed,
                output_path=output_path,
                device=args.device,
            )
        elif ds_info['is_2d']:
            # 2D super-resolution comparison
            # Note: 4x scaling for 64x64 -> 256x256 requires significant GPU memory
            scale_factors = args.scale
            
            fig = create_superres_comparison_2d(
                dataset_key=dataset_key,
                outputs_dir=outputs_dir,
                scale_factors=scale_factors,
                n_samples=min(args.n_samples, 4),  # Fewer samples for 2D
                seed=args.seed,
                output_path=output_path,
                device=args.device,
            )
        else:
            # 1D super-resolution comparison
            fig = create_superres_comparison_1d(
                dataset_key=dataset_key,
                outputs_dir=outputs_dir,
                scale_factors=args.scale,
                n_samples=args.n_samples,
                seed=args.seed,
                output_path=output_path,
                device=args.device,
                train_fresh=args.train_fresh,
                epochs=args.epochs,
            )
        
        if fig:
            plt.close(fig)
    
    print("\n" + "="*60)
    print("Done!")


if __name__ == '__main__':
    main()
