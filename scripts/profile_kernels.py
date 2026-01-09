#!/usr/bin/env python3
"""
Profile kernel-based OT methods by running one epoch and measuring time.

Creates bar plots comparing training time per epoch for each kernel type:
- Independent (no OT)
- Euclidean
- RBF
- Signature

Usage:
    python profile_kernels.py
    python profile_kernels.py --seq_dataset aemet --pde_dataset navier_stokes
    python profile_kernels.py --n_epochs 3
"""

import sys
sys.path.append('../')

import argparse
import json
import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import pandas as pd

from functional_fm_ot import FFMModelOT
from diffusion import DiffusionModel
from models.fno import FNO


# =============================================================================
# Configuration
# =============================================================================

# All methods to profile (FFM variants + diffusion baselines)
KERNEL_TYPES = ['Independent', 'Euclidean', 'RBF', 'Signature', 'DDPM', 'NCSN']

# FFM OT kernel configurations
KERNEL_CONFIGS = {
    'Independent': {
        'use_ot': False,
    },
    'Euclidean': {
        'use_ot': True,
        'ot_method': 'sinkhorn',
        'ot_reg': 0.1,
        'ot_kernel': 'euclidean',
        'ot_coupling': 'sample',
    },
    'RBF': {
        'use_ot': True,
        'ot_method': 'sinkhorn',
        'ot_reg': 0.1,
        'ot_kernel': 'rbf',
        'ot_coupling': 'sample',
        'ot_kernel_params': {'sigma': 5.0},
    },
    'Signature': {
        'use_ot': True,
        'ot_method': 'sinkhorn',
        'ot_reg': 0.1,
        'ot_kernel': 'signature',
        'ot_coupling': 'sample',
        'ot_kernel_params': {
            'time_aug': True,
            'lead_lag': False,
            'dyadic_order': 1,
            'static_kernel_type': 'rbf',
            'static_kernel_sigma': 1.0,
            'add_basepoint': True,
            'normalize': True,
            'max_seq_len': 64,
            'max_batch': 32,
        },
    },
}

# Diffusion model configurations (DDPM, NCSN)
DIFFUSION_CONFIGS = {
    'DDPM': {
        'method': 'DDPM',
        'T': 1000,
        'beta_min': 1e-4,
        'beta_max': 0.02,
    },
    'NCSN': {
        'method': 'NCSN',
        'T': 10,
        'sigma1': 1.0,
        'sigmaT': 0.01,
        'precondition': True,
    },
}

KERNEL_COLORS = {
    'Independent': '#7f7f7f',   # Gray
    'Euclidean': '#1f77b4',     # Blue
    'RBF': '#ff7f0e',           # Orange
    'Signature': '#2ca02c',     # Green
    'DDPM': '#d62728',          # Red
    'NCSN': '#9467bd',          # Purple
}

def is_diffusion_method(method_name: str) -> bool:
    """Check if a method is a diffusion baseline (DDPM/NCSN)."""
    return method_name in ['DDPM', 'NCSN']


# =============================================================================
# Dataset Setup Functions
# =============================================================================

def setup_aemet():
    """Setup AEMET dataset."""
    data_raw = pd.read_csv("../data/aemet.csv", index_col=0)
    data = torch.tensor(data_raw.values, dtype=torch.float32).squeeze().unsqueeze(1)
    
    min_data = torch.min(data)
    max_data = torch.max(data)
    rescaled_data = 6 * (data - min_data) / (max_data - min_data) - 3
    
    n_repeat = 10  # Smaller for profiling
    train_data = rescaled_data.repeat(n_repeat, 1, 1)
    n_x = 365
    
    return {
        'train_data': train_data,
        'n_x': n_x,
        'batch_size': 64,
        'batch_size_sig': 32,
        'modes': 64,
        'width': 256,
        'mlp_width': 128,
        'kernel_length': 0.01,
        'kernel_variance': 0.1,
        'is_2d': False,
        'name': 'AEMET',
    }


def setup_economy():
    """Setup Economy dataset."""
    econ1 = torch.load('../data/economy/econ1.pt').float()
    econ1 = econ1 / torch.mean(econ1, dim=1).unsqueeze(-1)
    
    dmax = torch.max(econ1)
    dmin = torch.min(econ1)
    econ1scaled = -1 + 2 * (econ1 - dmin) / (dmax - dmin)
    
    n_repeat = 10
    train_data = econ1scaled.repeat(n_repeat, 1).unsqueeze(1)
    n_x = econ1scaled.shape[1]
    
    return {
        'train_data': train_data,
        'n_x': n_x,
        'batch_size': 64,
        'batch_size_sig': 32,
        'modes': 16,
        'width': 128,
        'mlp_width': 64,
        'kernel_length': 0.01,
        'kernel_variance': 0.1,
        'is_2d': False,
        'name': 'Economy',
    }


def setup_heston():
    """Setup Heston dataset."""
    from data.Heston import generate_heston_dataset
    
    dataset = generate_heston_dataset(
        n_samples=1000,  # Smaller for profiling
        n_steps=99,
        T=1.0,
        mu=0.05,
        kappa=1.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.7,
        seed=42,
    )
    
    train_data = dataset['log_V_normalized'].unsqueeze(1)
    n_x = train_data.shape[-1]
    
    return {
        'train_data': train_data,
        'n_x': n_x,
        'batch_size': 64,
        'batch_size_sig': 32,
        'modes': 32,
        'width': 256,
        'mlp_width': 128,
        'kernel_length': 0.01,
        'kernel_variance': 0.1,
        'is_2d': False,
        'name': 'Heston',
    }


def setup_navier_stokes():
    """Setup Navier-Stokes dataset."""
    from util.util import load_navier_stokes
    
    data = load_navier_stokes('../data/ns.mat', shuffle=True, subsample_time=5)
    n_total = data.shape[0]
    ntr = min(500, int(0.8 * n_total))  # Smaller for profiling
    train_data = data[:ntr]
    spatial_dims = train_data.shape[2:]
    
    return {
        'train_data': train_data,
        'spatial_dims': spatial_dims,
        'batch_size': 64,
        'batch_size_sig': 64,
        'modes': 16,
        'hch': 32,
        'pch': 64,
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
        'is_2d': True,
        'name': 'Navier-Stokes',
    }


def setup_stochastic_ns():
    """Setup Stochastic NS dataset."""
    from util.util import load_stochastic_ns
    
    data = load_stochastic_ns('../data/stochastic_ns_64.mat', shuffle=True, subsample_time=5)
    n_total = data.shape[0]
    ntr = min(500, int(0.8 * n_total))  # Smaller for profiling
    train_data = data[:ntr]
    spatial_dims = train_data.shape[2:]
    
    return {
        'train_data': train_data,
        'spatial_dims': spatial_dims,
        'batch_size': 64,
        'batch_size_sig': 64,
        'modes': 16,
        'hch': 32,
        'pch': 64,
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
        'is_2d': True,
        'name': 'Stochastic NS',
    }


def setup_expr_genes():
    """Setup Gene Expression dataset."""
    data = torch.load('../data/genes_reduced.pt')
    if isinstance(data, dict):
        data = data.get('data', data.get('X', list(data.values())[0]))
    
    # Normalize
    data = data.float()
    data = (data - data.mean()) / data.std()
    
    n_repeat = 10  # Smaller for profiling
    if data.ndim == 2:
        train_data = data.repeat(n_repeat, 1).unsqueeze(1)
    else:
        train_data = data.repeat(n_repeat, 1, 1)
    n_x = train_data.shape[-1]
    
    return {
        'train_data': train_data,
        'n_x': n_x,
        'batch_size': 64,
        'batch_size_sig': 32,
        'modes': 32,
        'width': 256,
        'mlp_width': 128,
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
        'is_2d': False,
        'name': 'Gene Expr.',
    }


def setup_rbergomi():
    """Setup Rough Bergomi dataset."""
    data = torch.load('../data/rBergomi_H0p10_n5000.pt')
    if isinstance(data, dict):
        train_data = data.get('log_V_normalized', data.get('data', list(data.values())[0]))
    else:
        train_data = data
    
    train_data = train_data.float()
    if train_data.ndim == 2:
        train_data = train_data.unsqueeze(1)
    
    # Use subset for profiling
    train_data = train_data[:1000]
    n_x = train_data.shape[-1]
    
    return {
        'train_data': train_data,
        'n_x': n_x,
        'batch_size': 64,
        'batch_size_sig': 32,
        'modes': 32,
        'width': 256,
        'mlp_width': 128,
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
        'is_2d': False,
        'name': 'rBergomi',
    }


def setup_kdv():
    """Setup KdV dataset."""
    import scipy.io as sio
    
    mat_data = sio.loadmat('../data/KdV.mat')
    # Find the data array in the mat file
    for key in mat_data.keys():
        if not key.startswith('__'):
            data = mat_data[key]
            break
    
    data = torch.tensor(data, dtype=torch.float32)
    if data.ndim == 2:
        data = data.unsqueeze(1)
    
    # Normalize
    data = (data - data.mean()) / data.std()
    
    # Use subset for profiling
    train_data = data[:500]
    n_x = train_data.shape[-1]
    
    return {
        'train_data': train_data,
        'n_x': n_x,
        'batch_size': 64,
        'batch_size_sig': 32,
        'modes': 32,
        'width': 256,
        'mlp_width': 128,
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
        'is_2d': False,
        'name': 'KdV',
    }


def setup_stochastic_kdv():
    """Setup Stochastic KdV dataset."""
    import scipy.io as sio
    
    mat_data = sio.loadmat('../data/stochastic_kdv.mat')
    # Find the data array in the mat file
    for key in mat_data.keys():
        if not key.startswith('__'):
            data = mat_data[key]
            break
    
    data = torch.tensor(data, dtype=torch.float32)
    if data.ndim == 2:
        data = data.unsqueeze(1)
    
    # Normalize
    data = (data - data.mean()) / data.std()
    
    # Use subset for profiling
    train_data = data[:500]
    n_x = train_data.shape[-1]
    
    return {
        'train_data': train_data,
        'n_x': n_x,
        'batch_size': 64,
        'batch_size_sig': 32,
        'modes': 64,
        'width': 256,
        'mlp_width': 128,
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
        'is_2d': False,
        'name': 'Stochastic KdV',
    }


DATASET_SETUP = {
    'aemet': setup_aemet,
    'economy': setup_economy,
    'heston': setup_heston,
    'navier_stokes': setup_navier_stokes,
    'stochastic_ns': setup_stochastic_ns,
    'expr_genes': setup_expr_genes,
    'rbergomi': setup_rbergomi,
    'kdv': setup_kdv,
    'stochastic_kdv': setup_stochastic_kdv,
}

# Datasets for the 3x3 combined plot (in order)
DATASETS_3X3 = [
    'aemet', 'expr_genes', 'economy',
    'heston', 'rbergomi', 'kdv',
    'navier_stokes', 'stochastic_kdv', 'stochastic_ns',
]


# =============================================================================
# Model Creation
# =============================================================================

def create_model_1d(modes, width, mlp_width, device):
    """Create 1D FNO model."""
    return FNO(
        modes,
        vis_channels=1,
        hidden_channels=width,
        proj_channels=mlp_width,
        x_dim=1,
        t_scaling=1000
    ).to(device)


def create_model_2d(modes, hch, pch, device):
    """Create 2D FNO model."""
    return FNO(
        modes,
        vis_channels=1,
        hidden_channels=hch,
        proj_channels=pch,
        x_dim=2,
        t_scaling=1000.0
    ).to(device)


def build_ffm_kwargs(config: dict, setup: dict, device: str) -> dict:
    """Build kwargs for FFMModelOT from config dict."""
    kwargs = {
        'kernel_length': setup['kernel_length'],
        'kernel_variance': setup['kernel_variance'],
        'sigma_min': 1e-4,
        'device': device,
        'dtype': torch.float32,
        'use_ot': config.get('use_ot', False),
    }
    
    if config.get('use_ot', False):
        if 'ot_method' in config:
            kwargs['ot_method'] = config['ot_method']
        if 'ot_reg' in config:
            kwargs['ot_reg'] = config['ot_reg']
        if 'ot_kernel' in config:
            kwargs['ot_kernel'] = config['ot_kernel']
        if 'ot_coupling' in config:
            kwargs['ot_coupling'] = config['ot_coupling']
        if 'ot_kernel_params' in config:
            kwargs['ot_kernel_params'] = config['ot_kernel_params']
    
    return kwargs


# =============================================================================
# Profiling
# =============================================================================

def profile_ffm_kernel(
    kernel_type: str,
    setup: dict,
    device: str,
    n_epochs: int = 1,
    warmup: bool = True,
) -> Dict:
    """
    Profile an FFM kernel type (Independent, Euclidean, RBF, Signature).
    
    Returns dict with timing information.
    """
    config = KERNEL_CONFIGS[kernel_type]
    is_2d = setup.get('is_2d', False)
    
    # Skip signature for 2D datasets
    if kernel_type == 'Signature' and is_2d:
        return {'time_per_epoch': None, 'total_time': None, 'skipped': True}
    
    print(f"  Profiling {kernel_type}...")
    
    # Create model
    if is_2d:
        model = create_model_2d(
            setup['modes'], setup['hch'], setup['pch'], device
        )
    else:
        model = create_model_1d(
            setup['modes'], setup['width'], setup['mlp_width'], device
        )
    
    # Create FFM
    ffm_kwargs = build_ffm_kwargs(config, setup, device)
    ffm = FFMModelOT(model, **ffm_kwargs)
    
    # Create data loader
    batch_size = setup['batch_size_sig'] if kernel_type == 'Signature' else setup['batch_size']
    train_loader = DataLoader(setup['train_data'], batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Helper function to compute loss (replicates FFMModelOT.train logic)
    def compute_batch_loss(batch, ffm, model, train_dims, n_channels):
        """Compute loss for a single batch (replicates FFMModelOT.train logic)."""
        batch_size = batch.shape[0]
        dtype = ffm.dtype
        
        # Sample base GP noise
        z_noise = ffm.sample_base(batch_size, n_channels, train_dims)
        z_noise = z_noise.to(dtype)
        
        # OT pairing (or independent if use_ot=False)
        x_data, z_paired = ffm.pair_samples(batch, z_noise)
        
        # Sample time t ~ Unif[0, 1)
        t = torch.rand(batch_size, device=device)
        
        # Sample from probability path p_t(x | x_data, z_paired)
        x_noisy = ffm.simulate(t, x_data, z_paired)
        
        # Get conditional vector field target
        target = ffm.get_conditional_fields(t, x_data, x_noisy, z_paired)
        
        x_noisy = x_noisy.to(device)
        target = target.to(device)
        
        # Forward pass and loss
        model_out = model(t, x_noisy)
        loss = torch.mean((model_out - target) ** 2)
        
        return loss
    
    # Determine train_dims and n_channels from data
    sample_batch = next(iter(train_loader))
    n_channels = sample_batch.shape[1]
    train_dims = sample_batch.shape[2:]
    
    # Warmup run (not timed)
    if warmup:
        try:
            for batch in train_loader:
                batch = batch.to(device).to(ffm.dtype)
                optimizer.zero_grad()
                loss = compute_batch_loss(batch, ffm, model, train_dims, n_channels)
                loss.backward()
                optimizer.step()
                break  # Just one batch for warmup
        except Exception as e:
            print(f"    Warmup failed: {e}")
    
    # Timed run
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    epoch_times = []
    for epoch in range(n_epochs):
        start_time = time.perf_counter()
        
        for batch in train_loader:
            batch = batch.to(device).to(ffm.dtype)
            optimizer.zero_grad()
            loss = compute_batch_loss(batch, ffm, model, train_dims, n_channels)
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)
        print(f"    Epoch {epoch + 1}: {epoch_time:.2f}s")
    
    avg_time = np.mean(epoch_times)
    std_time = np.std(epoch_times) if len(epoch_times) > 1 else 0
    
    return {
        'time_per_epoch': avg_time,
        'time_std': std_time,
        'total_time': sum(epoch_times),
        'epoch_times': epoch_times,
        'skipped': False,
    }


def profile_diffusion(
    method_name: str,
    setup: dict,
    device: str,
    n_epochs: int = 1,
    warmup: bool = True,
) -> Dict:
    """
    Profile a diffusion model (DDPM or NCSN).
    
    Returns dict with timing information.
    """
    config = DIFFUSION_CONFIGS[method_name]
    is_2d = setup.get('is_2d', False)
    
    print(f"  Profiling {method_name}...")
    
    # Create model
    if is_2d:
        model = create_model_2d(
            setup['modes'], setup['hch'], setup['pch'], device
        )
    else:
        model = create_model_1d(
            setup['modes'], setup['width'], setup['mlp_width'], device
        )
    
    # Create diffusion model wrapper
    method = config["method"]
    if method == "DDPM":
        diffusion = DiffusionModel(
            model, 
            method=method, 
            T=config["T"],
            device=device,
            kernel_length=setup['kernel_length'], 
            kernel_variance=setup['kernel_variance'],
            beta_min=config["beta_min"], 
            beta_max=config["beta_max"],
            dtype=torch.float32,
        )
    else:  # NCSN
        diffusion = DiffusionModel(
            model, 
            method=method, 
            T=config["T"],
            device=device,
            kernel_length=setup['kernel_length'], 
            kernel_variance=setup['kernel_variance'],
            sigma1=config["sigma1"], 
            sigmaT=config["sigmaT"], 
            precondition=config.get("precondition", True),
            dtype=torch.float32,
        )
    
    # Create data loader
    batch_size = setup['batch_size']
    train_loader = DataLoader(setup['train_data'], batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Determine train_dims from data
    sample_batch = next(iter(train_loader))
    train_dims = sample_batch.shape[2:]
    
    # Set support for loss calculation (must be on same device as model)
    from util.util import make_grid
    diffusion.train_support = make_grid(train_dims).to(device)
    diffusion.make_loss()
    
    # Warmup run (not timed)
    if warmup:
        try:
            for batch in train_loader:
                batch = batch.to(device).to(diffusion.dtype)
                batch_size_curr = batch.shape[0]
                
                # Sample random timesteps
                t = torch.randint(1, diffusion.T + 1, (batch_size_curr,), device=device)
                
                # Forward process
                u_t, noise = diffusion.simulate_fwd_process(batch, t, return_noise=True)
                
                # Model prediction (normalize t to [0, 1])
                model_out = model(t.float() / diffusion.T, u_t)
                
                # Compute loss: loss_fxn(noise, model_output)
                loss = diffusion.loss_fxn(noise, model_out)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                break  # Just one batch for warmup
        except Exception as e:
            print(f"    Warmup failed: {e}")
    
    # Timed run
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    epoch_times = []
    for epoch in range(n_epochs):
        start_time = time.perf_counter()
        
        for batch in train_loader:
            batch = batch.to(device).to(diffusion.dtype)
            batch_size_curr = batch.shape[0]
            
            # Sample random timesteps
            t = torch.randint(1, diffusion.T + 1, (batch_size_curr,), device=device)
            
            # Forward process
            u_t, noise = diffusion.simulate_fwd_process(batch, t, return_noise=True)
            
            # Model prediction (normalize t to [0, 1])
            model_out = model(t.float() / diffusion.T, u_t)
            
            # Compute loss: loss_fxn(noise, model_output)
            loss = diffusion.loss_fxn(noise, model_out)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)
        print(f"    Epoch {epoch + 1}: {epoch_time:.2f}s")
    
    avg_time = np.mean(epoch_times)
    std_time = np.std(epoch_times) if len(epoch_times) > 1 else 0
    
    return {
        'time_per_epoch': avg_time,
        'time_std': std_time,
        'total_time': sum(epoch_times),
        'epoch_times': epoch_times,
        'skipped': False,
    }


def profile_kernel(
    kernel_type: str,
    setup: dict,
    device: str,
    n_epochs: int = 1,
    warmup: bool = True,
) -> Dict:
    """
    Profile a single kernel/method type (dispatches to appropriate function).
    
    Returns dict with timing information.
    """
    if is_diffusion_method(kernel_type):
        return profile_diffusion(kernel_type, setup, device, n_epochs, warmup)
    else:
        return profile_ffm_kernel(kernel_type, setup, device, n_epochs, warmup)


def profile_dataset(
    dataset_key: str,
    device: str,
    n_epochs: int = 1,
) -> Dict[str, Dict]:
    """Profile all kernels for a dataset."""
    setup_fn = DATASET_SETUP.get(dataset_key)
    if setup_fn is None:
        print(f"Unknown dataset: {dataset_key}")
        return {}
    
    print(f"\nSetting up {dataset_key}...")
    setup = setup_fn()
    
    results = {}
    for kernel_type in KERNEL_TYPES:
        result = profile_kernel(kernel_type, setup, device, n_epochs)
        results[kernel_type] = result
    
    return results, setup.get('name', dataset_key)


# =============================================================================
# Plotting
# =============================================================================

def create_barplot(
    results: Dict[str, Dict],
    dataset_name: str,
    output_path: Path,
    figsize: Tuple[float, float] = (8, 5),
):
    """Create a bar plot of profiling results."""
    # Filter out skipped kernels
    kernels = [k for k in KERNEL_TYPES if k in results and not results[k].get('skipped', False)]
    times = [results[k]['time_per_epoch'] for k in kernels]
    stds = [results[k].get('time_std', 0) for k in kernels]
    colors = [KERNEL_COLORS[k] for k in kernels]
    
    if not kernels:
        print(f"  No data to plot for {dataset_name}")
        return False
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(kernels))
    bars = ax.bar(x, times, yerr=stds, capsize=5, color=colors, 
                  edgecolor='black', linewidth=1, alpha=0.85)
    
    # Add time labels on bars
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.annotate(f'{t:.2f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Kernel Type', fontsize=12)
    ax.set_ylabel('Time per Epoch (seconds)', fontsize=12)
    ax.set_title(f'Training Time per Epoch: {dataset_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(kernels, fontsize=11)
    
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Extend y-axis for labels
    ymax = max(times) * 1.2
    ax.set_ylim(0, ymax)
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {output_path}")
    return True


def create_combined_barplot(
    all_results: Dict[str, Tuple[Dict, str]],
    output_path: Path,
    figsize: Tuple[float, float] = (12, 5),
):
    """Create a combined bar plot for multiple datasets."""
    datasets = list(all_results.keys())
    n_datasets = len(datasets)
    
    fig, axes = plt.subplots(1, n_datasets, figsize=figsize)
    if n_datasets == 1:
        axes = [axes]
    
    for ax, dataset_key in zip(axes, datasets):
        results, dataset_name = all_results[dataset_key]
        
        # Filter out skipped kernels
        kernels = [k for k in KERNEL_TYPES if k in results and not results[k].get('skipped', False)]
        times = [results[k]['time_per_epoch'] for k in kernels]
        stds = [results[k].get('time_std', 0) for k in kernels]
        colors = [KERNEL_COLORS[k] for k in kernels]
        
        x = np.arange(len(kernels))
        bars = ax.bar(x, times, yerr=stds, capsize=4, color=colors,
                      edgecolor='black', linewidth=1, alpha=0.85)
        
        # Add time labels
        for bar, t in zip(bars, times):
            height = bar.get_height()
            ax.annotate(f'{t:.1f}s',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Kernel Type', fontsize=11)
        ax.set_ylabel('Time (s)', fontsize=11)
        ax.set_title(dataset_name, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(kernels, fontsize=9, rotation=15, ha='right')
        
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Extend y-axis
        if times:
            ax.set_ylim(0, max(times) * 1.25)
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")


def create_3x3_combined_plot(
    all_results: Dict[str, Tuple[Dict, str]],
    output_path: Path,
    figsize: Tuple[float, float] = (16, 14),
):
    """Create a 3x3 combined bar plot for all 9 datasets."""
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    axes = axes.flatten()
    
    for idx, dataset_key in enumerate(DATASETS_3X3):
        ax = axes[idx]
        
        if dataset_key not in all_results:
            ax.text(0.5, 0.5, f'{dataset_key}\nNo data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        results, dataset_name = all_results[dataset_key]
        
        # Filter out skipped kernels
        kernels = [k for k in KERNEL_TYPES if k in results and not results[k].get('skipped', False)]
        times = [results[k]['time_per_epoch'] for k in kernels]
        stds = [results[k].get('time_std', 0) for k in kernels]
        colors = [KERNEL_COLORS[k] for k in kernels]
        
        if not kernels:
            ax.text(0.5, 0.5, f'{dataset_name}\nNo data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        x = np.arange(len(kernels))
        bars = ax.bar(x, times, yerr=stds, capsize=3, color=colors,
                      edgecolor='black', linewidth=0.8, alpha=0.85)
        
        # Add time labels on bars
        for bar, t in zip(bars, times):
            height = bar.get_height()
            ax.annotate(f'{t:.1f}s',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=7, fontweight='bold')
        
        ax.set_title(dataset_name, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(kernels, fontsize=8, rotation=30, ha='right')
        ax.set_ylabel('Time (s)', fontsize=10)
        
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Extend y-axis for labels
        if times:
            ax.set_ylim(0, max(times) * 1.3)
    
    plt.suptitle('Training Time per Epoch: All Datasets', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Profile kernel-based OT methods'
    )
    parser.add_argument('--seq_dataset', type=str, default='aemet',
                        choices=list(DATASET_SETUP.keys()),
                        help='Sequence dataset to profile (default: aemet)')
    parser.add_argument('--pde_dataset', type=str, default='navier_stokes',
                        choices=['navier_stokes', 'stochastic_ns'],
                        help='PDE dataset to profile (default: navier_stokes)')
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs to run (default: 1)')
    parser.add_argument('--output_dir', type=str, default='../outputs',
                        help='Output directory for plots')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--skip_pde', action='store_true',
                        help='Skip PDE dataset profiling')
    parser.add_argument('--skip_seq', action='store_true',
                        help='Skip sequence dataset profiling')
    parser.add_argument('--all-3x3', action='store_true',
                        help='Profile all 9 datasets and generate a 3x3 combined plot')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("Kernel Profiling")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {args.n_epochs}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Mode: Profile all 9 datasets for 3x3 plot
    if getattr(args, 'all_3x3', False):
        print(f"\n{'=' * 60}")
        print("Profiling ALL 9 datasets for 3x3 combined plot")
        print(f"Datasets: {', '.join(DATASETS_3X3)}")
        print("=" * 60)
        
        for dataset_key in DATASETS_3X3:
            print(f"\n{'=' * 60}")
            print(f"Profiling: {dataset_key}")
            print("=" * 60)
            
            try:
                results, name = profile_dataset(dataset_key, device, args.n_epochs)
                all_results[dataset_key] = (results, name)
                
                # Individual plot
                output_path = output_dir / f"profile_{dataset_key}.png"
                create_barplot(results, name, output_path)
            except Exception as e:
                print(f"  Error profiling {dataset_key}: {e}")
                continue
        
        # Create 3x3 combined plot
        print(f"\n{'=' * 60}")
        print("Generating 3x3 Combined Plot")
        print("=" * 60)
        
        combined_3x3_path = output_dir / "profile_all_3x3.png"
        create_3x3_combined_plot(all_results, combined_3x3_path)
    
    else:
        # Original mode: profile selected seq and pde datasets
        
        # Profile sequence dataset
        if not args.skip_seq:
            print(f"\n{'=' * 60}")
            print(f"Profiling Sequence Dataset: {args.seq_dataset}")
            print("=" * 60)
            
            results, name = profile_dataset(args.seq_dataset, device, args.n_epochs)
            all_results[args.seq_dataset] = (results, name)
            
            # Individual plot
            output_path = output_dir / f"profile_{args.seq_dataset}.png"
            create_barplot(results, name, output_path)
        
        # Profile PDE dataset
        if not args.skip_pde:
            print(f"\n{'=' * 60}")
            print(f"Profiling PDE Dataset: {args.pde_dataset}")
            print("=" * 60)
            
            results, name = profile_dataset(args.pde_dataset, device, args.n_epochs)
            all_results[args.pde_dataset] = (results, name)
            
            # Individual plot
            output_path = output_dir / f"profile_{args.pde_dataset}.png"
            create_barplot(results, name, output_path)
        
        # Combined plot
        if len(all_results) > 1:
            print(f"\n{'=' * 60}")
            print("Generating Combined Plot")
            print("=" * 60)
            
            combined_path = output_dir / "profile_combined.png"
            create_combined_barplot(all_results, combined_path)
    
    # Save raw results
    results_path = output_dir / "profile_results.json"
    save_results = {}
    for k, (r, n) in all_results.items():
        save_results[k] = {
            'name': n,
            'results': {kk: {kk2: vv2 for kk2, vv2 in vv.items() if kk2 != 'epoch_times'} 
                       for kk, vv in r.items()}
        }
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
