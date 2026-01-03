"""
Comprehensive OT-FFM Experiments on AEMET Weather Data.

This script runs experiments with multiple OT configurations on the AEMET
dataset (Spanish weather station temperature data) comparing:
- Different kernel types (euclidean, rbf, signature)
- Different OT methods (exact, sinkhorn)
- Different coupling strategies (sample, barycentric)

Each configuration gets its own subdirectory with:
- Trained model checkpoint
- Generated samples
- Training metrics (loss, path length, gradient variance)
- Generation quality statistics (Mean, Variance, Skewness, Kurtosis, Autocorrelation MSE)

Usage:
    python AEMET_ot.py                    # Run all experiments
    python AEMET_ot.py --load-only        # Load saved results and regenerate plots
    python AEMET_ot.py --load-only --spath ../outputs/my_run/  # Load from custom path
"""

import sys
sys.path.append('../')

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Any, Tuple
import pandas as pd

# =============================================================================
# Argument Parsing
# =============================================================================

parser = argparse.ArgumentParser(description='AEMET OT-FFM Experiments')
parser.add_argument('--load-only', action='store_true',
                    help='Load saved results instead of training (for regenerating plots)')
parser.add_argument('--baselines-only', action='store_true',
                    help='Run only DDPM and NCSN baselines (not OT experiments)')
parser.add_argument('--spath', type=str, default='../outputs/AEMET_ot_comprehensive/',
                    help='Output/load directory')
args, _ = parser.parse_known_args()

from util.gaussian_process import GPPrior
from util.util import make_grid
from util.ot_monitoring import (
    TrainingMonitor,
    TrainingMetrics,
    compare_training_runs,
    compare_training_runs_simplified,
    plot_convergence_comparison,
    print_comparison_table,
)
from util.eval import (
    GenerationQualityMetrics,
    compare_generation_quality,
    compare_generation_quality_simplified,
    print_quality_table,
    print_spectrum_table,
    print_seasonal_table,
    print_comprehensive_table,
    compute_all_pointwise_statistics,
    compare_spectra_1d,
    compare_seasonal_patterns,
    compare_convergence,
    compare_convergence_simplified,
    print_convergence_table,
    save_all_metrics_summary,
)

from functional_fm_ot import FFMModelOT
from diffusion import DiffusionModel
from models.fno import FNO

# =============================================================================
# Load and Preprocess Data
# =============================================================================

print("Loading AEMET weather data...")
data_raw = pd.read_csv("../data/aemet.csv", index_col=0)
data = torch.tensor(data_raw.values, dtype=torch.float32).squeeze().unsqueeze(1)
x_grid = torch.linspace(0, 1, 365)

print(f"AEMET data shape: {data.shape}")
print(f"Number of weather stations: {data.shape[0]}")
print(f"Number of days: {data.shape[2] if len(data.shape) > 2 else data.shape[1]}")

# Rescale data to [-3, 3] range
min_data = torch.min(data)
max_data = torch.max(data)
rescaled_data = 6 * (data - min_data) / (max_data - min_data) - 3

# Repeat data to have enough samples for training
n_repeat = 50
rescaled_data_repeated = rescaled_data.repeat(n_repeat, 1, 1)

n_x = 365  # Number of time points (days in a year)

def original_scale(samples):
    """Convert samples back to original temperature scale."""
    return (samples + 3.) * (max_data - min_data) / 6 + min_data

# =============================================================================
# Configuration
# =============================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Batch sizes - signature kernel needs smaller batch due to memory
batch_size_default = 512  # For euclidean/rbf OT
batch_size_signature = 64  # Reduced for signature kernel (365 time points is memory-intensive)
num_workers = 0
pin_memory = True

# We'll create dataloaders per-config to handle different batch sizes
def create_dataloader(batch_size):
    return DataLoader(
        rescaled_data_repeated,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

# Model params
modes = 64
width = 256
mlp_width = 128

# GP hyperparameters
kernel_length = 0.01
kernel_variance = 0.1

# Training params
epochs = 50
lr = 1e-3

# FFM params
sigma_min = 1e-4

# Number of random seeds
n_seeds = 3
random_seeds = [2**i for i in range(n_seeds)]

# Number of samples to generate
n_gen_samples = 500

# Output directory
spath = Path(args.spath)
spath.mkdir(parents=True, exist_ok=True)

# =============================================================================
# OT Configurations to Compare
# =============================================================================

OT_CONFIGS = {
    # =========================================================================
    # Baseline: Independent pairing (no OT)
    # =========================================================================
    "independent": {
        "use_ot": False,
    },
    
    # =========================================================================
    # Gaussian OT (Bures-Wasserstein) - closed-form, fast, stable
    # Note: Higher regularization needed for high-dimensional data (365 dims)
    # to ensure covariance matrix stability in Bures-Wasserstein mapping.
    # =========================================================================
    "gaussian_ot": {
        "use_ot": True,
        "ot_method": "gaussian",
        "ot_reg": 0.1,  # Higher reg for 365-dim covariance stability
        "ot_coupling": "barycentric",
    },
    
    # =========================================================================
    # Euclidean OT - All Methods
    # =========================================================================
    "euclidean_exact": {
        "use_ot": True,
        "ot_method": "exact",
        "ot_kernel": "euclidean",
        "ot_coupling": "sample",
    },
    "euclidean_sinkhorn_reg0.1": {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.1,
        "ot_kernel": "euclidean",
        "ot_coupling": "sample",
    },
    "euclidean_sinkhorn_reg0.5": {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.5,
        "ot_kernel": "euclidean",
        "ot_coupling": "sample",
    },
    "euclidean_sinkhorn_reg1.0": {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 1.0,
        "ot_kernel": "euclidean",
        "ot_coupling": "sample",
    },
    # Note: unbalanced and partial OT are not included because they
    # don't preserve marginal constraints required for flow matching.
    
    # =========================================================================
    # RBF Kernel OT - All Methods
    # =========================================================================
    "rbf_exact": {
        "use_ot": True,
        "ot_method": "exact",
        "ot_kernel": "rbf",
        "ot_coupling": "sample",
        "ot_kernel_params": {"sigma": 5.0},
    },
    "rbf_sinkhorn_reg0.1": {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.1,
        "ot_kernel": "rbf",
        "ot_coupling": "sample",
        "ot_kernel_params": {"sigma": 5.0},
    },
    "rbf_sinkhorn_reg0.5": {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.5,
        "ot_kernel": "rbf",
        "ot_coupling": "sample",
        "ot_kernel_params": {"sigma": 5.0},
    },
    "rbf_sinkhorn_reg1.0": {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 1.0,
        "ot_kernel": "rbf",
        "ot_coupling": "sample",
        "ot_kernel_params": {"sigma": 5.0},
    },
    "rbf_sinkhorn_barycentric": {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.1,
        "ot_kernel": "rbf",
        "ot_coupling": "barycentric",
        "ot_kernel_params": {"sigma": 5.0},
    },
    # Note: unbalanced and partial OT are not included because they
    # don't preserve marginal constraints required for flow matching.
    
    # =========================================================================
    # Signature Kernel OT - All Methods (for time series)
    # Note: Uses smaller batch size (set via _batch_size) due to memory
    # =========================================================================
    "signature_sinkhorn_reg0.1": {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.1,
        "ot_kernel": "signature",
        "ot_coupling": "sample",
        "ot_kernel_params": {
            "time_aug": True,
            "lead_lag": False,
            "dyadic_order": 1,
            "static_kernel_type": "rbf",
            "static_kernel_sigma": 1.0,
            "add_basepoint": True,
            "normalize": True,
            "max_seq_len": 64,
            "max_batch": 32,
        },
        "_batch_size": batch_size_signature,
    },
    "signature_sinkhorn_reg0.5": {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.5,
        "ot_kernel": "signature",
        "ot_coupling": "sample",
        "ot_kernel_params": {
            "time_aug": True,
            "lead_lag": False,
            "dyadic_order": 1,
            "static_kernel_type": "rbf",
            "static_kernel_sigma": 1.0,
            "add_basepoint": True,
            "normalize": True,
            "max_seq_len": 64,
            "max_batch": 32,
        },
        "_batch_size": batch_size_signature,
    },
    "signature_sinkhorn_reg1.0": {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 1.0,
        "ot_kernel": "signature",
        "ot_coupling": "sample",
        "ot_kernel_params": {
            "time_aug": True,
            "lead_lag": False,
            "dyadic_order": 1,
            "static_kernel_type": "rbf",
            "static_kernel_sigma": 1.0,
            "add_basepoint": True,
            "normalize": True,
            "max_seq_len": 64,
            "max_batch": 32,
        },
        "_batch_size": batch_size_signature,
    },
    "signature_sinkhorn_barycentric": {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.1,
        "ot_kernel": "signature",
        "ot_coupling": "barycentric",
        "ot_kernel_params": {
            "time_aug": True,
            "lead_lag": False,
            "dyadic_order": 1,
            "static_kernel_type": "rbf",
            "static_kernel_sigma": 1.0,
            "add_basepoint": True,
            "normalize": True,
            "max_seq_len": 64,
            "max_batch": 32,
        },
        "_batch_size": batch_size_signature,
    },
}

# =============================================================================
# Baseline Configurations (DDPM, NCSN) - Not OT methods
# =============================================================================

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

# =============================================================================
# Training Functions
# =============================================================================

def create_model(device):
    """Create FNO model."""
    return FNO(
        modes, 
        vis_channels=1, 
        hidden_channels=width, 
        proj_channels=mlp_width, 
        x_dim=1, 
        t_scaling=1000
    ).to(device)


def build_ffm_kwargs(config: dict) -> dict:
    """Build kwargs for FFMModelOT from config dict."""
    kwargs = {
        "kernel_length": kernel_length,
        "kernel_variance": kernel_variance,
        "sigma_min": sigma_min,
        "device": device,
        "dtype": torch.float32,
        "use_ot": config.get("use_ot", False),
    }
    
    if config.get("use_ot", False):
        if "ot_method" in config:
            kwargs["ot_method"] = config["ot_method"]
        if "ot_reg" in config:
            kwargs["ot_reg"] = config["ot_reg"]
        if "ot_reg_m" in config:
            kwargs["ot_reg_m"] = config["ot_reg_m"]  # For unbalanced OT
        if "ot_kernel" in config:
            kwargs["ot_kernel"] = config["ot_kernel"]
        if "ot_coupling" in config:
            kwargs["ot_coupling"] = config["ot_coupling"]
        if "ot_kernel_params" in config:
            kwargs["ot_kernel_params"] = config["ot_kernel_params"]
    
    return kwargs


def train_single_config(
    config_name: str,
    config: dict,
    seed: int,
    train_loader: DataLoader,
    ground_truth: torch.Tensor,
    save_dir: Path,
) -> Dict[str, Any]:
    """Train a single configuration with a single seed."""
    
    print(f"\n  Training {config_name} (seed={seed})...")
    
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create model
    model = create_model(device)
    
    # Create FFM wrapper
    ffm_kwargs = build_ffm_kwargs(config)
    ffm = FFMModelOT(model, **ffm_kwargs)
    
    # Create monitor
    monitor = TrainingMonitor(
        method_name=config_name,
        use_ot=config.get("use_ot", False),
        ot_kernel=config.get("ot_kernel", "") or "",
        track_batch_metrics=False,
    )
    
    # Train
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25)
    
    training_metrics = ffm.train(
        train_loader=train_loader,
        optimizer=optimizer,
        epochs=epochs,
        scheduler=scheduler,
        eval_int=0,
        save_int=epochs,
        generate=False,
        save_path=save_dir,
        monitor=monitor,
    )
    
    # Generate samples
    print(f"    Generating {n_gen_samples} samples...")
    samples = ffm.sample([n_x], n_samples=n_gen_samples).cpu().squeeze()
    
    # Convert back to original scale for quality comparison
    samples_original_scale = original_scale(samples)
    ground_truth_original_scale = original_scale(ground_truth.squeeze())
    
    # Compute generation quality statistics
    print(f"    Computing quality statistics...")
    quality_metrics = GenerationQualityMetrics(
        config_name=config_name,
        ot_kernel=config.get("ot_kernel", "") or "",
        ot_method=config.get("ot_method", "") or "",
        ot_coupling=config.get("ot_coupling", "") or "",
        use_ot=config.get("use_ot", False),
    )
    quality_metrics.compute_from_samples(ground_truth_original_scale, samples_original_scale)
    
    # Add training metrics
    if training_metrics is not None:
        if training_metrics.train_losses:
            quality_metrics.final_train_loss = training_metrics.train_losses[-1]
            # Set convergence metrics
            quality_metrics.set_convergence_metrics(training_metrics.train_losses)
        if training_metrics.epoch_times:
            quality_metrics.total_train_time = sum(training_metrics.epoch_times)
        if training_metrics.path_lengths:
            quality_metrics.mean_path_length = np.mean(training_metrics.path_lengths)
        if training_metrics.gradient_variances:
            quality_metrics.mean_grad_variance = np.mean(training_metrics.gradient_variances)
    
    # Save everything
    torch.save(samples, save_dir / 'samples.pt')
    torch.save(samples_original_scale, save_dir / 'samples_original_scale.pt')
    torch.save(model.state_dict(), save_dir / 'model.pt')
    if training_metrics is not None:
        training_metrics.save(save_dir / 'training_metrics.json')
    quality_metrics.save(save_dir / 'quality_metrics.json')
    
    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    return {
        'samples': samples,
        'samples_original_scale': samples_original_scale,
        'training_metrics': training_metrics,
        'quality_metrics': quality_metrics,
    }


def train_baseline_config(
    config_name: str,
    config: dict,
    seed: int,
    train_loader: DataLoader,
    ground_truth: torch.Tensor,
    save_dir: Path,
) -> Dict[str, Any]:
    """
    Train a baseline (DDPM/NCSN) configuration with a single seed.
    """
    import time
    
    print(f"\n  Training {config_name} (seed={seed})...")
    
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create model
    model = create_model(device)
    
    # Create diffusion model wrapper
    method = config["method"]
    if method == "DDPM":
        diffusion = DiffusionModel(
            model, 
            method=method, 
            T=config["T"],
            device=device,
            kernel_length=kernel_length, 
            kernel_variance=kernel_variance,
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
            kernel_length=kernel_length, 
            kernel_variance=kernel_variance,
            sigma1=config["sigma1"], 
            sigmaT=config["sigmaT"], 
            precondition=config.get("precondition", True),
            dtype=torch.float32,
        )
    
    # Train
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25)
    
    t0 = time.time()
    diffusion.train(
        train_loader=train_loader,
        optimizer=optimizer,
        epochs=epochs,
        scheduler=scheduler,
        eval_int=0,
        save_int=epochs,
        generate=False,
        save_path=save_dir,
    )
    train_time = time.time() - t0
    
    # Generate samples
    print(f"    Generating {n_gen_samples} samples...")
    samples = diffusion.sample([n_x], n_samples=n_gen_samples, n_channels=1)
    samples = samples.cpu().squeeze()
    
    # Convert back to original scale for quality comparison
    samples_original_scale = original_scale(samples)
    ground_truth_original_scale = original_scale(ground_truth.squeeze())
    
    # Compute generation quality statistics
    print(f"    Computing quality statistics...")
    quality_metrics = GenerationQualityMetrics(
        config_name=config_name,
        ot_kernel="",
        ot_method="",
        ot_coupling="",
        use_ot=False,
    )
    quality_metrics.compute_from_samples(ground_truth_original_scale, samples_original_scale)
    quality_metrics.total_train_time = train_time
    
    # Save everything
    torch.save(samples, save_dir / 'samples.pt')
    torch.save(samples_original_scale, save_dir / 'samples_original_scale.pt')
    torch.save(model.state_dict(), save_dir / 'model.pt')
    quality_metrics.save(save_dir / 'quality_metrics.json')
    
    # Save config for reference
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    return {
        'samples': samples,
        'samples_original_scale': samples_original_scale,
        'training_metrics': None,  # Baselines don't track OT-specific metrics
        'quality_metrics': quality_metrics,
    }


def run_baseline_experiments() -> Dict[str, Dict[str, Any]]:
    """Run baseline experiments (DDPM, NCSN) with all seeds."""
    all_results = {}
    
    # Create data loader using the same data as OT experiments
    loader = create_dataloader(batch_size_default)
    
    for config_name, config in BASELINE_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Baseline: {config_name}")
        print(f"{'='*60}")
        
        config_results = {}
        
        for seed in random_seeds:
            # Create subdirectory for this config + seed
            config_dir = spath / config_name / f"seed_{seed}"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            result = train_baseline_config(
                config_name=config_name,
                config=config,
                seed=seed,
                train_loader=loader,
                ground_truth=rescaled_data,
                save_dir=config_dir,
            )
            
            config_results[seed] = result
        
        all_results[config_name] = config_results
    
    return all_results


def run_all_experiments() -> Dict[str, Dict[int, Dict]]:
    """Run all configurations with all seeds."""
    
    all_results = {}
    
    for config_name, config in OT_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Configuration: {config_name}")
        print(f"{'='*60}")
        
        # Get config-specific batch size (signature needs smaller batch for memory)
        config_batch_size = config.get("_batch_size", batch_size_default)
        print(f"Using batch size: {config_batch_size}")
        
        # Create dataloader for this config
        train_loader = create_dataloader(config_batch_size)
        
        config_results = {}
        
        for seed in random_seeds:
            # Create subdirectory
            config_dir = spath / config_name / f"seed_{seed}"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            result = train_single_config(
                config_name=config_name,
                config=config,
                seed=seed,
                train_loader=train_loader,
                ground_truth=rescaled_data,
                save_dir=config_dir,
            )
            
            config_results[seed] = result
        
        all_results[config_name] = config_results
    
    return all_results


def load_all_results(include_baselines: bool = True) -> Dict[str, Dict[int, Dict]]:
    """Load all results from saved files instead of training.
    
    This function loads samples, training metrics, and quality metrics
    from the output directory, allowing re-generation of visualizations
    without retraining.
    
    Parameters
    ----------
    include_baselines : bool
        Whether to also load DDPM/NCSN baseline results.
    
    Returns
    -------
    all_results : dict
        Nested dictionary: config_name -> seed -> results
    """
    print("Loading saved results from", spath)
    
    all_results = {}
    
    # Load OT configurations
    for config_name in OT_CONFIGS.keys():
        config_dir = spath / config_name
        if not config_dir.exists():
            print(f"  Warning: {config_name} not found, skipping...")
            continue
        
        print(f"  Loading {config_name}...")
        config_results = {}
        
        for seed in random_seeds:
            seed_dir = config_dir / f"seed_{seed}"
            if not seed_dir.exists():
                print(f"    Warning: seed_{seed} not found, skipping...")
                continue
            
            # Load samples
            samples = None
            samples_original_scale = None
            if (seed_dir / 'samples.pt').exists():
                samples = torch.load(seed_dir / 'samples.pt', weights_only=True)
            if (seed_dir / 'samples_original_scale.pt').exists():
                samples_original_scale = torch.load(seed_dir / 'samples_original_scale.pt', weights_only=True)
            
            # Load training metrics
            training_metrics = None
            if (seed_dir / 'training_metrics.json').exists():
                training_metrics = TrainingMetrics.load(seed_dir / 'training_metrics.json')
            
            # Load quality metrics
            quality_metrics = None
            if (seed_dir / 'quality_metrics.json').exists():
                quality_metrics = GenerationQualityMetrics.load(seed_dir / 'quality_metrics.json')
            
            if samples is not None and quality_metrics is not None:
                config_results[seed] = {
                    'samples': samples,
                    'samples_original_scale': samples_original_scale if samples_original_scale is not None else samples,
                    'training_metrics': training_metrics,
                    'quality_metrics': quality_metrics,
                }
        
        if config_results:
            all_results[config_name] = config_results
    
    # Load baseline configurations (DDPM, NCSN)
    if include_baselines:
        for config_name in BASELINE_CONFIGS.keys():
            config_dir = spath / config_name
            if not config_dir.exists():
                print(f"  Warning: Baseline {config_name} not found, skipping...")
                continue
            
            print(f"  Loading baseline {config_name}...")
            config_results = {}
            
            for seed in random_seeds:
                seed_dir = config_dir / f"seed_{seed}"
                if not seed_dir.exists():
                    continue
                
                samples = None
                samples_original_scale = None
                if (seed_dir / 'samples.pt').exists():
                    samples = torch.load(seed_dir / 'samples.pt', weights_only=True)
                if (seed_dir / 'samples_original_scale.pt').exists():
                    samples_original_scale = torch.load(seed_dir / 'samples_original_scale.pt', weights_only=True)
                
                quality_metrics = None
                if (seed_dir / 'quality_metrics.json').exists():
                    quality_metrics = GenerationQualityMetrics.load(seed_dir / 'quality_metrics.json')
                
                if samples is not None and quality_metrics is not None:
                    config_results[seed] = {
                        'samples': samples,
                        'samples_original_scale': samples_original_scale if samples_original_scale is not None else samples,
                        'training_metrics': None,  # Baselines don't have OT training metrics
                        'quality_metrics': quality_metrics,
                    }
            
            if config_results:
                all_results[config_name] = config_results
    
    print(f"Loaded {len(all_results)} configurations")
    return all_results


def aggregate_results(all_results: Dict) -> Dict[str, Dict]:
    """Aggregate results across seeds for each config."""
    
    aggregated = {}
    
    for config_name, seed_results in all_results.items():
        quality_list = [r['quality_metrics'] for r in seed_results.values()]
        training_list = [r['training_metrics'] for r in seed_results.values() if r['training_metrics']]
        
        # Average quality metrics
        mean_quality = GenerationQualityMetrics(
            config_name=config_name,
            use_ot=quality_list[0].use_ot if quality_list else False,
            ot_kernel=quality_list[0].ot_kernel if quality_list else "",
            ot_method=quality_list[0].ot_method if quality_list else "",
            ot_coupling=quality_list[0].ot_coupling if quality_list else "",
        )
        
        # All metrics to average across seeds (including new spectrum, seasonal, convergence metrics)
        metrics_to_average = [
            # Pointwise statistics
            'mean_mse', 'variance_mse', 'skewness_mse', 'kurtosis_mse', 
            'autocorrelation_mse', 'density_mse', 
            # Training metrics
            'final_train_loss', 'total_train_time', 'mean_path_length', 'mean_grad_variance',
            # Spectrum metrics
            'spectrum_mse', 'spectrum_mse_log', 
            # Seasonal metrics
            'seasonal_mse', 'seasonal_amplitude_error', 'seasonal_phase_correlation',
            # Convergence metrics
            'convergence_rate', 'final_stability',
        ]
        
        for attr in metrics_to_average:
            values = [getattr(q, attr, None) for q in quality_list if getattr(q, attr, None) is not None]
            if values:
                setattr(mean_quality, attr, np.mean(values))
        
        # For epochs_to_90pct, take the median (it's an integer)
        epochs_values = [q.epochs_to_90pct for q in quality_list if q.epochs_to_90pct is not None]
        if epochs_values:
            mean_quality.epochs_to_90pct = int(np.median(epochs_values))
        
        representative_training = training_list[0] if training_list else None
        first_seed = list(seed_results.keys())[0]
        
        aggregated[config_name] = {
            'mean_quality': mean_quality,
            'training_metrics': representative_training,
            'samples': seed_results[first_seed]['samples'],
            'samples_original_scale': seed_results[first_seed]['samples_original_scale'],
        }
    
    return aggregated


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_samples_comparison(
    aggregated: Dict,
    ground_truth: torch.Tensor,
    save_path: Path,
):
    """Create individual comparison plots for each config (ground truth vs model)."""
    n_plot = 30
    
    # Create a subdirectory for sample comparison plots
    samples_dir = save_path / 'sample_comparisons'
    samples_dir.mkdir(exist_ok=True)
    
    # Ground truth in original scale
    gt_original = original_scale(ground_truth.squeeze())
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(aggregated)))
    
    for idx, (config_name, data) in enumerate(aggregated.items()):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        samples = data['samples_original_scale']
        
        # Left: Ground Truth
        ax = axes[0]
        for i in range(min(n_plot, gt_original.shape[0])):
            ax.plot(x_grid.numpy(), gt_original[i].numpy(), color='gray', alpha=0.5, linewidth=0.8)
        ax.set_title(f'Ground Truth\n({gt_original.shape[0]} stations)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Day of Year', fontsize=10)
        ax.set_ylabel('Temperature (°C)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Right: Generated samples
        ax = axes[1]
        for i in range(min(n_plot, samples.shape[0])):
            ax.plot(x_grid.numpy(), samples[i].numpy(), color=colors[idx], alpha=0.4, linewidth=0.8)
        
        title = config_name.replace('_', ' ').title()
        ax.set_title(f'{title}\n({samples.shape[0]} samples)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Day of Year', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Match y-axis limits
        y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
        y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
        axes[0].set_ylim(y_min, y_max)
        axes[1].set_ylim(y_min, y_max)
        
        plt.suptitle(f'AEMET: Ground Truth vs {config_name}', fontsize=12, y=1.02)
        plt.tight_layout()
        
        # Save individual plot
        safe_name = config_name.replace('/', '_').replace('\\', '_')
        plt.savefig(samples_dir / f'{safe_name}.pdf', dpi=150, bbox_inches='tight')
        plt.savefig(samples_dir / f'{safe_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {len(aggregated)} individual sample comparison plots to {samples_dir}")


def plot_seasonal_pattern_comparison(
    aggregated: Dict,
    ground_truth: torch.Tensor,
    save_path: Path,
):
    """Compare seasonal patterns (mean and std) across configs."""
    n_configs = len(aggregated)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Ground truth statistics
    gt_original = original_scale(ground_truth.squeeze())
    gt_mean = gt_original.mean(dim=0).numpy()
    gt_std = gt_original.std(dim=0).numpy()
    days = np.arange(1, 366)
    
    colors = ['gray'] + list(plt.cm.tab10(np.linspace(0, 1, n_configs)))
    
    # Mean plot
    ax = axes[0]
    ax.plot(days, gt_mean, color='gray', linewidth=2, label='Ground Truth')
    ax.fill_between(days, gt_mean - gt_std, gt_mean + gt_std, color='gray', alpha=0.2)
    
    for idx, (config_name, data) in enumerate(aggregated.items()):
        samples = data['samples_original_scale']
        sample_mean = samples.mean(dim=0).numpy()
        ax.plot(days, sample_mean, color=colors[idx + 1], linewidth=1.5, 
                label=config_name, alpha=0.8)
    
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Mean Temperature (°C)')
    ax.set_title('Mean Temperature Profile Comparison')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Std plot
    ax = axes[1]
    ax.plot(days, gt_std, color='gray', linewidth=2, label='Ground Truth')
    
    for idx, (config_name, data) in enumerate(aggregated.items()):
        samples = data['samples_original_scale']
        sample_std = samples.std(dim=0).numpy()
        ax.plot(days, sample_std, color=colors[idx + 1], linewidth=1.5, 
                label=config_name, alpha=0.8)
    
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Temperature Std Dev (°C)')
    ax.set_title('Temperature Variability Comparison')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'seasonal_pattern_comparison.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(save_path / 'seasonal_pattern_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved seasonal pattern comparison to {save_path / 'seasonal_pattern_comparison.pdf'}")


def create_summary_report(aggregated: Dict, save_path: Path):
    """Create comprehensive summary JSON report."""
    summary = {
        'dataset': 'AEMET_weather',
        'n_stations': int(rescaled_data.shape[0]),
        'n_days': int(n_x),
        'epochs': epochs,
        'batch_size_default': batch_size_default,
        'batch_size_signature': batch_size_signature,
        'n_seeds': n_seeds,
        'temperature_range': {
            'min': float(min_data),
            'max': float(max_data),
        },
        'configs': {}
    }
    
    for config_name, data in aggregated.items():
        q = data['mean_quality']
        summary['configs'][config_name] = {
            'use_ot': q.use_ot,
            'ot_kernel': q.ot_kernel,
            'ot_method': q.ot_method,
            'ot_coupling': q.ot_coupling,
            'quality_metrics': {
                'mean_mse': float(q.mean_mse) if q.mean_mse is not None else None,
                'variance_mse': float(q.variance_mse) if q.variance_mse is not None else None,
                'skewness_mse': float(q.skewness_mse) if q.skewness_mse is not None else None,
                'kurtosis_mse': float(q.kurtosis_mse) if q.kurtosis_mse is not None else None,
                'autocorrelation_mse': float(q.autocorrelation_mse) if q.autocorrelation_mse is not None else None,
                'density_mse': float(q.density_mse) if q.density_mse is not None else None,
            },
            'training_metrics': {
                'final_train_loss': float(q.final_train_loss) if q.final_train_loss is not None else None,
                'total_train_time': float(q.total_train_time) if q.total_train_time is not None else None,
                'mean_path_length': float(q.mean_path_length) if q.mean_path_length is not None else None,
                'mean_grad_variance': float(q.mean_grad_variance) if q.mean_grad_variance is not None else None,
            }
        }
    
    with open(save_path / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved experiment summary to {save_path / 'experiment_summary.json'}")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Comprehensive OT-FFM Experiments on AEMET Weather Data")
    print(f"OT Configs: {len(OT_CONFIGS)}, Baselines: {len(BASELINE_CONFIGS)}, Seeds per config: {n_seeds}")
    print(f"Output directory: {spath}")
    if args.load_only:
        print("MODE: Load-only (loading saved results)")
    elif args.baselines_only:
        print("MODE: Baselines-only (running DDPM and NCSN)")
    else:
        print("MODE: Training (running OT experiments)")
    print("="*60)
    
    if args.load_only:
        # Load saved results
        all_results = load_all_results(include_baselines=True)
        if not all_results:
            print("ERROR: No saved results found. Run without --load-only first.")
            sys.exit(1)
    elif args.baselines_only:
        # Save ground truth if not exists
        if not (spath / 'ground_truth_rescaled.pt').exists():
            torch.save(rescaled_data, spath / 'ground_truth_rescaled.pt')
            torch.save(original_scale(rescaled_data.squeeze()), spath / 'ground_truth_original.pt')
        
        # Run only baseline experiments
        all_results = run_baseline_experiments()
        print("\nBaseline experiments complete!")
        print("Run with --load-only to generate visualizations with all results.")
        sys.exit(0)
    else:
        # Save ground truth
        torch.save(rescaled_data, spath / 'ground_truth_rescaled.pt')
        torch.save(original_scale(rescaled_data.squeeze()), spath / 'ground_truth_original.pt')
        
        # Run all OT experiments
        all_results = run_all_experiments()
    
    # Aggregate results
    print("\n" + "="*60)
    print("Aggregating results...")
    print("="*60)
    aggregated = aggregate_results(all_results)
    
    # Separate OT methods from baselines for different visualizations
    ot_aggregated = {k: v for k, v in aggregated.items() if k not in BASELINE_CONFIGS}
    baseline_aggregated = {k: v for k, v in aggregated.items() if k in BASELINE_CONFIGS}
    
    print(f"  OT methods: {len(ot_aggregated)}")
    print(f"  Baselines: {len(baseline_aggregated)}")
    
    # Generate visualizations
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)
    
    # 1. Sample comparison plot (ALL methods including baselines)
    plot_samples_comparison(aggregated, rescaled_data, spath)
    
    # 2. Seasonal pattern comparison (visual)
    plot_seasonal_pattern_comparison(aggregated, rescaled_data, spath)
    
    # 3. Seasonal pattern comparison with numerical metrics
    gt_original = original_scale(rescaled_data.squeeze())
    generated_list = [data['samples_original_scale'] for data in aggregated.values()]
    config_names = list(aggregated.keys())
    
    fig, seasonal_metrics = compare_seasonal_patterns(
        gt_original, 
        generated_list, 
        config_names,
        save_path=spath / 'seasonal_metrics_comparison.pdf'
    )
    plt.close(fig)
    
    # Save seasonal metrics to JSON
    with open(spath / 'seasonal_metrics.json', 'w') as f:
        json.dump(seasonal_metrics, f, indent=2)
    print(f"Saved seasonal metrics to {spath / 'seasonal_metrics.json'}")
    
    # 4. Spectrum comparison (OT methods only - for ablation)
    if ot_aggregated:
        print("\nGenerating spectrum comparison (OT methods only)...")
        ot_generated_list = [data['samples_original_scale'] for data in ot_aggregated.values()]
        ot_config_names = list(ot_aggregated.keys())
        quality_list = [data['mean_quality'] for data in ot_aggregated.values()]
        # Top-5 spectrum comparison
        fig = compare_spectra_1d(
            gt_original,
            ot_generated_list,
            config_names=ot_config_names,
            save_path=spath / 'spectrum_comparison_top5.pdf',
            top_k=5,
        )
        plt.close(fig)
        # Best per category spectrum comparison
        fig = compare_spectra_1d(
            gt_original,
            ot_generated_list,
            config_names=ot_config_names,
            save_path=spath / 'spectrum_comparison.pdf',
            quality_metrics_list=quality_list,
            save_best_per_category=True,
        )
        plt.close(fig)
    
    # 5. Training metrics comparison (OT methods only - for ablation)
    training_metrics_list = [
        data['training_metrics'] for name, data in ot_aggregated.items() 
        if data['training_metrics'] is not None
    ]
    quality_metrics_list_ot = [
        data['mean_quality'] for name, data in ot_aggregated.items()
        if data['training_metrics'] is not None
    ]
    if training_metrics_list:
        # Full comparison (all configs)
        figs = compare_training_runs(training_metrics_list, save_path=spath / 'training')
        for fig in figs:
            plt.close(fig)
        
        # Simplified comparison (best of each kernel type only)
        print("\nGenerating simplified training comparison (best per kernel type)...")
        figs_simple = compare_training_runs_simplified(
            training_metrics_list,
            quality_metrics_list=quality_metrics_list_ot,
            metric_key='mean_mse',
            save_path=spath / 'training_simplified'
        )
        for fig in figs_simple:
            plt.close(fig)
        
        plot_convergence_comparison(training_metrics_list, save_path=spath / 'convergence_comparison.pdf')
        print_comparison_table(training_metrics_list)
    
    # 6. Convergence analysis with numerical metrics
    losses_dict = {
        name: data['training_metrics'].train_losses 
        for name, data in aggregated.items() 
        if data['training_metrics'] is not None
    }
    if losses_dict:
        figs, convergence_metrics = compare_convergence(
            losses_dict,
            save_path=spath / 'convergence_metrics'
        )
        for fig in figs:
            plt.close(fig)
        
        # Simplified convergence (best per kernel type)
        quality_dict = {name: data['mean_quality'].mean_mse for name, data in aggregated.items()
                       if data['mean_quality'] is not None and data['mean_quality'].mean_mse is not None}
        fig_simple = compare_convergence_simplified(
            losses_dict,
            quality_metrics_dict=quality_dict,
            save_path=spath / 'convergence_metrics'
        )
        if fig_simple:
            plt.close(fig_simple)
        
        # Save convergence metrics to JSON
        with open(spath / 'convergence_metrics.json', 'w') as f:
            # Convert numpy types to Python types for JSON
            conv_json = {
                k: {kk: float(vv) if vv is not None else None for kk, vv in v.items()}
                for k, v in convergence_metrics.items()
            }
            json.dump(conv_json, f, indent=2)
        print(f"Saved convergence metrics to {spath / 'convergence_metrics.json'}")
        print_convergence_table(convergence_metrics)
    
    # 7. Generation quality comparison (including spectrum and seasonal metrics)
    quality_metrics_list = [data['mean_quality'] for data in aggregated.values()]
    # Top-5 ranked comparison
    compare_generation_quality(
        quality_metrics_list, 
        save_path=spath / 'quality_comparison_top5.pdf',
        include_spectrum=True,
        include_seasonal=True,
        top_k=5,
    )
    # All configs for reference
    compare_generation_quality(
        quality_metrics_list, 
        save_path=spath / 'quality_comparison_all.pdf',
        include_spectrum=True,
        include_seasonal=True,
        show_all=True,
    )
    # Simplified: Best per kernel category (5 entries)
    print("\nGenerating simplified quality comparison (best per kernel type)...")
    compare_generation_quality_simplified(
        quality_metrics_list,
        save_path=spath / 'quality_comparison.pdf',
    )
    
    # Print all metrics tables
    print_comprehensive_table(quality_metrics_list)
    
    # 8. Save comprehensive metrics summary
    save_all_metrics_summary(quality_metrics_list, spath / 'comprehensive_metrics.json')
    
    # 9. Create summary report
    create_summary_report(aggregated, spath)
    
    # Cleanup
    plt.close('all')
    
    print("\n" + "="*60)
    print(f"All experiments complete! Results saved to: {spath}")
    print("="*60)
    
    # Print directory structure
    print("\nDirectory structure:")
    for config_name in OT_CONFIGS.keys():
        config_dir = spath / config_name
        if config_dir.exists():
            print(f"  {config_name}/")
            for seed_dir in sorted(config_dir.iterdir()):
                if seed_dir.is_dir():
                    n_files = len(list(seed_dir.iterdir()))
                    print(f"    {seed_dir.name}/ ({n_files} files)")

