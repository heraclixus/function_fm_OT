"""
Comprehensive OT-FFM Experiments on Economics Time Series.

This script runs experiments with multiple OT configurations on three
economics datasets (Population, GDP, Labor) comparing:
- Different kernel types (euclidean, rbf, signature)
- Different OT methods (exact, sinkhorn)
- Different coupling strategies (sample, barycentric)

Each configuration gets its own subdirectory with:
- Trained model checkpoint
- Generated samples (original and upsampled resolution)
- Training metrics (loss, path length, gradient variance)
- Generation quality statistics (Mean, Variance, Skewness, Kurtosis, Autocorrelation MSE)

Usage:
    python econ_ot.py                    # Run all experiments
    python econ_ot.py --load-only        # Load saved results and regenerate plots
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

# =============================================================================
# Argument Parsing
# =============================================================================

parser = argparse.ArgumentParser(description='Economics OT-FFM Experiments')
parser.add_argument('--load-only', action='store_true',
                    help='Load saved results instead of training')
parser.add_argument('--baselines-only', action='store_true',
                    help='Run only DDPM and NCSN baselines (not OT experiments)')
parser.add_argument('--spath', type=str, default='../outputs/econ_ot_comprehensive/',
                    help='Output/load directory')
args, _ = parser.parse_known_args()
from typing import Dict, List, Any, Tuple

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
    print_comprehensive_table,
    compute_all_pointwise_statistics,
    compare_spectra_1d,
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

econ1 = torch.load('../data/economy/econ1.pt').float()
econ2 = torch.load('../data/economy/econ2.pt').float()
econ2 = econ2[~torch.any(econ2.isnan(), dim=1)]
econ3 = torch.load('../data/economy/econ3.pt').float()

# Normalize by mean
econ1 = econ1 / torch.mean(econ1, dim=1).unsqueeze(-1)
econ3 = econ3 / torch.mean(econ3, dim=1).unsqueeze(-1)

def maxmin_rescale(data):
    dmax = torch.max(data)
    dmin = torch.min(data)
    scaled_data = -1 + 2 * (data - dmin) / (dmax - dmin)
    return scaled_data

econ1scaled = maxmin_rescale(econ1)
econ2scaled = maxmin_rescale(econ2)
econ3scaled = maxmin_rescale(econ3)

# Repeat data for training (small dataset augmentation)
n_repeat = 10
econ1scaled_repeat = econ1scaled.repeat(n_repeat, 1)
econ2scaled_repeat = econ2scaled.repeat(n_repeat, 1)
econ3scaled_repeat = econ3scaled.repeat(n_repeat, 1)

# =============================================================================
# Configuration
# =============================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

batch_size = 512
num_workers = 0
pin_memory = True

# Create data loaders
train_loader1 = DataLoader(
    econ1scaled_repeat.unsqueeze(1),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
train_loader2 = DataLoader(
    econ2scaled_repeat.unsqueeze(1),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
train_loader3 = DataLoader(
    econ3scaled_repeat.unsqueeze(1),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

# Dataset info
DATASETS = {
    "econ1_population": {
        "loader": train_loader1,
        "ground_truth": econ1scaled,
        "n_x": econ1.shape[1],
        "modes": 32,
        "width": 256,
    },
    "econ2_gdp": {
        "loader": train_loader2,
        "ground_truth": econ2scaled,
        "n_x": econ2.shape[1],
        "modes": 32,
        "width": 256,
    },
    "econ3_labor": {
        "loader": train_loader3,
        "ground_truth": econ3scaled,
        "n_x": econ3.shape[1],
        "modes": 8,  # Shorter series
        "width": 128,
    },
}

print(f"Dataset sizes: econ1={econ1.shape[1]}, econ2={econ2.shape[1]}, econ3={econ3.shape[1]}")
print(f"Training samples after repeat: {econ1scaled_repeat.shape[0]}")

# Model params
mlp_width = 128

# GP hyperparameters
kernel_length = 0.01
kernel_variance = 0.1

# Training params
epochs = 100
lr = 1e-3

# FFM params
sigma_min = 1e-4
upsample = 5

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
    # =========================================================================
    "gaussian_ot": {
        "use_ot": True,
        "ot_method": "gaussian",
        "ot_reg": 1e-3,  # Regularization for covariance stability
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
    # Note: max_seq_len=64, max_batch=32 to avoid GPU memory issues on long series
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
            "dyadic_order": 2,
            "static_kernel_type": "rbf",
            "static_kernel_sigma": 1.0,
            "add_basepoint": True,
            "normalize": True,
            "max_seq_len": 64,
            "max_batch": 32,
        },
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
            "dyadic_order": 2,
            "static_kernel_type": "rbf",
            "static_kernel_sigma": 1.0,
            "add_basepoint": True,
            "normalize": True,
            "max_seq_len": 64,
            "max_batch": 32,
        },
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
            "dyadic_order": 2,
            "static_kernel_type": "rbf",
            "static_kernel_sigma": 1.0,
            "add_basepoint": True,
            "normalize": True,
            "max_seq_len": 64,
            "max_batch": 32,
        },
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
            "dyadic_order": 2,
            "static_kernel_type": "rbf",
            "static_kernel_sigma": 1.0,
            "add_basepoint": True,
            "normalize": True,
            "max_seq_len": 64,
            "max_batch": 32,
        },
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

def create_model(modes, width, device):
    """Create FNO model with given hyperparameters."""
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
    dataset_name: str,
    dataset_info: dict,
    save_dir: Path,
) -> Dict[str, Any]:
    """Train a single configuration on a single dataset with a single seed."""
    
    print(f"\n  Training {config_name} on {dataset_name} (seed={seed})...")
    
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create model
    model = create_model(dataset_info["modes"], dataset_info["width"], device)
    
    # Create FFM wrapper
    ffm_kwargs = build_ffm_kwargs(config)
    ffm = FFMModelOT(model, **ffm_kwargs)
    
    # Create monitor
    monitor = TrainingMonitor(
        method_name=f"{dataset_name}_{config_name}",
        use_ot=config.get("use_ot", False),
        ot_kernel=config.get("ot_kernel", "") or "",
        track_batch_metrics=False,
    )
    
    # Train
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50)
    
    training_metrics = ffm.train(
        train_loader=dataset_info["loader"],
        optimizer=optimizer,
        epochs=epochs,
        scheduler=scheduler,
        eval_int=0,
        save_int=epochs,
        generate=False,
        save_path=save_dir,
        monitor=monitor,
    )
    
    n_x = dataset_info["n_x"]
    ground_truth = dataset_info["ground_truth"]
    
    # Generate samples at original and upsampled resolution
    print(f"    Generating {n_gen_samples} samples...")
    samples_original = ffm.sample([n_x], n_samples=n_gen_samples).cpu().squeeze()
    samples_upsampled = ffm.sample([n_x * upsample], n_samples=n_gen_samples).cpu().squeeze()
    
    # Compute generation quality statistics
    print(f"    Computing quality statistics...")
    quality_metrics = GenerationQualityMetrics(
        config_name=f"{dataset_name}_{config_name}",
        ot_kernel=config.get("ot_kernel", "") or "",
        ot_method=config.get("ot_method", "") or "",
        ot_coupling=config.get("ot_coupling", "") or "",
        use_ot=config.get("use_ot", False),
    )
    quality_metrics.compute_from_samples(ground_truth, samples_original)
    
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
    torch.save(samples_original, save_dir / 'samples_original.pt')
    torch.save(samples_upsampled, save_dir / 'samples_upsampled.pt')
    torch.save(model.state_dict(), save_dir / 'model.pt')
    if training_metrics is not None:
        training_metrics.save(save_dir / 'training_metrics.json')
    quality_metrics.save(save_dir / 'quality_metrics.json')
    
    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    return {
        'samples_original': samples_original,
        'samples_upsampled': samples_upsampled,
        'training_metrics': training_metrics,
        'quality_metrics': quality_metrics,
    }


def train_baseline_config(
    config_name: str,
    config: dict,
    seed: int,
    dataset_name: str,
    dataset_info: dict,
    save_dir: Path,
) -> Dict[str, Any]:
    """Train a baseline (DDPM/NCSN) configuration on a single dataset."""
    import time
    
    print(f"\n  Training {config_name} on {dataset_name} (seed={seed})...")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    n_x = dataset_info["n_x"]
    model = create_model(dataset_info["modes"], dataset_info["width"], device)
    
    method = config["method"]
    if method == "DDPM":
        diffusion = DiffusionModel(
            model, method=method, T=config["T"], device=device,
            kernel_length=kernel_length, kernel_variance=kernel_variance,
            beta_min=config["beta_min"], beta_max=config["beta_max"],
            dtype=torch.float32,
        )
    else:
        diffusion = DiffusionModel(
            model, method=method, T=config["T"], device=device,
            kernel_length=kernel_length, kernel_variance=kernel_variance,
            sigma1=config["sigma1"], sigmaT=config["sigmaT"],
            precondition=config.get("precondition", True),
            dtype=torch.float32,
        )
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50)
    
    t0 = time.time()
    diffusion.train(train_loader=dataset_info["loader"], optimizer=optimizer, epochs=epochs,
                    scheduler=scheduler, eval_int=0, save_int=epochs,
                    generate=False, save_path=save_dir)
    train_time = time.time() - t0
    
    print(f"    Generating {n_gen_samples} samples...")
    samples_original = diffusion.sample([n_x], n_samples=n_gen_samples, n_channels=1).cpu().squeeze()
    samples_upsampled = diffusion.sample([n_x * upsample], n_samples=n_gen_samples, n_channels=1).cpu().squeeze()
    
    quality_metrics = GenerationQualityMetrics(
        config_name=config_name, ot_kernel="", ot_method="", ot_coupling="", use_ot=False,
    )
    quality_metrics.compute_from_samples(dataset_info["ground_truth"], samples_original)
    quality_metrics.total_train_time = train_time
    
    torch.save(samples_original, save_dir / 'samples_original.pt')
    torch.save(samples_upsampled, save_dir / 'samples_upsampled.pt')
    torch.save(model.state_dict(), save_dir / 'model.pt')
    quality_metrics.save(save_dir / 'quality_metrics.json')
    
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    return {
        'samples_original': samples_original,
        'samples_upsampled': samples_upsampled,
        'training_metrics': None,
        'quality_metrics': quality_metrics,
    }


def run_baseline_dataset_experiments(
    dataset_name: str,
    dataset_info: dict,
    output_dir: Path,
) -> Dict[str, Dict[int, Dict]]:
    """Run all baseline configurations for a single dataset."""
    
    print(f"\n{'='*70}")
    print(f"Baselines for Dataset: {dataset_name}")
    print(f"{'='*70}")
    
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for config_name, config in BASELINE_CONFIGS.items():
        print(f"\n--- Baseline: {config_name} ---")
        
        config_results = {}
        for seed in random_seeds:
            config_dir = dataset_dir / config_name / f"seed_{seed}"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            result = train_baseline_config(
                config_name=config_name,
                config=config,
                seed=seed,
                dataset_name=dataset_name,
                dataset_info=dataset_info,
                save_dir=config_dir,
            )
            config_results[seed] = result
        
        all_results[config_name] = config_results
    
    return all_results


def run_dataset_experiments(
    dataset_name: str,
    dataset_info: dict,
    output_dir: Path,
) -> Dict[str, Dict[int, Dict]]:
    """Run all configurations for a single dataset."""
    
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name} (n_x={dataset_info['n_x']})")
    print(f"{'='*70}")
    
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Save ground truth
    torch.save(dataset_info["ground_truth"], dataset_dir / 'ground_truth.pt')
    
    all_results = {}
    
    for config_name, config in OT_CONFIGS.items():
        print(f"\n--- Configuration: {config_name} ---")
        
        config_results = {}
        
        for seed in random_seeds:
            # Create subdirectory
            config_dir = dataset_dir / config_name / f"seed_{seed}"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            result = train_single_config(
                config_name=config_name,
                config=config,
                seed=seed,
                dataset_name=dataset_name,
                dataset_info=dataset_info,
                save_dir=config_dir,
            )
            
            config_results[seed] = result
        
        all_results[config_name] = config_results
    
    return all_results


def load_dataset_results(dataset_name: str, output_dir: Path, include_baselines: bool = True) -> Dict[str, Dict[int, Dict]]:
    """Load results for a single dataset from saved files."""
    dataset_dir = output_dir / dataset_name
    if not dataset_dir.exists():
        return {}
    
    print(f"  Loading {dataset_name}...")
    all_results = {}
    
    # Load OT configs
    all_configs = dict(OT_CONFIGS)
    if include_baselines:
        all_configs.update(BASELINE_CONFIGS)
    
    for config_name in all_configs.keys():
        config_dir = dataset_dir / config_name
        if not config_dir.exists():
            continue
        
        config_results = {}
        for seed in random_seeds:
            seed_dir = config_dir / f"seed_{seed}"
            if not seed_dir.exists():
                continue
            
            samples_original = None
            samples_upsampled = None
            if (seed_dir / 'samples_original.pt').exists():
                samples_original = torch.load(seed_dir / 'samples_original.pt', weights_only=True)
            if (seed_dir / 'samples_upsampled.pt').exists():
                samples_upsampled = torch.load(seed_dir / 'samples_upsampled.pt', weights_only=True)
            
            training_metrics = None
            if (seed_dir / 'training_metrics.json').exists():
                from util.ot_monitoring import TrainingMetrics
                training_metrics = TrainingMetrics.load(seed_dir / 'training_metrics.json')
            
            quality_metrics = None
            if (seed_dir / 'quality_metrics.json').exists():
                quality_metrics = GenerationQualityMetrics.load(seed_dir / 'quality_metrics.json')
            
            if samples_original is not None and quality_metrics is not None:
                config_results[seed] = {
                    'samples_original': samples_original,
                    'samples_upsampled': samples_upsampled,
                    'training_metrics': training_metrics,
                    'quality_metrics': quality_metrics,
                }
        
        if config_results:
            all_results[config_name] = config_results
    
    return all_results


def aggregate_dataset_results(results: Dict) -> Dict[str, Dict]:
    """Aggregate results across seeds for a single dataset."""
    aggregated = {}
    
    for config_name, seed_results in results.items():
        quality_list = [r['quality_metrics'] for r in seed_results.values()]
        training_list = [r['training_metrics'] for r in seed_results.values() if r['training_metrics']]
        
        # Average quality metrics
        mean_quality = GenerationQualityMetrics(
            config_name=quality_list[0].config_name if quality_list else config_name,
            use_ot=quality_list[0].use_ot if quality_list else False,
            ot_kernel=quality_list[0].ot_kernel if quality_list else "",
            ot_method=quality_list[0].ot_method if quality_list else "",
            ot_coupling=quality_list[0].ot_coupling if quality_list else "",
        )
        
        # All metrics to average across seeds (including spectrum and convergence)
        metrics_to_average = [
            # Pointwise statistics
            'mean_mse', 'variance_mse', 'skewness_mse', 'kurtosis_mse', 
            'autocorrelation_mse', 'density_mse', 
            # Training metrics
            'final_train_loss', 'total_train_time', 'mean_path_length', 'mean_grad_variance',
            # Spectrum metrics
            'spectrum_mse', 'spectrum_mse_log', 
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
            'samples_original': seed_results[first_seed]['samples_original'],
            'samples_upsampled': seed_results[first_seed]['samples_upsampled'],
        }
    
    return aggregated


def plot_dataset_comparison(
    aggregated: Dict,
    ground_truth: torch.Tensor,
    dataset_name: str,
    n_x: int,
    save_dir: Path,
):
    """Create individual comparison plots for each configuration (ground truth vs model)."""
    x_grid = torch.linspace(0, 1, n_x)
    n_plot = 30
    
    # Create a subdirectory for sample comparison plots
    samples_dir = save_dir / 'sample_comparisons'
    samples_dir.mkdir(exist_ok=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(aggregated)))
    
    for idx, (config_name, data) in enumerate(aggregated.items()):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        samples = data['samples_original']
        
        # Left: Ground Truth
        ax = axes[0]
        for i in range(min(n_plot, ground_truth.shape[0])):
            ax.plot(x_grid.numpy(), ground_truth[i].numpy(), color='gray', alpha=0.4, linewidth=0.8)
        ax.set_title(f'Ground Truth\n({ground_truth.shape[0]} samples)', fontsize=11, fontweight='bold')
        ax.set_xlabel('t', fontsize=10)
        ax.set_ylabel('value', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Right: Generated samples
        ax = axes[1]
        for i in range(min(n_plot, samples.shape[0])):
            ax.plot(x_grid.numpy(), samples[i].numpy(), color=colors[idx], alpha=0.4, linewidth=0.8)
        
        # Clean up config name for title
        title = config_name.replace('_', ' ').title()
        ax.set_title(f'{title}\n({samples.shape[0]} samples)', fontsize=11, fontweight='bold')
        ax.set_xlabel('t', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Match y-axis limits
        y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
        y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
        axes[0].set_ylim(y_min, y_max)
        axes[1].set_ylim(y_min, y_max)
        
        plt.suptitle(f'{dataset_name}: Ground Truth vs {config_name}', fontsize=12, y=1.02)
        plt.tight_layout()
        
        # Save individual plot
        safe_name = config_name.replace('/', '_').replace('\\', '_')
        plt.savefig(samples_dir / f'{dataset_name}_{safe_name}.pdf', dpi=150, bbox_inches='tight')
        plt.savefig(samples_dir / f'{dataset_name}_{safe_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  Saved {len(aggregated)} individual sample comparison plots to {samples_dir}")


def to_python_float(val):
    """Convert numpy types to native Python float for JSON serialization."""
    if val is None:
        return None
    if isinstance(val, (np.floating, np.integer)):
        return val.item()
    if isinstance(val, np.ndarray):
        return val.item() if val.ndim == 0 else float(val.flat[0])
    return float(val)


def create_full_summary(all_aggregated: Dict[str, Dict], save_path: Path):
    """Create comprehensive summary across all datasets."""
    summary = {}
    
    for dataset_name, aggregated in all_aggregated.items():
        summary[dataset_name] = {}
        for config_name, data in aggregated.items():
            q = data['mean_quality']
            summary[dataset_name][config_name] = {
                'quality_metrics': {
                    'mean_mse': to_python_float(q.mean_mse),
                    'variance_mse': to_python_float(q.variance_mse),
                    'skewness_mse': to_python_float(q.skewness_mse),
                    'kurtosis_mse': to_python_float(q.kurtosis_mse),
                    'autocorrelation_mse': to_python_float(q.autocorrelation_mse),
                },
                'training_metrics': {
                    'final_train_loss': to_python_float(q.final_train_loss),
                    'mean_path_length': to_python_float(q.mean_path_length),
                    'mean_grad_variance': to_python_float(q.mean_grad_variance),
                }
            }
    
    with open(save_path / 'full_experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved full summary to {save_path / 'full_experiment_summary.json'}")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Comprehensive OT-FFM Experiments on Economics Time Series")
    print(f"Datasets: {len(DATASETS)}, Configs: {len(OT_CONFIGS)}, Seeds: {n_seeds}")
    print(f"Output directory: {spath}")
    if args.baselines_only:
        print("MODE: Baselines-only (running DDPM and NCSN only)")
    elif args.load_only:
        print("MODE: Load-only (loading saved results)")
    else:
        print("MODE: Training (running experiments)")
    print("="*70)
    
    all_results = {}
    all_aggregated = {}
    
    # Run or load experiments for each dataset
    for dataset_name, dataset_info in DATASETS.items():
        if args.baselines_only:
            # Run only baselines
            results = run_baseline_dataset_experiments(dataset_name, dataset_info, spath)
            # Also load existing OT results for visualization
            ot_results = load_dataset_results(dataset_name, spath, include_baselines=False)
            results.update(ot_results)
        elif args.load_only:
            results = load_dataset_results(dataset_name, spath, include_baselines=True)
            if not results:
                print(f"  Warning: No saved results for {dataset_name}, skipping...")
                continue
        else:
            results = run_dataset_experiments(dataset_name, dataset_info, spath)
        all_results[dataset_name] = results
        
        # Aggregate
        aggregated = aggregate_dataset_results(results)
        all_aggregated[dataset_name] = aggregated
        
        # Generate per-dataset visualizations
        dataset_dir = spath / dataset_name
        
        plot_dataset_comparison(
            aggregated,
            dataset_info["ground_truth"],
            dataset_name,
            dataset_info["n_x"],
            dataset_dir,
        )
        
        # Filter out baselines for OT-specific training plots
        baseline_names = set(BASELINE_CONFIGS.keys())
        ot_aggregated = {k: v for k, v in aggregated.items() if k not in baseline_names}
        
        # Training comparison (OT configs only)
        training_list = [d['training_metrics'] for d in ot_aggregated.values() if d['training_metrics']]
        quality_list = [d['mean_quality'] for d in ot_aggregated.values() if d['training_metrics']]
        if training_list:
            figs = compare_training_runs(training_list, save_path=dataset_dir / 'training')
            for fig in figs:
                plt.close(fig)
            
            # Simplified comparison
            figs_simple = compare_training_runs_simplified(
                training_list, quality_metrics_list=quality_list, metric_key='mean_mse',
                save_path=dataset_dir / 'training_simplified'
            )
            for fig in figs_simple:
                plt.close(fig)
            
            plot_convergence_comparison(training_list, save_path=dataset_dir / 'convergence.pdf')
        
        # Convergence analysis with numerical metrics (OT configs only)
        losses_dict = {
            name: d['training_metrics'].train_losses 
            for name, d in ot_aggregated.items() 
            if d['training_metrics'] is not None
        }
        if losses_dict:
            figs, conv_metrics = compare_convergence(
                losses_dict,
                save_path=dataset_dir / 'convergence_metrics'
            )
            for fig in figs:
                plt.close(fig)
            
            # Save convergence metrics
            with open(dataset_dir / 'convergence_metrics.json', 'w') as f:
                conv_json = {
                    k: {kk: float(vv) if vv is not None else None for kk, vv in v.items()}
                    for k, v in conv_metrics.items()
                }
                json.dump(conv_json, f, indent=2)
            print_convergence_table(conv_metrics)
        
        # Spectrum comparison (OT configs only)
        ground_truth = dataset_info["ground_truth"]
        generated_list = [d['samples_original'] for d in ot_aggregated.values()]
        config_names = list(ot_aggregated.keys())
        quality_list_for_spectrum = [d['mean_quality'] for d in ot_aggregated.values()]
        
        # Top-5 spectrum comparison
        fig = compare_spectra_1d(
            ground_truth,
            generated_list,
            config_names=config_names,
            save_path=dataset_dir / 'spectrum_comparison_top5.pdf',
            top_k=5,
        )
        plt.close(fig)
        # Best per category spectrum comparison
        fig = compare_spectra_1d(
            ground_truth,
            generated_list,
            config_names=config_names,
            save_path=dataset_dir / 'spectrum_comparison.pdf',
            quality_metrics_list=quality_list_for_spectrum,
            save_best_per_category=True,
        )
        plt.close(fig)
        
        # Quality comparison (including spectrum metrics)
        quality_list = [d['mean_quality'] for d in aggregated.values()]
        # Top-5 ranked comparison
        compare_generation_quality(
            quality_list, 
            save_path=dataset_dir / 'quality_comparison_top5.pdf',
            include_spectrum=True,
            include_seasonal=False,  # Econ data doesn't have clear seasonal patterns
            top_k=5,
        )
        # All configs for reference
        compare_generation_quality(
            quality_list, 
            save_path=dataset_dir / 'quality_comparison_all.pdf',
            include_spectrum=True,
            include_seasonal=False,
            show_all=True,
        )
        # Simplified: Best per kernel category
        compare_generation_quality_simplified(
            quality_list,
            save_path=dataset_dir / 'quality_comparison.pdf',
        )
        
        # Save comprehensive metrics
        save_all_metrics_summary(quality_list, dataset_dir / 'comprehensive_metrics.json')
        
        print(f"\n--- Quality Results for {dataset_name} ---")
        print_comprehensive_table(quality_list)
    
    # Create overall summary
    print("\n" + "="*70)
    print("Creating comprehensive summary...")
    print("="*70)
    
    create_full_summary(all_aggregated, spath)
    
    # Cross-dataset comparison (all quality metrics in one view)
    all_quality = []
    for dataset_name, aggregated in all_aggregated.items():
        for config_name, data in aggregated.items():
            q = data['mean_quality']
            q.config_name = f"{dataset_name[:6]}_{config_name}"
            all_quality.append(q)
    
    compare_generation_quality(all_quality, save_path=spath / 'all_datasets_quality.pdf', top_k=5)
    compare_generation_quality(all_quality, save_path=spath / 'all_datasets_quality_all.pdf', show_all=True)
    
    plt.close('all')
    
    print("\n" + "="*70)
    print(f"All experiments complete! Results saved to: {spath}")
    print("="*70)
    
    # Print directory structure
    print("\nDirectory structure:")
    for dataset_name in DATASETS.keys():
        print(f"  {dataset_name}/")
        dataset_dir = spath / dataset_name
        if dataset_dir.exists():
            for config_name in OT_CONFIGS.keys():
                config_dir = dataset_dir / config_name
                if config_dir.exists():
                    print(f"    {config_name}/")
                    for seed_dir in sorted(config_dir.iterdir()):
                        if seed_dir.is_dir():
                            n_files = len(list(seed_dir.iterdir()))
                            print(f"      {seed_dir.name}/ ({n_files} files)")
