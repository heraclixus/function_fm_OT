"""
Comprehensive OT-FFM Experiments on Mixture of GPs.

This script runs experiments with multiple OT configurations:
- Different kernel types (euclidean, rbf, signature)
- Different OT methods (exact, sinkhorn)
- Different coupling strategies (sample, barycentric)

Each configuration gets its own subdirectory with:
- Trained model checkpoint
- Generated samples
- Training metrics (loss, path length, gradient variance)
- Generation quality statistics (Mean, Variance, Skewness, Kurtosis, Autocorrelation MSE)

At the end, comprehensive comparison visualizations are generated.

Usage:
    python moGP_ot.py                    # Run all experiments
    python moGP_ot.py --load-only        # Load saved results and regenerate plots
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
from typing import Dict, List, Any

# =============================================================================
# Argument Parsing
# =============================================================================

parser = argparse.ArgumentParser(description='Mixture of GPs OT-FFM Experiments')
parser.add_argument('--load-only', action='store_true',
                    help='Load saved results instead of training')
parser.add_argument('--baselines-only', action='store_true',
                    help='Run only DDPM and NCSN baselines (not OT experiments)')
parser.add_argument('--spath', type=str, default='../outputs/moGP_ot_comprehensive/',
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

from data.moGP import load_mixture_gps

# =============================================================================
# Configuration
# =============================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Data params
train_samples = 5000
n_x = 64
batch_size = 512
batch_size_signature = 128  # Smaller batch for signature kernel (memory intensive)

# Model params
modes = 32
width = 256
mlp_width = 128

# GP prior params (for base distribution)
kernel_length = 0.01
kernel_variance = 0.1

# Training params
epochs = 50
lr = 1e-3

# FFM params
sigma_min = 1e-4

# Number of random seeds for each config
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
        "ot_method": None,
        "ot_kernel": None,
        "ot_coupling": None,
        "ot_kernel_params": None,
    },
    
    # =========================================================================
    # Gaussian OT (Bures-Wasserstein) - closed-form, fast, stable
    # =========================================================================
    "gaussian_ot": {
        "use_ot": True,
        "ot_method": "gaussian",
        "ot_reg": 1e-3,  # Regularization for covariance stability
        "ot_kernel": None,
        "ot_coupling": "barycentric",  # Deterministic map
        "ot_kernel_params": None,
    },
    
    # =========================================================================
    # Euclidean OT - All Methods
    # =========================================================================
    # Exact solver (no regularization)
    "euclidean_exact": {
        "use_ot": True,
        "ot_method": "exact",
        "ot_kernel": "euclidean",
        "ot_coupling": "sample",
        "ot_kernel_params": None,
    },
    
    # Sinkhorn with different regularization values
    "euclidean_sinkhorn_reg0.1": {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.1,
        "ot_kernel": "euclidean",
        "ot_coupling": "sample",
        "ot_kernel_params": None,
    },
    "euclidean_sinkhorn_reg0.5": {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.5,
        "ot_kernel": "euclidean",
        "ot_coupling": "sample",
        "ot_kernel_params": None,
    },
    "euclidean_sinkhorn_reg1.0": {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 1.0,
        "ot_kernel": "euclidean",
        "ot_coupling": "sample",
        "ot_kernel_params": None,
    },
    
    # Note: unbalanced and partial OT are not included because they
    # don't preserve marginal constraints required for flow matching.
    
    # =========================================================================
    # RBF Kernel OT - All Methods
    # =========================================================================
    # Exact solver
    "rbf_exact": {
        "use_ot": True,
        "ot_method": "exact",
        "ot_kernel": "rbf",
        "ot_coupling": "sample",
        "ot_kernel_params": {"sigma": 5.0},
    },
    
    # Sinkhorn with different regularization values
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
    
    # Barycentric coupling (deterministic)
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
    # =========================================================================
    # Sinkhorn with different regularization values
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
    },
    
    # Barycentric coupling
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
# Load Data
# =============================================================================

print("Loading mixture of GPs dataset...")
base_seed = 42
torch.manual_seed(base_seed)

# Load or generate data (saved in data/ folder for reuse)
data_dir = Path(__file__).parent.parent / 'data'
x_tr, y_tr = load_mixture_gps(n_samples=train_samples, n_x=n_x, grid_x=True, 
                               data_dir=data_dir, seed=base_seed)
y_tr_tensor = y_tr.unsqueeze(1).float()
loader_tr = DataLoader(y_tr_tensor, batch_size=batch_size, shuffle=True)
loader_tr_signature = DataLoader(y_tr_tensor, batch_size=batch_size_signature, shuffle=True)
print(f"Data shape: {y_tr_tensor.shape}")

# Save ground truth for reference
torch.save(y_tr, spath / 'ground_truth.pt')

# =============================================================================
# Training Function
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
        kwargs["ot_method"] = config.get("ot_method", "sinkhorn")
        if "ot_reg" in config:
            kwargs["ot_reg"] = config["ot_reg"]
        if "ot_reg_m" in config:
            kwargs["ot_reg_m"] = config["ot_reg_m"]  # For unbalanced OT
        if config.get("ot_kernel"):
            kwargs["ot_kernel"] = config["ot_kernel"]
        if config.get("ot_coupling"):
            kwargs["ot_coupling"] = config["ot_coupling"]
        if config.get("ot_kernel_params"):
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
    """
    Train a single configuration with a single seed.
    
    Returns dictionary with training metrics, samples, and quality metrics.
    """
    print(f"\n  Training {config_name} (seed={seed})...")
    
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create model and wrapper
    model = create_model(device)
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
    samples = ffm.sample([n_x], n_samples=n_gen_samples, n_channels=1)
    samples = samples.cpu().squeeze()
    
    # Compute generation quality statistics
    print(f"    Computing quality statistics...")
    quality_metrics = GenerationQualityMetrics(
        config_name=config_name,
        ot_kernel=config.get("ot_kernel", "") or "",
        ot_method=config.get("ot_method", "") or "",
        ot_coupling=config.get("ot_coupling", "") or "",
        use_ot=config.get("use_ot", False),
    )
    quality_metrics.compute_from_samples(ground_truth, samples)
    
    # Add training metrics to quality metrics
    if training_metrics is not None:
        if training_metrics.train_losses:
            quality_metrics.final_train_loss = training_metrics.train_losses[-1]
            quality_metrics.set_convergence_metrics(training_metrics.train_losses)
        if training_metrics.epoch_times:
            quality_metrics.total_train_time = sum(training_metrics.epoch_times)
        if training_metrics.path_lengths:
            quality_metrics.mean_path_length = np.mean(training_metrics.path_lengths)
        if training_metrics.gradient_variances:
            quality_metrics.mean_grad_variance = np.mean(training_metrics.gradient_variances)
    
    # Save everything
    torch.save(samples, save_dir / 'samples.pt')
    torch.save(model.state_dict(), save_dir / 'model.pt')
    if training_metrics is not None:
        training_metrics.save(save_dir / 'training_metrics.json')
    quality_metrics.save(save_dir / 'quality_metrics.json')
    
    # Save config for reference
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    return {
        'samples': samples,
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
    
    Returns dictionary with samples and quality metrics.
    Note: Baselines don't use TrainingMonitor since they're not OT methods.
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
    
    # Compute generation quality statistics
    print(f"    Computing quality statistics...")
    quality_metrics = GenerationQualityMetrics(
        config_name=config_name,
        ot_kernel="",
        ot_method="",
        ot_coupling="",
        use_ot=False,
    )
    quality_metrics.compute_from_samples(ground_truth, samples)
    quality_metrics.total_train_time = train_time
    
    # Save everything
    torch.save(samples, save_dir / 'samples.pt')
    torch.save(model.state_dict(), save_dir / 'model.pt')
    quality_metrics.save(save_dir / 'quality_metrics.json')
    
    # Save config for reference
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    return {
        'samples': samples,
        'training_metrics': None,  # Baselines don't track OT-specific metrics
        'quality_metrics': quality_metrics,
    }


def run_baseline_experiments() -> Dict[str, Dict[str, Any]]:
    """
    Run baseline experiments (DDPM, NCSN) with all seeds.
    
    Returns nested dict: config_name -> seed -> results
    """
    all_results = {}
    
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
                train_loader=loader_tr,
                ground_truth=y_tr,
                save_dir=config_dir,
            )
            
            config_results[seed] = result
        
        all_results[config_name] = config_results
    
    return all_results


def run_all_experiments() -> Dict[str, Dict[str, Any]]:
    """
    Run all configurations with all seeds.
    
    Returns nested dict: config_name -> seed -> results
    """
    all_results = {}
    
    for config_name, config in OT_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Configuration: {config_name}")
        print(f"{'='*60}")
        
        # Use smaller batch size for signature kernel (memory intensive)
        loader = loader_tr_signature if config.get("ot_kernel") == "signature" else loader_tr
        
        config_results = {}
        
        for seed in random_seeds:
            # Create subdirectory for this config + seed
            config_dir = spath / config_name / f"seed_{seed}"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            result = train_single_config(
                config_name=config_name,
                config=config,
                seed=seed,
                train_loader=loader,
                ground_truth=y_tr,
                save_dir=config_dir,
            )
            
            config_results[seed] = result
        
        all_results[config_name] = config_results
    
    return all_results


def load_all_results(include_baselines: bool = True) -> Dict[str, Dict[int, Dict]]:
    """Load all results from saved files instead of training.
    
    Parameters
    ----------
    include_baselines : bool
        Whether to also load DDPM/NCSN baseline results.
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
                continue
            
            samples = None
            if (seed_dir / 'samples.pt').exists():
                samples = torch.load(seed_dir / 'samples.pt', weights_only=True)
            
            training_metrics = None
            if (seed_dir / 'training_metrics.json').exists():
                training_metrics = TrainingMetrics.load(seed_dir / 'training_metrics.json')
            
            quality_metrics = None
            if (seed_dir / 'quality_metrics.json').exists():
                quality_metrics = GenerationQualityMetrics.load(seed_dir / 'quality_metrics.json')
            
            if samples is not None and quality_metrics is not None:
                config_results[seed] = {
                    'samples': samples,
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
                if (seed_dir / 'samples.pt').exists():
                    samples = torch.load(seed_dir / 'samples.pt', weights_only=True)
                
                quality_metrics = None
                if (seed_dir / 'quality_metrics.json').exists():
                    quality_metrics = GenerationQualityMetrics.load(seed_dir / 'quality_metrics.json')
                
                if samples is not None and quality_metrics is not None:
                    config_results[seed] = {
                        'samples': samples,
                        'training_metrics': None,  # Baselines don't have OT training metrics
                        'quality_metrics': quality_metrics,
                    }
            
            if config_results:
                all_results[config_name] = config_results
    
    print(f"Loaded {len(all_results)} configurations")
    return all_results


def aggregate_results(all_results: Dict) -> Dict[str, Dict]:
    """
    Aggregate results across seeds for each config.
    
    Returns dict with aggregated metrics per config.
    """
    aggregated = {}
    
    for config_name, seed_results in all_results.items():
        # Collect quality metrics across seeds
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
        
        # All metrics to average (including spectrum and convergence)
        metrics_to_average = [
            'mean_mse', 'variance_mse', 'skewness_mse', 'kurtosis_mse', 
            'autocorrelation_mse', 'density_mse', 'final_train_loss',
            'total_train_time', 'mean_path_length', 'mean_grad_variance',
            'spectrum_mse', 'spectrum_mse_log', 
            'convergence_rate', 'final_stability',
        ]
        
        for attr in metrics_to_average:
            values = [getattr(q, attr, None) for q in quality_list if getattr(q, attr, None) is not None]
            if values:
                setattr(mean_quality, attr, np.mean(values))
        
        # For epochs_to_90pct, take the median
        epochs_values = [q.epochs_to_90pct for q in quality_list if q.epochs_to_90pct is not None]
        if epochs_values:
            mean_quality.epochs_to_90pct = int(np.median(epochs_values))
        
        # Take first seed's training metrics for visualization
        representative_training = training_list[0] if training_list else None
        
        # Get first seed's samples for visualization
        first_seed = list(seed_results.keys())[0]
        representative_samples = seed_results[first_seed]['samples']
        
        aggregated[config_name] = {
            'mean_quality': mean_quality,
            'training_metrics': representative_training,
            'samples': representative_samples,
            'all_quality': quality_list,
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
    x_grid = torch.linspace(0, 1, n_x)
    n_plot = 30
    
    # Create a subdirectory for sample comparison plots
    samples_dir = save_path / 'sample_comparisons'
    samples_dir.mkdir(exist_ok=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(aggregated)))
    
    for idx, (config_name, data) in enumerate(aggregated.items()):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        samples = data['samples']
        
        # Left: Ground Truth
        ax = axes[0]
        for i in range(min(n_plot, ground_truth.shape[0])):
            ax.plot(x_grid.numpy(), ground_truth[i].numpy(), color='gray', alpha=0.4, linewidth=0.8)
        ax.set_title(f'Ground Truth\n({ground_truth.shape[0]} samples)', fontsize=11, fontweight='bold')
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Right: Generated samples
        ax = axes[1]
        for i in range(min(n_plot, samples.shape[0])):
            ax.plot(x_grid.numpy(), samples[i].numpy(), color=colors[idx], alpha=0.4, linewidth=0.8)
        
        title = config_name.replace('_', ' ').title()
        ax.set_title(f'{title}\n({samples.shape[0]} samples)', fontsize=11, fontweight='bold')
        ax.set_xlabel('x', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Match y-axis limits
        y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
        y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
        axes[0].set_ylim(y_min, y_max)
        axes[1].set_ylim(y_min, y_max)
        
        plt.suptitle(f'moGP: Ground Truth vs {config_name}', fontsize=12, y=1.02)
        plt.tight_layout()
        
        # Save individual plot
        safe_name = config_name.replace('/', '_').replace('\\', '_')
        plt.savefig(samples_dir / f'{safe_name}.pdf', dpi=150, bbox_inches='tight')
        plt.savefig(samples_dir / f'{safe_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {len(aggregated)} individual sample comparison plots to {samples_dir}")


def create_summary_report(aggregated: Dict, save_path: Path):
    """Create a summary JSON report of all experiments."""
    summary = {}
    
    for config_name, data in aggregated.items():
        q = data['mean_quality']
        summary[config_name] = {
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
    print("Comprehensive OT-FFM Experiments on Mixture of GPs")
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
        all_results = load_all_results(include_baselines=True)
        if not all_results:
            print("ERROR: No saved results found. Run without --load-only first.")
            sys.exit(1)
    elif args.baselines_only:
        # Run only baseline experiments
        all_results = run_baseline_experiments()
        print("\nBaseline experiments complete!")
        print("Run with --load-only to generate visualizations with all results.")
        sys.exit(0)
    else:
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
    plot_samples_comparison(aggregated, y_tr, spath)
    
    # 2. Spectrum comparison (OT methods only - for ablation)
    if ot_aggregated:
        print("\nGenerating spectrum comparison (OT methods only)...")
        generated_list = [data['samples'] for data in ot_aggregated.values()]
        config_names = list(ot_aggregated.keys())
        quality_list = [data['mean_quality'] for data in ot_aggregated.values()]
        
        # Top-5 spectrum comparison
        fig = compare_spectra_1d(
            y_tr,
            generated_list,
            config_names=config_names,
            save_path=spath / 'spectrum_comparison_top5.pdf',
            top_k=5,
        )
        plt.close(fig)
        # Best per category spectrum comparison
        fig = compare_spectra_1d(
            y_tr,
            generated_list,
            config_names=config_names,
            save_path=spath / 'spectrum_comparison.pdf',
            quality_metrics_list=quality_list,
            save_best_per_category=True,
        )
        plt.close(fig)
    
    # 3. Training metrics comparison (OT methods only - for ablation)
    training_metrics_list = [
        data['training_metrics'] for name, data in ot_aggregated.items() 
        if data['training_metrics'] is not None
    ]
    quality_metrics_list_ot = [
        data['mean_quality'] for name, data in ot_aggregated.items()
        if data['training_metrics'] is not None
    ]
    if training_metrics_list:
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
    
    # 4. Convergence analysis (OT methods only - for ablation)
    losses_dict = {
        name: data['training_metrics'].train_losses 
        for name, data in ot_aggregated.items() 
        if data['training_metrics'] is not None
    }
    if losses_dict:
        figs, conv_metrics = compare_convergence(
            losses_dict,
            save_path=spath / 'convergence_metrics'
        )
        for fig in figs:
            plt.close(fig)
        
        # Simplified convergence (best per kernel type)
        quality_dict = {name: data['mean_quality'].mean_mse for name, data in ot_aggregated.items()
                       if data['mean_quality'] is not None and data['mean_quality'].mean_mse is not None}
        fig_simple = compare_convergence_simplified(
            losses_dict,
            quality_metrics_dict=quality_dict,
            save_path=spath / 'convergence_metrics'
        )
        if fig_simple:
            plt.close(fig_simple)
        
        with open(spath / 'convergence_metrics.json', 'w') as f:
            conv_json = {
                k: {kk: float(vv) if vv is not None else None for kk, vv in v.items()}
                for k, v in conv_metrics.items()
            }
            json.dump(conv_json, f, indent=2)
        print_convergence_table(conv_metrics)
    
    # 5. Generation quality comparison - ALL methods including baselines
    quality_metrics_list = [data['mean_quality'] for data in aggregated.values()]
    # Top-5 ranked comparison (all methods)
    compare_generation_quality(
        quality_metrics_list, 
        save_path=spath / 'quality_comparison_top5.pdf',
        include_spectrum=True,
        include_seasonal=False,
        top_k=5,
    )
    # All configs for reference (all methods including baselines)
    compare_generation_quality(
        quality_metrics_list, 
        save_path=spath / 'quality_comparison_all.pdf',
        include_spectrum=True,
        include_seasonal=False,
        show_all=True,
    )
    # Best per kernel category (OT methods only for ablation)
    if ot_aggregated:
        quality_metrics_list_ot = [data['mean_quality'] for data in ot_aggregated.values()]
        print("\nGenerating simplified quality comparison (best per kernel type)...")
        compare_generation_quality_simplified(
            quality_metrics_list_ot,
            save_path=spath / 'quality_comparison.pdf',
        )
    
    # Print comprehensive metrics and save
    print_comprehensive_table(quality_metrics_list)
    save_all_metrics_summary(quality_metrics_list, spath / 'comprehensive_metrics.json')
    
    # 6. Create summary report
    create_summary_report(aggregated, spath)
    
    # Cleanup
    plt.close('all')

    print("\n" + "="*60)
    print(f"All experiments complete! Results saved to: {spath}")
    print("="*60)

    # Print directory structure
    print("\nDirectory structure:")
    all_configs = list(OT_CONFIGS.keys()) + list(BASELINE_CONFIGS.keys())
    for config_name in all_configs:
        config_dir = spath / config_name
        if config_dir.exists():
            marker = "[baseline]" if config_name in BASELINE_CONFIGS else ""
            print(f"  {config_name}/ {marker}")
            for seed_dir in sorted(config_dir.iterdir()):
                if seed_dir.is_dir():
                    print(f"    {seed_dir.name}/")
                    for f in sorted(seed_dir.iterdir()):
                        print(f"      {f.name}")
