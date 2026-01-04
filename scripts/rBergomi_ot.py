"""
Comprehensive OT-FFM Experiments on rBergomi Rough Volatility.

This script runs experiments with multiple OT configurations on rBergomi
log-variance paths comparing:
- Different kernel types (euclidean, rbf, signature)
- Different OT methods (exact, sinkhorn)
- Different coupling strategies (sample, barycentric)

The rBergomi model generates "rough" variance paths with Hurst parameter
H = alpha + 0.5. For the default alpha=-0.4, H=0.1, which is much rougher
than Brownian motion (H=0.5).

HYPOTHESIS: Signature-OT should excel on rough paths because:
1. Signature kernels capture path roughness/regularity
2. Rough paths have complex temporal structure
3. Euclidean/RBF OT treats paths as flat vectors, missing structure

Usage:
    python rBergomi_ot.py
    python rBergomi_ot.py --alpha -0.3  # Different roughness (H=0.2)
"""

import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Any
import argparse

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

# Import data generator
from data.rBergomi import rBergomi, generate_rBergomi_dataset

# =============================================================================
# Configuration
# =============================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Parse arguments
parser = argparse.ArgumentParser(description='rBergomi OT-FFM Experiments')
parser.add_argument('--alpha', type=float, default=-0.4, help='Roughness parameter')
parser.add_argument('--n_samples', type=int, default=5000, help='Number of samples')
parser.add_argument('--n_steps', type=int, default=100, help='Time steps')
parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
parser.add_argument('--n_seeds', type=int, default=3, help='Number of random seeds')
parser.add_argument('--load-only', action='store_true', help='Load saved results instead of training')
parser.add_argument('--baselines-only', action='store_true', help='Run only DDPM and NCSN baselines')
parser.add_argument('--spath', type=str, default=None, help='Output/load directory')
parser.add_argument('--config-file', type=str, default=None,
                    help='Path to JSON/YAML file with additional OT configurations')
parser.add_argument('--config', type=str, default=None, 
                    help='Run only this config (for parallelization). Use --list-configs to see available.')
parser.add_argument('--seed', type=int, default=None,
                    help='Run only this seed (use with --config for single run)')
parser.add_argument('--list-configs', action='store_true', help='List available configurations and exit')
args, _ = parser.parse_known_args()

# Data params
n_samples = args.n_samples
n_x = args.n_steps
alpha = args.alpha
H = alpha + 0.5

print(f"Roughness: alpha={alpha}, H={H:.2f}")

# Generate dataset
print("Generating rBergomi dataset...")
dataset = generate_rBergomi_dataset(
    n_samples=n_samples,
    n_steps=n_x,
    T=1.0,
    alpha=alpha,
    rho=-0.7,
    eta=1.5,
    xi=0.04,
    seed=42,
)

# Use normalized log-variance for training
train_data = dataset['log_V_normalized']
ground_truth = train_data.clone()
print(f"Training data shape: {train_data.shape}")

batch_size = 512  # Larger batch for OT
batch_size_signature = 128  # Smaller batch for signature kernel (memory intensive)

train_loader = DataLoader(
    train_data.unsqueeze(1),
    batch_size=batch_size,
    shuffle=True,
)
train_loader_signature = DataLoader(
    train_data.unsqueeze(1),
    batch_size=batch_size_signature,
    shuffle=True,
)

# Model hyperparameters
modes = 32
width = 256
mlp_width = 128

# GP hyperparameters
kernel_length = 0.01
kernel_variance = 0.1

# Training params
epochs = args.epochs
lr = 1e-3

# FFM params
sigma_min = 1e-4

# Random seeds
n_seeds = args.n_seeds
random_seeds = [2**i for i in range(n_seeds)]

# Number of samples to generate
n_gen_samples = 500

# Output directory
H_str = f"{H:.2f}".replace('.', 'p')
if args.spath:
    spath = Path(args.spath)
else:
    spath = Path(f'../outputs/rBergomi_ot_H{H_str}/')
spath.mkdir(parents=True, exist_ok=True)

# Save dataset
torch.save(dataset, spath / 'dataset.pt')

# =============================================================================
# OT Configurations
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
    # Signature Kernel OT - All Methods (should excel on rough paths!)
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
            "normalize": True,
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
# Config Loading Functions
# =============================================================================

def load_configs_from_file(config_path: str) -> Dict:
    """
    Load additional OT configurations from JSON or YAML file.
    
    Expected format:
    {
        "config_name": {
            "use_ot": true,
            "ot_method": "sinkhorn",
            "ot_reg": 0.1,
            "ot_kernel": "signature",
            "ot_coupling": "sample",
            "ot_kernel_params": {...}
        },
        ...
    }
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if config_path.suffix in ['.yaml', '.yml']:
        try:
            import yaml
            with open(config_path, 'r') as f:
                configs = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required for YAML config files. Install with: pip install pyyaml")
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            configs = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}. Use .json or .yaml")
    
    print(f"Loaded {len(configs)} configurations from {config_path}")
    return configs


def get_all_configs(include_external: bool = True) -> Dict:
    """Get all OT configurations, optionally including external config file."""
    all_configs = dict(OT_CONFIGS)
    
    if include_external and args.config_file:
        external_configs = load_configs_from_file(args.config_file)
        all_configs.update(external_configs)
    
    return all_configs


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
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    model = create_model(device)
    ffm_kwargs = build_ffm_kwargs(config)
    ffm = FFMModelOT(model, **ffm_kwargs)
    
    monitor = TrainingMonitor(
        method_name=config_name,
        use_ot=config.get("use_ot", False),
        ot_kernel=config.get("ot_kernel", "") or "",
        track_batch_metrics=False,
    )
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50)
    
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
    
    print(f"    Generating {n_gen_samples} samples...")
    samples = ffm.sample([n_x], n_samples=n_gen_samples).cpu().squeeze()
    
    print(f"    Computing quality statistics...")
    quality_metrics = GenerationQualityMetrics(
        config_name=config_name,
        ot_kernel=config.get("ot_kernel", "") or "",
        ot_method=config.get("ot_method", "") or "",
        ot_coupling=config.get("ot_coupling", "") or "",
        use_ot=config.get("use_ot", False),
    )
    quality_metrics.compute_from_samples(ground_truth, samples)
    
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
    
    # Save
    torch.save(samples, save_dir / 'samples.pt')
    torch.save(model.state_dict(), save_dir / 'model.pt')
    if training_metrics is not None:
        training_metrics.save(save_dir / 'training_metrics.json')
    quality_metrics.save(save_dir / 'quality_metrics.json')
    
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
    """Train a baseline (DDPM/NCSN) configuration."""
    import time
    
    print(f"\n  Training {config_name} (seed={seed})...")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    model = create_model(device)
    
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
    diffusion.train(train_loader=train_loader, optimizer=optimizer, epochs=epochs,
                    scheduler=scheduler, eval_int=0, save_int=epochs,
                    generate=False, save_path=save_dir)
    train_time = time.time() - t0
    
    print(f"    Generating {n_gen_samples} samples...")
    samples = diffusion.sample([n_x], n_samples=n_gen_samples, n_channels=1).cpu().squeeze()
    
    quality_metrics = GenerationQualityMetrics(
        config_name=config_name, ot_kernel="", ot_method="", ot_coupling="", use_ot=False,
    )
    quality_metrics.compute_from_samples(ground_truth, samples)
    quality_metrics.total_train_time = train_time
    
    torch.save(samples, save_dir / 'samples.pt')
    torch.save(model.state_dict(), save_dir / 'model.pt')
    quality_metrics.save(save_dir / 'quality_metrics.json')
    
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    return {'samples': samples, 'training_metrics': None, 'quality_metrics': quality_metrics}


def run_baseline_experiments() -> Dict[str, Dict[str, Any]]:
    """Run baseline experiments (DDPM, NCSN)."""
    all_results = {}
    
    for config_name, config in BASELINE_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Baseline: {config_name}")
        print(f"{'='*60}")
        
        config_results = {}
        for seed in random_seeds:
            config_dir = spath / config_name / f"seed_{seed}"
            config_dir.mkdir(parents=True, exist_ok=True)
            result = train_baseline_config(config_name, config, seed, train_loader, ground_truth, config_dir)
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
        
        # Use smaller batch size for signature kernel (memory intensive)
        loader = train_loader_signature if config.get("ot_kernel") == "signature" else train_loader
        
        config_results = {}
        
        for seed in random_seeds:
            config_dir = spath / config_name / f"seed_{seed}"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            result = train_single_config(
                config_name=config_name,
                config=config,
                seed=seed,
                train_loader=loader,
                ground_truth=ground_truth,
                save_dir=config_dir,
            )
            
            config_results[seed] = result
        
        all_results[config_name] = config_results
    
    return all_results


def load_all_results(include_baselines: bool = True) -> Dict[str, Dict[int, Dict]]:
    """Load all results from saved files instead of training."""
    print("Loading saved results from", spath)
    
    all_results = {}
    
    all_configs = dict(OT_CONFIGS)
    if include_baselines:
        all_configs.update(BASELINE_CONFIGS)
    
    for config_name in all_configs.keys():
        config_dir = spath / config_name
        if not config_dir.exists():
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
    
    print(f"Loaded {len(all_results)} configurations")
    return all_results


def aggregate_results(all_results: Dict) -> Dict[str, Dict]:
    """Aggregate results across seeds."""
    
    aggregated = {}
    
    for config_name, seed_results in all_results.items():
        quality_list = [r['quality_metrics'] for r in seed_results.values()]
        training_list = [r['training_metrics'] for r in seed_results.values() if r['training_metrics']]
        
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
        
        representative_training = training_list[0] if training_list else None
        first_seed = list(seed_results.keys())[0]
        
        aggregated[config_name] = {
            'mean_quality': mean_quality,
            'training_metrics': representative_training,
            'samples': seed_results[first_seed]['samples'],
        }
    
    return aggregated


# =============================================================================
# Visualization
# =============================================================================

def plot_samples_comparison(aggregated: Dict, ground_truth: torch.Tensor, save_path: Path):
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
        ax.set_title(f'Ground Truth\n(rBergomi H={H:.2f})', fontsize=11, fontweight='bold')
        ax.set_xlabel('t', fontsize=10)
        ax.set_ylabel('log V (normalized)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Right: Generated samples
        ax = axes[1]
        for i in range(min(n_plot, samples.shape[0])):
            ax.plot(x_grid.numpy(), samples[i].numpy(), color=colors[idx], alpha=0.4, linewidth=0.8)
        
        title = config_name.replace('_', ' ').title()
        ax.set_title(f'{title}\n({samples.shape[0]} samples)', fontsize=11, fontweight='bold')
        ax.set_xlabel('t', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Match y-axis limits
        y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
        y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
        axes[0].set_ylim(y_min, y_max)
        axes[1].set_ylim(y_min, y_max)
        
        plt.suptitle(f'rBergomi: Ground Truth vs {config_name}', fontsize=12, y=1.02)
        plt.tight_layout()
        
        # Save individual plot
        safe_name = config_name.replace('/', '_').replace('\\', '_')
        plt.savefig(samples_dir / f'{safe_name}.pdf', dpi=150, bbox_inches='tight')
        plt.savefig(samples_dir / f'{safe_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {len(aggregated)} individual sample comparison plots to {samples_dir}")


def create_summary_report(aggregated: Dict, save_path: Path):
    """Create summary JSON."""
    summary = {
        'dataset': 'rBergomi',
        'alpha': alpha,
        'H': H,
        'n_samples': n_samples,
        'n_steps': n_x,
        'epochs': epochs,
        'batch_size': batch_size,
        'n_seeds': n_seeds,
        'configs': {}
    }
    
    for config_name, data in aggregated.items():
        q = data['mean_quality']
        summary['configs'][config_name] = {
            'use_ot': q.use_ot,
            'ot_kernel': q.ot_kernel,
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


# =============================================================================
# Main
# =============================================================================
# Single Config Runner (for parallelization)
# =============================================================================

def run_single_config_with_seeds(config_name: str, seeds_to_run: list = None):
    """Run a single configuration with specified seeds."""
    # Get all configs including external
    all_ot_configs = get_all_configs()
    
    # Determine which config dict to use
    if config_name in all_ot_configs:
        config = all_ot_configs[config_name]
        is_baseline = False
    elif config_name in BASELINE_CONFIGS:
        config = BASELINE_CONFIGS[config_name]
        is_baseline = True
    else:
        print(f"ERROR: Unknown config '{config_name}'")
        print(f"Available OT configs: {list(all_ot_configs.keys())}")
        print(f"Available baseline configs: {list(BASELINE_CONFIGS.keys())}")
        sys.exit(1)
    
    seeds = seeds_to_run if seeds_to_run else random_seeds
    
    print(f"\n{'='*60}")
    print(f"Configuration: {config_name}")
    print(f"Seeds: {seeds}")
    print(f"{'='*60}")
    
    # Select appropriate loader for signature kernel configs
    loader = train_loader_signature if config.get("ot_kernel") == "signature" else train_loader
    
    config_results = {}
    for seed in seeds:
        config_dir = spath / config_name / f"seed_{seed}"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        if is_baseline:
            result = train_baseline_config(
                config_name, config, seed, train_loader, ground_truth, config_dir
            )
        else:
            result = train_single_config(
                config_name, config, seed, loader, ground_truth, config_dir
            )
        config_results[seed] = result
    
    return {config_name: config_results}


def list_all_configs():
    """Print all available configurations."""
    all_ot_configs = get_all_configs()
    
    print("\n" + "="*70)
    print("Available OT Configurations:")
    print("="*70)
    
    for config_name, config in all_ot_configs.items():
        kernel = config.get('ot_kernel', 'none')
        method = config.get('ot_method', 'none')
        reg = config.get('ot_reg', 'N/A')
        coupling = config.get('ot_coupling', 'N/A')
        use_ot = config.get('use_ot', False)
        
        if use_ot:
            print(f"  {config_name}")
            print(f"    kernel={kernel}, method={method}, reg={reg}, coupling={coupling}")
            if 'ot_kernel_params' in config:
                params = config['ot_kernel_params']
                if kernel == 'signature':
                    print(f"    signature: dyadic_order={params.get('dyadic_order')}, "
                          f"lead_lag={params.get('lead_lag')}, "
                          f"static_sigma={params.get('static_kernel_sigma')}")
                elif kernel == 'rbf':
                    print(f"    rbf: sigma={params.get('sigma')}")
        else:
            print(f"  {config_name} (independent, no OT)")
    
    print("\n" + "="*70)
    print("Available Baseline Configurations:")
    print("="*70)
    for config_name, config in BASELINE_CONFIGS.items():
        print(f"  {config_name}: {config.get('method')}")
    
    print(f"\nTotal: {len(all_ot_configs)} OT configs + {len(BASELINE_CONFIGS)} baselines")
    print(f"Seeds: {random_seeds}")
    
    print("\n" + "="*70)
    print("Usage Examples:")
    print("="*70)
    print(f"  # Run single config with single seed:")
    print(f"  python {sys.argv[0]} --config signature_sinkhorn_reg0.1 --seed 1")
    print()
    print(f"  # Run with external config file:")
    print(f"  python {sys.argv[0]} --config-file ../configs/rbergomi_sweep.yaml --config sig_leadlag")
    print()
    print(f"  # After all parallel runs complete:")
    print(f"  python {sys.argv[0]} --load-only")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Handle --list-configs first
    if args.list_configs:
        list_all_configs()
        sys.exit(0)
    
    print("="*60)
    print(f"OT-FFM Experiments on rBergomi (H={H:.2f})")
    print(f"Output: {spath}")
    
    # Determine mode
    if args.config:
        seeds_to_run = [args.seed] if args.seed else random_seeds
        print(f"MODE: Single config ('{args.config}', seeds={seeds_to_run})")
    elif args.baselines_only:
        print(f"MODE: Baselines-only (running DDPM and NCSN only)")
    elif args.load_only:
        print(f"MODE: Load-only (loading and aggregating saved results)")
    else:
        print(f"MODE: Full training (all {len(OT_CONFIGS)} configs, {n_seeds} seeds)")
    print("="*60)
    
    # Execute based on mode
    if args.config:
        # Single config mode (for parallelization)
        seeds_to_run = [args.seed] if args.seed else random_seeds
        all_results = run_single_config_with_seeds(args.config, seeds_to_run)
        print(f"\nCompleted {args.config} with seeds {seeds_to_run}")
        print(f"Results saved to: {spath / args.config}")
        print("\nTo aggregate all results after parallel runs complete:")
        print(f"  python {sys.argv[0]} --load-only")
        sys.exit(0)
    elif args.baselines_only:
        # Only run baselines, don't load any OT results (fixes signature kernel init issue)
        baseline_results = run_baseline_experiments()
        all_results = baseline_results
    elif args.load_only:
        all_results = load_all_results(include_baselines=True)
        if not all_results:
            print("ERROR: No saved results found. Run without --load-only first.")
            sys.exit(1)
    else:
        all_results = run_all_experiments()
    
    # Aggregate
    print("\n" + "="*60)
    print("Aggregating results...")
    print("="*60)
    aggregated = aggregate_results(all_results)
    
    # Filter baselines for OT-specific plots
    baseline_names = set(BASELINE_CONFIGS.keys())
    ot_aggregated = {k: v for k, v in aggregated.items() if k not in baseline_names}
    
    # Visualize
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)
    
    plot_samples_comparison(aggregated, ground_truth, spath)
    
    # Spectrum comparison (OT configs only)
    print("\nGenerating spectrum comparison...")
    generated_list = [data['samples'] for data in ot_aggregated.values()]
    config_names = list(ot_aggregated.keys())
    quality_list = [data['mean_quality'] for data in ot_aggregated.values()]
    
    # Top-5 spectrum comparison
    fig = compare_spectra_1d(
        ground_truth,
        generated_list,
        config_names=config_names,
        save_path=spath / 'spectrum_comparison_top5.pdf',
        top_k=5,
    )
    plt.close(fig)
    # Best per category spectrum comparison
    fig = compare_spectra_1d(
        ground_truth,
        generated_list,
        config_names=config_names,
        save_path=spath / 'spectrum_comparison.pdf',
        quality_metrics_list=quality_list,
        save_best_per_category=True,
    )
    plt.close(fig)
    
    # Training metrics comparison (OT configs only)
    training_metrics_list = [
        data['training_metrics'] for data in ot_aggregated.values()
        if data['training_metrics'] is not None
    ]
    quality_metrics_list = [
        data['mean_quality'] for data in ot_aggregated.values()
        if data['training_metrics'] is not None
    ]
    if training_metrics_list:
        figs = compare_training_runs(training_metrics_list, save_path=spath / 'training')
        for fig in figs:
            plt.close(fig)
        
        figs_simple = compare_training_runs_simplified(
            training_metrics_list, quality_metrics_list=quality_metrics_list, metric_key='mean_mse',
            save_path=spath / 'training_simplified'
        )
        for fig in figs_simple:
            plt.close(fig)
        
        plot_convergence_comparison(training_metrics_list, save_path=spath / 'convergence_comparison.pdf')
        print_comparison_table(training_metrics_list)
    
    # Convergence analysis with numerical metrics (OT configs only)
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
    
    # Quality comparison (including spectrum metrics)
    quality_metrics_list = [data['mean_quality'] for data in aggregated.values()]
    # Top-5 ranked comparison
    compare_generation_quality(
        quality_metrics_list, 
        save_path=spath / 'quality_comparison_top5.pdf',
        include_spectrum=True,
        include_seasonal=False,
        top_k=5,
    )
    # All configs for reference
    compare_generation_quality(
        quality_metrics_list, 
        save_path=spath / 'quality_comparison_all.pdf',
        include_spectrum=True,
        include_seasonal=False,
        show_all=True,
    )
    # Best per kernel category
    print("\nGenerating simplified quality comparison (best per kernel type)...")
    compare_generation_quality_simplified(
        quality_metrics_list,
        save_path=spath / 'quality_comparison.pdf',
    )
    
    # Print comprehensive metrics and save
    print_comprehensive_table(quality_metrics_list)
    save_all_metrics_summary(quality_metrics_list, spath / 'comprehensive_metrics.json')
    
    create_summary_report(aggregated, spath)
    
    plt.close('all')
    
    print("\n" + "="*60)
    print(f"Complete! Results: {spath}")
    print("="*60)

