"""
Comprehensive OT-FFM Experiments on Gene Expression Time Series.

This script runs experiments with multiple OT configurations on the gene
expression dataset comparing:
- Different kernel types (euclidean, rbf, signature)
- Different OT methods (exact, sinkhorn)
- Different coupling strategies (sample, barycentric)

Each configuration gets its own subdirectory with:
- Trained model checkpoint
- Generated samples (original and upsampled resolution)
- Training metrics (loss, path length, gradient variance)
- Generation quality statistics (Mean, Variance, Skewness, Kurtosis, Autocorrelation MSE)

Usage:
    python expr_genes_ot.py                    # Run all experiments
    python expr_genes_ot.py --load-only        # Load saved results and regenerate plots
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

# =============================================================================
# Argument Parsing
# =============================================================================

parser = argparse.ArgumentParser(description='Gene Expression OT-FFM Experiments')
parser.add_argument('--load-only', action='store_true',
                    help='Load saved results instead of training')
parser.add_argument('--baselines-only', action='store_true',
                    help='Run only DDPM and NCSN baselines (not OT experiments)')
parser.add_argument('--spath', type=str, default='../outputs/expr_genes_ot_comprehensive/',
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
    compute_all_pointwise_statistics,
    compare_spectra_1d,
    compare_convergence,
    compare_convergence_simplified,
)

from functional_fm_ot import FFMModelOT
from diffusion import DiffusionModel
from models.fno import FNO

# =============================================================================
# Load and Preprocess Data
# =============================================================================

print("Loading gene expression data...")
full_data = torch.load('../data/full_genes.pt').float()  # Use float32 for GPU efficiency
centered_loggen = full_data.log10() - full_data.log10().mean(1).unsqueeze(-1)
expr_genes = centered_loggen[(centered_loggen.std(1) > .3), :]

print(f"Gene expression data shape: {expr_genes.shape}")
print(f"Number of genes: {expr_genes.shape[0]}")
print(f"Number of time points: {expr_genes.shape[1]}")

n_x = expr_genes.shape[1]

# =============================================================================
# Configuration
# =============================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

batch_size = 512  # Larger batch for OT
num_workers = 0
pin_memory = True

train_loader = DataLoader(
    expr_genes.unsqueeze(1),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

# Model params
modes = 16
width = 256
mlp_width = 128

# GP hyperparameters
kernel_length = 0.01
kernel_variance = 0.1

# Training params
epochs = 200
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
    
    # Generate samples at original and upsampled resolution
    print(f"    Generating {n_gen_samples} samples...")
    samples_original = ffm.sample([n_x], n_samples=n_gen_samples).cpu().squeeze()
    samples_upsampled = ffm.sample([n_x * upsample], n_samples=n_gen_samples).cpu().squeeze()
    
    # Compute generation quality statistics
    print(f"    Computing quality statistics...")
    quality_metrics = GenerationQualityMetrics(
        config_name=config_name,
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
            # Compute convergence metrics from the loss history
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
    samples_original = diffusion.sample([n_x], n_samples=n_gen_samples, n_channels=1).cpu().squeeze()
    samples_upsampled = diffusion.sample([n_x * upsample], n_samples=n_gen_samples, n_channels=1).cpu().squeeze()
    
    quality_metrics = GenerationQualityMetrics(
        config_name=config_name, ot_kernel="", ot_method="", ot_coupling="", use_ot=False,
    )
    quality_metrics.compute_from_samples(ground_truth, samples_original)
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
            result = train_baseline_config(config_name, config, seed, train_loader, expr_genes, config_dir)
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
                ground_truth=expr_genes,
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
            
            samples_original = None
            samples_upsampled = None
            if (seed_dir / 'samples_original.pt').exists():
                samples_original = torch.load(seed_dir / 'samples_original.pt', weights_only=True)
            if (seed_dir / 'samples_upsampled.pt').exists():
                samples_upsampled = torch.load(seed_dir / 'samples_upsampled.pt', weights_only=True)
            
            training_metrics = None
            if (seed_dir / 'training_metrics.json').exists():
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
        epochs_values = [q.epochs_to_90pct for q in quality_list if getattr(q, 'epochs_to_90pct', None) is not None]
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
        
        samples = data['samples_original']
        
        # Left: Ground Truth
        ax = axes[0]
        for i in range(min(n_plot, ground_truth.shape[0])):
            ax.plot(x_grid.numpy(), ground_truth[i].numpy(), color='gray', alpha=0.4, linewidth=0.8)
        ax.set_title(f'Ground Truth\n({ground_truth.shape[0]} genes)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Expression', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Right: Generated samples
        ax = axes[1]
        for i in range(min(n_plot, samples.shape[0])):
            ax.plot(x_grid.numpy(), samples[i].numpy(), color=colors[idx], alpha=0.4, linewidth=0.8)
        
        title = config_name.replace('_', ' ').title()
        ax.set_title(f'{title}\n({samples.shape[0]} samples)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Match y-axis limits
        y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
        y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
        axes[0].set_ylim(y_min, y_max)
        axes[1].set_ylim(y_min, y_max)
        
        plt.suptitle(f'Gene Expression: Ground Truth vs {config_name}', fontsize=12, y=1.02)
        plt.tight_layout()
        
        # Save individual plot
        safe_name = config_name.replace('/', '_').replace('\\', '_')
        plt.savefig(samples_dir / f'{safe_name}.pdf', dpi=150, bbox_inches='tight')
        plt.savefig(samples_dir / f'{safe_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {len(aggregated)} individual sample comparison plots to {samples_dir}")


def plot_upsampled_comparison(
    aggregated: Dict,
    save_path: Path,
):
    """Create comparison of upsampled (super-resolution) samples."""
    n_configs = len(aggregated)
    
    fig, axes = plt.subplots(1, n_configs, figsize=(4 * n_configs, 4))
    if n_configs == 1:
        axes = [axes]
    
    x_grid = torch.linspace(0, 1, n_x * upsample)
    n_plot = 30
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_configs))
    
    for idx, (config_name, data) in enumerate(aggregated.items()):
        ax = axes[idx]
        samples = data['samples_upsampled']
        
        for i in range(min(n_plot, samples.shape[0])):
            ax.plot(x_grid.numpy(), samples[i].numpy(), color=colors[idx], alpha=0.3, linewidth=0.5)
        
        title = f"{config_name}\n(5x super-resolution)"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Time')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'upsampled_comparison.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(save_path / 'upsampled_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved upsampled comparison to {save_path / 'upsampled_comparison.pdf'}")


def create_summary_report(aggregated: Dict, save_path: Path):
    """Create comprehensive summary JSON report."""
    summary = {
        'dataset': 'gene_expression',
        'n_genes': int(expr_genes.shape[0]),
        'n_timepoints': int(n_x),
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
            'ot_method': q.ot_method,
            'ot_coupling': q.ot_coupling,
            'quality_metrics': {
                'mean_mse': float(q.mean_mse) if q.mean_mse is not None else None,
                'variance_mse': float(q.variance_mse) if q.variance_mse is not None else None,
                'skewness_mse': float(q.skewness_mse) if q.skewness_mse is not None else None,
                'kurtosis_mse': float(q.kurtosis_mse) if q.kurtosis_mse is not None else None,
                'autocorrelation_mse': float(q.autocorrelation_mse) if q.autocorrelation_mse is not None else None,
                'density_mse': float(q.density_mse) if q.density_mse is not None else None,
                'spectrum_mse': float(q.spectrum_mse) if getattr(q, 'spectrum_mse', None) is not None else None,
                'convergence_rate': float(q.convergence_rate) if getattr(q, 'convergence_rate', None) is not None else None,
                'final_stability': float(q.final_stability) if getattr(q, 'final_stability', None) is not None else None,
                'epochs_to_90pct': int(q.epochs_to_90pct) if getattr(q, 'epochs_to_90pct', None) is not None else None,
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
    print("Comprehensive OT-FFM Experiments on Gene Expression Data")
    print(f"Configs: {len(OT_CONFIGS)}, Seeds per config: {n_seeds}")
    print(f"Output directory: {spath}")
    if args.baselines_only:
        print("MODE: Baselines-only (running DDPM and NCSN only)")
    elif args.load_only:
        print("MODE: Load-only (loading saved results)")
    else:
        print("MODE: Training (running experiments)")
    print("="*60)
    
    if args.baselines_only:
        baseline_results = run_baseline_experiments()
        ot_results = load_all_results(include_baselines=False)
        all_results = {**ot_results, **baseline_results}
    elif args.load_only:
        all_results = load_all_results(include_baselines=True)
        if not all_results:
            print("ERROR: No saved results found. Run without --load-only first.")
            sys.exit(1)
    else:
        # Save ground truth
        torch.save(expr_genes, spath / 'ground_truth.pt')
        
        # Run all experiments
        all_results = run_all_experiments()
    
    # Aggregate results
    print("\n" + "="*60)
    print("Aggregating results...")
    print("="*60)
    aggregated = aggregate_results(all_results)
    
    # Filter baselines for OT-specific plots
    baseline_names = set(BASELINE_CONFIGS.keys())
    ot_aggregated = {k: v for k, v in aggregated.items() if k not in baseline_names}
    
    # Generate visualizations
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)
    
    # 1. Sample comparison plot
    plot_samples_comparison(aggregated, expr_genes, spath)
    
    # 2. Upsampled comparison plot
    plot_upsampled_comparison(aggregated, spath)
    
    # 3. Training metrics comparison (OT configs only)
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
    
    # 4. Generation quality comparison
    quality_metrics_list = [data['mean_quality'] for data in aggregated.values()]
    # Top-5 ranked comparison
    compare_generation_quality(quality_metrics_list, save_path=spath / 'quality_comparison_top5.pdf', top_k=5)
    # All configs for reference
    compare_generation_quality(quality_metrics_list, save_path=spath / 'quality_comparison_all.pdf', show_all=True)
    # Best per kernel category
    print("\nGenerating simplified quality comparison (best per kernel type)...")
    compare_generation_quality_simplified(quality_metrics_list, save_path=spath / 'quality_comparison.pdf')
    print_quality_table(quality_metrics_list)
    
    # 5. Create summary report
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

