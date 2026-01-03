"""
Comprehensive OT-FFM Experiments on Navier-Stokes.

This script runs experiments with multiple OT configurations on Navier-Stokes
equation solutions (2D spatial) comparing:
- Different kernel types (euclidean, rbf)
- Different OT methods (exact, sinkhorn, gaussian)
- Different coupling strategies (sample, barycentric)

Note: Signature kernel is not included for 2D data as it's designed for 1D paths.

Data shape: (batch_size, N, N, T) - typically (10, 64, 64, 15001)

Usage:
    python navier_stokes_ot.py
    python navier_stokes_ot.py --epochs 50 --n_seeds 3
"""

import sys
sys.path.append('../')

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from typing import Dict, Any
import argparse

from torch.optim.lr_scheduler import StepLR

from util.util import load_navier_stokes, plot_samples
from util.ot_monitoring import (
    TrainingMonitor,
    compare_training_runs,
    compare_training_runs_simplified,
    plot_convergence_comparison,
    print_comparison_table,
)
from util.eval import (
    GenerationQualityMetrics,
    compare_generation_quality,
    compare_generation_quality_simplified,
    print_comprehensive_table,
    compare_convergence,
    compare_convergence_simplified,
    print_convergence_table,
    save_all_metrics_summary,
)

from functional_fm_ot import FFMModelOT
from diffusion import DiffusionModel
from models.fno import FNO

# =============================================================================
# Configuration
# =============================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dpath', type=str, default='../data/ns.mat')
parser.add_argument('--spath', type=str, default='../outputs/navier_stokes_ot/')
parser.add_argument('--ntr', type=int, default=20000, help='Training samples')
parser.add_argument('--subsample_time', type=int, default=5, help='Time subsampling')
parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
parser.add_argument('--n_seeds', type=int, default=3, help='Number of random seeds')
parser.add_argument('--bs', help='Batch size', type=int, default=512)
parser.add_argument('--load-only', action='store_true', help='Load saved results instead of training')
parser.add_argument('--baselines-only', action='store_true', help='Run only DDPM and NCSN baselines')
parser.add_argument('--config', type=str, default=None, 
                    help='Run only this config (for parallelization). Use --list-configs to see available.')
parser.add_argument('--seed', type=int, default=None,
                    help='Run only this seed (use with --config for single run)')
parser.add_argument('--list-configs', action='store_true', help='List available configurations and exit')
args, _ = parser.parse_known_args()

# Load data
print("\nLoading Navier-Stokes data...")
data = load_navier_stokes(args.dpath, shuffle=True, subsample_time=args.subsample_time)
print(f"Data shape: {data.shape}")

# Split data
n_total = data.shape[0]
ntr = min(args.ntr, int(0.8 * n_total))
train_data = data[:ntr]
ground_truth = train_data.squeeze(1).clone()  # Remove channel dim for metrics
spatial_dims = train_data.shape[2:]  # (64, 64)

print(f"Training samples: {train_data.shape[0]}")
print(f"Spatial dimensions: {spatial_dims}")

batch_size = args.bs

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Model hyperparameters (2D FNO)
modes = 16
hch = 32
pch = 64
tscale = 1000.0

# GP hyperparameters
kernel_length = 0.001
kernel_variance = 1.0

# Training params
epochs = args.epochs
lr = 1e-3

# FFM params
sigma_min = 1e-4

# Random seeds
n_seeds = args.n_seeds
random_seeds = [2**i for i in range(n_seeds)]

# Number of samples to generate
n_gen_samples = 100

# Output directory
spath = Path(args.spath)
spath.mkdir(parents=True, exist_ok=True)

# Save ground truth (subset for metrics)
torch.save(ground_truth[:1000], spath / 'ground_truth.pt')

# =============================================================================
# OT Configurations (2D - no signature kernel)
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
        "ot_reg": 1e-3,
        "ot_coupling": "barycentric",
    },
    
    # =========================================================================
    # Euclidean OT
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
    
    # =========================================================================
    # RBF Kernel OT
    # =========================================================================
    "rbf_exact": {
        "use_ot": True,
        "ot_method": "exact",
        "ot_kernel": "rbf",
        "ot_coupling": "sample",
        "ot_kernel_params": {"sigma": 10.0},  # Larger sigma for 2D
    },
    "rbf_sinkhorn_reg0.1": {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.1,
        "ot_kernel": "rbf",
        "ot_coupling": "sample",
        "ot_kernel_params": {"sigma": 10.0},
    },
    "rbf_sinkhorn_reg0.5": {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.5,
        "ot_kernel": "rbf",
        "ot_coupling": "sample",
        "ot_kernel_params": {"sigma": 10.0},
    },
    "rbf_sinkhorn_reg1.0": {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 1.0,
        "ot_kernel": "rbf",
        "ot_coupling": "sample",
        "ot_kernel_params": {"sigma": 10.0},
    },
    "rbf_sinkhorn_barycentric": {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.1,
        "ot_kernel": "rbf",
        "ot_coupling": "barycentric",
        "ot_kernel_params": {"sigma": 10.0},
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
    """Create 2D FNO model."""
    return FNO(
        modes,
        vis_channels=1,
        hidden_channels=hch,
        proj_channels=pch,
        x_dim=2,
        t_scaling=tscale
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
    scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
    
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
    samples = ffm.sample(list(spatial_dims), n_samples=n_gen_samples, n_channels=1)
    samples = samples.cpu().squeeze(1)  # Remove channel dim
    
    # Save sample visualizations
    plot_samples(samples[:16].unsqueeze(1), save_dir / 'samples_viz.pdf')
    
    print(f"    Computing quality statistics...")
    # For 2D data, flatten for statistics
    gt_flat = ground_truth[:1000].reshape(1000, -1)
    samples_flat = samples.reshape(n_gen_samples, -1)
    
    quality_metrics = GenerationQualityMetrics(
        config_name=config_name,
        ot_kernel=config.get("ot_kernel", "") or "",
        ot_method=config.get("ot_method", "") or "",
        ot_coupling=config.get("ot_coupling", "") or "",
        use_ot=config.get("use_ot", False),
    )
    
    # Compute basic statistics
    quality_metrics.mean_mse = float(((gt_flat.mean(0) - samples_flat.mean(0))**2).mean())
    quality_metrics.variance_mse = float(((gt_flat.var(0) - samples_flat.var(0))**2).mean())
    
    if training_metrics is not None:
        if training_metrics.train_losses:
            quality_metrics.final_train_loss = training_metrics.train_losses[-1]
            quality_metrics.set_convergence_metrics(training_metrics.train_losses)
        if training_metrics.epoch_times:
            quality_metrics.total_train_time = sum(training_metrics.epoch_times)
    
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
    """Train a baseline (DDPM/NCSN) configuration for 2D data."""
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
    samples = diffusion.sample(list(spatial_dims), n_samples=n_gen_samples, n_channels=1)
    samples = samples.cpu().squeeze(1)  # Remove channel dim
    
    # Compute quality metrics on flattened data
    gt_flat = ground_truth[:1000].reshape(1000, -1)
    samples_flat = samples.reshape(n_gen_samples, -1)
    
    quality_metrics = GenerationQualityMetrics(
        config_name=config_name, ot_kernel="", ot_method="", ot_coupling="", use_ot=False,
    )
    quality_metrics.compute_from_samples(gt_flat, samples_flat)
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
        
        config_results = {}
        
        for seed in random_seeds:
            config_dir = spath / config_name / f"seed_{seed}"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            result = train_single_config(
                config_name=config_name,
                config=config,
                seed=seed,
                train_loader=train_loader,
                ground_truth=ground_truth,
                save_dir=config_dir,
            )
            
            config_results[seed] = result
        
        all_results[config_name] = config_results
    
    return all_results


def load_all_results(include_baselines: bool = True) -> Dict[str, Dict[int, Dict]]:
    """Load all results from saved files instead of training."""
    from util.ot_monitoring import TrainingMetrics
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
        
        metrics_to_average = [
            'mean_mse', 'variance_mse', 'final_train_loss',
            'total_train_time', 'convergence_rate', 'final_stability',
        ]
        
        for attr in metrics_to_average:
            values = [getattr(q, attr, None) for q in quality_list if getattr(q, attr, None) is not None]
            if values:
                setattr(mean_quality, attr, np.mean(values))
        
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

def plot_samples_grid(aggregated: Dict, ground_truth: torch.Tensor, save_path: Path):
    """Create sample comparison grid."""
    n_plot = 4
    
    samples_dir = save_path / 'sample_comparisons'
    samples_dir.mkdir(exist_ok=True)
    
    for config_name, data in aggregated.items():
        fig, axes = plt.subplots(2, n_plot, figsize=(12, 6))
        
        samples = data['samples']
        
        # Top row: ground truth
        for i in range(n_plot):
            axes[0, i].imshow(ground_truth[i].numpy(), cmap='RdBu_r')
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            if i == 0:
                axes[0, i].set_ylabel('Ground Truth', fontsize=10)
        
        # Bottom row: generated
        for i in range(n_plot):
            axes[1, i].imshow(samples[i].numpy(), cmap='RdBu_r')
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
            if i == 0:
                axes[1, i].set_ylabel(config_name, fontsize=10)
        
        plt.suptitle(f'Navier-Stokes: {config_name}', fontsize=12)
        plt.tight_layout()
        
        safe_name = config_name.replace('/', '_').replace('\\', '_')
        plt.savefig(samples_dir / f'{safe_name}.pdf', dpi=150, bbox_inches='tight')
        plt.savefig(samples_dir / f'{safe_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {len(aggregated)} sample comparison plots to {samples_dir}")


def create_summary_report(aggregated: Dict, save_path: Path):
    """Create summary JSON."""
    summary = {
        'dataset': 'Navier-Stokes',
        'n_samples': ntr,
        'spatial_dims': list(spatial_dims),
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
            },
            'training_metrics': {
                'final_train_loss': float(q.final_train_loss) if q.final_train_loss is not None else None,
                'total_train_time': float(q.total_train_time) if q.total_train_time is not None else None,
            }
        }
    
    with open(save_path / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


# =============================================================================
# Single Config Runner (for parallelization)
# =============================================================================

def run_single_config_with_seeds(config_name: str, seeds_to_run: list = None):
    """Run a single configuration with specified seeds."""
    # Determine which config dict to use
    if config_name in OT_CONFIGS:
        config = OT_CONFIGS[config_name]
        is_baseline = False
    elif config_name in BASELINE_CONFIGS:
        config = BASELINE_CONFIGS[config_name]
        is_baseline = True
    else:
        print(f"ERROR: Unknown config '{config_name}'")
        print(f"Available OT configs: {list(OT_CONFIGS.keys())}")
        print(f"Available baseline configs: {list(BASELINE_CONFIGS.keys())}")
        sys.exit(1)
    
    seeds = seeds_to_run if seeds_to_run else random_seeds
    
    print(f"\n{'='*60}")
    print(f"Configuration: {config_name}")
    print(f"Seeds: {seeds}")
    print(f"{'='*60}")
    
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
                config_name, config, seed, train_loader, ground_truth, config_dir
            )
        config_results[seed] = result
    
    return {config_name: config_results}


def list_all_configs():
    """Print all available configurations."""
    print("\nAvailable OT configurations:")
    print("-" * 40)
    for name in OT_CONFIGS.keys():
        print(f"  {name}")
    
    print("\nAvailable baseline configurations:")
    print("-" * 40)
    for name in BASELINE_CONFIGS.keys():
        print(f"  {name}")
    
    print(f"\nTotal: {len(OT_CONFIGS)} OT configs + {len(BASELINE_CONFIGS)} baselines")
    print(f"Seeds: {random_seeds}")
    print("\nExample usage for parallel runs:")
    print(f"  python {sys.argv[0]} --config independent --seed 1")
    print(f"  python {sys.argv[0]} --config euclidean_exact --seed 2")
    print(f"  python {sys.argv[0]} --config DDPM --seed 1")
    print("\nAfter all parallel runs complete:")
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
    print(f"OT-FFM Experiments on Navier-Stokes")
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
        baseline_results = run_baseline_experiments()
        ot_results = load_all_results(include_baselines=False)
        all_results = {**ot_results, **baseline_results}
    elif args.load_only:
        all_results = load_all_results(include_baselines=True)
        if not all_results:
            print("ERROR: No saved results found. Run without --load-only first.")
            sys.exit(1)
    else:
        all_results = run_all_experiments()
    
    print("\n" + "="*60)
    print("Aggregating results...")
    print("="*60)
    aggregated = aggregate_results(all_results)
    
    # Filter baselines for OT-specific plots
    baseline_names = set(BASELINE_CONFIGS.keys())
    ot_aggregated = {k: v for k, v in aggregated.items() if k not in baseline_names}
    
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)
    
    plot_samples_grid(aggregated, ground_truth, spath)
    
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
    
    # Convergence analysis (OT configs only)
    losses_dict = {
        name: data['training_metrics'].train_losses 
        for name, data in ot_aggregated.items() 
        if data['training_metrics'] is not None
    }
    if losses_dict:
        figs, conv_metrics = compare_convergence(losses_dict, save_path=spath / 'convergence_metrics')
        for fig in figs:
            plt.close(fig)
        
        # Simplified convergence (best per kernel type)
        quality_dict = {name: data['mean_quality'].mean_mse for name, data in ot_aggregated.items()
                       if data['mean_quality'] is not None and data['mean_quality'].mean_mse is not None}
        fig_simple = compare_convergence_simplified(losses_dict, quality_metrics_dict=quality_dict, save_path=spath / 'convergence_metrics')
        if fig_simple:
            plt.close(fig_simple)
        
        print_convergence_table(conv_metrics)
    
    # Quality comparison
    quality_metrics_list = [data['mean_quality'] for data in aggregated.values()]
    # Top-5 ranked comparison
    compare_generation_quality(
        quality_metrics_list, 
        save_path=spath / 'quality_comparison_top5.pdf',
        include_spectrum=False,
        include_seasonal=False,
        top_k=5,
    )
    # Best per kernel category
    print("\nGenerating simplified quality comparison (best per kernel type)...")
    compare_generation_quality_simplified(quality_metrics_list, save_path=spath / 'quality_comparison.pdf')
    
    print_comprehensive_table(quality_metrics_list)
    save_all_metrics_summary(quality_metrics_list, spath / 'comprehensive_metrics.json')
    
    create_summary_report(aggregated, spath)
    
    plt.close('all')
    
    print("\n" + "="*60)
    print(f"Complete! Results: {spath}")
    print("="*60)

