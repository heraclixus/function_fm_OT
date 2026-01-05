"""
Comprehensive OT-FFM Experiments on KdV (Korteweg-de Vries).

This script runs experiments with multiple OT configurations on deterministic
KdV equation solutions comparing:
- Different kernel types (euclidean, rbf, signature)
- Different OT methods (exact, sinkhorn, gaussian)
- Different coupling strategies (sample, barycentric)

**Important Note**: This dataset contains only a SINGLE trajectory with 201 time
steps. Each time step is a spatial snapshot of size 512. This means we only have
201 samples total, which is very limited for training generative models.

Data shape: (N=512, T=201) - a single trajectory

Usage:
    python kdv_ot.py
    python kdv_ot.py --epochs 200 --n_seeds 3
    python kdv_ot.py --config-file ../configs/kdv_sweep.yaml --config euclidean_reg0.05 --seed 1
    python kdv_ot.py --list-configs
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

from util.util import load_kdv
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
# Configuration
# =============================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dpath', type=str, default='../data/KdV.mat')
parser.add_argument('--spath', type=str, default='../outputs/kdv_ot/')
parser.add_argument('--ntr', type=int, default=160, help='Training samples (max ~160)')
parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
parser.add_argument('--n_seeds', type=int, default=3, help='Number of random seeds')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--batch_size_sig', type=int, default=16, help='Batch size for signature kernel')
parser.add_argument('--load-only', action='store_true', help='Load saved results instead of training')
parser.add_argument('--baselines-only', action='store_true', help='Run only DDPM and NCSN baselines')
parser.add_argument('--config-file', type=str, default=None,
                    help='Path to a YAML or JSON file containing OT configurations')
parser.add_argument('--config', type=str, default=None,
                    help='Run only this specific configuration (for sweep parallelization)')
parser.add_argument('--seed', type=int, default=None,
                    help='Run only this specific seed (for sweep parallelization)')
parser.add_argument('--list-configs', action='store_true',
                    help='List all available configs and exit')
parser.add_argument('--modes', type=int, default=None,
                    help='Override FNO modes (default: 64)')
parser.add_argument('--kernel-length', type=float, default=None,
                    help='Override GP kernel_length (default: 0.01)')
parser.add_argument('--kernel-variance', type=float, default=None,
                    help='Override GP kernel_variance (default: 0.1)')
args, _ = parser.parse_known_args()

# Load data
print("\nLoading KdV data...")
print("WARNING: KdV dataset is a SINGLE trajectory with only 201 time samples!")
print("         This is very limited data for training generative models.")
data = load_kdv(args.dpath, mode='snapshot')
print(f"Data shape: {data.shape}")  # Should be (201, 1, 512)

# Shuffle and split data
n_total = data.shape[0]  # 201
idx = torch.randperm(n_total)
data = data[idx]

ntr = min(args.ntr, int(0.8 * n_total))
train_data = data[:ntr]
ground_truth = train_data.squeeze(1).clone()  # Remove channel dim for metrics
n_x = train_data.shape[-1]  # 512

print(f"Training samples: {train_data.shape[0]}")
print(f"Spatial dimension: {n_x}")

batch_size = args.batch_size
batch_size_signature = args.batch_size_sig

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
train_loader_signature = DataLoader(train_data, batch_size=batch_size_signature, shuffle=True)

# Model hyperparameters (larger for 512 spatial points)
modes = 64
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
n_gen_samples = 200

# Output directory
spath = Path(args.spath)
spath.mkdir(parents=True, exist_ok=True)

# Save ground truth
torch.save(ground_truth, spath / 'ground_truth.pt')

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
        "ot_kernel_params": {"sigma": 10.0},  # Larger sigma for 512 points
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
    
    # =========================================================================
    # Signature Kernel OT - for capturing path structure
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
        },
    },
}

# =============================================================================
# Config loading functions
# =============================================================================

def load_configs_from_file(file_path: Path) -> Dict:
    """Load configurations from a YAML or JSON file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    
    if file_path.suffix in ['.yaml', '.yml']:
        import yaml
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    elif file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {file_path.suffix}. Use .yaml, .yml, or .json")


def get_all_configs() -> Dict:
    """Get all OT configurations, merging built-in with external if provided."""
    configs = dict(OT_CONFIGS)
    if args.config_file:
        external_configs = load_configs_from_file(Path(args.config_file))
        configs.update(external_configs)
    return configs


def get_fno_modes(config: dict) -> int:
    """Get FNO modes: prioritize command-line args > config > default."""
    if args.modes:
        return args.modes
    return config.get("fno_modes", modes)


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

def create_model(device, fno_modes: int = None):
    """Create FNO model."""
    m = fno_modes if fno_modes else modes
    return FNO(
        m,
        vis_channels=1,
        hidden_channels=width,
        proj_channels=mlp_width,
        x_dim=1,
        t_scaling=1000
    ).to(device)


def build_ffm_kwargs(config: dict) -> dict:
    """Build kwargs for FFMModelOT from config dict."""
    # GP prior: prioritize command-line args > config > defaults
    kl = args.kernel_length if args.kernel_length else config.get("gp_kernel_length", kernel_length)
    kv = args.kernel_variance if args.kernel_variance else config.get("gp_kernel_variance", kernel_variance)
    
    kwargs = {
        "kernel_length": kl,
        "kernel_variance": kv,
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
    
    fno_modes = get_fno_modes(config)
    model = create_model(device, fno_modes=fno_modes)
    ffm_kwargs = build_ffm_kwargs(config)
    ffm = FFMModelOT(model, **ffm_kwargs)
    
    monitor = TrainingMonitor(
        method_name=config_name,
        use_ot=config.get("use_ot", False),
        ot_kernel=config.get("ot_kernel", "") or "",
        track_batch_metrics=False,
    )
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100)
    
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100)
    
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


def run_single_config_with_seeds(
    config_name: str,
    seeds_to_run: list,
    config: dict,
) -> Dict[str, Dict[int, Dict]]:
    """Run a single configuration with specified seeds."""
    all_results = {}
    
    print(f"\n{'='*60}")
    print(f"Configuration: {config_name}")
    print(f"{'='*60}")
    
    loader = train_loader_signature if config.get("ot_kernel") == "signature" else train_loader
    
    config_results = {}
    for seed in seeds_to_run:
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


def run_all_experiments() -> Dict[str, Dict[int, Dict]]:
    """Run all configurations with all seeds."""
    
    all_results = {}
    all_configs = get_all_configs()
    
    for config_name, config in all_configs.items():
        print(f"\n{'='*60}")
        print(f"Configuration: {config_name}")
        print(f"{'='*60}")
        
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
    from util.ot_monitoring import TrainingMetrics
    print("Loading saved results from", spath)
    
    all_results = {}
    
    all_configs = get_all_configs()
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
    """Create comparison plots."""
    x_grid = torch.linspace(0, 1, n_x)
    n_plot = 30
    
    samples_dir = save_path / 'sample_comparisons'
    samples_dir.mkdir(exist_ok=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(aggregated)))
    
    for idx, (config_name, data) in enumerate(aggregated.items()):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        samples = data['samples']
        
        ax = axes[0]
        for i in range(min(n_plot, ground_truth.shape[0])):
            ax.plot(x_grid.numpy(), ground_truth[i].numpy(), color='gray', alpha=0.4, linewidth=0.8)
        ax.set_title('Ground Truth\n(KdV)', fontsize=11, fontweight='bold')
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('u(x)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        for i in range(min(n_plot, samples.shape[0])):
            ax.plot(x_grid.numpy(), samples[i].numpy(), color=colors[idx], alpha=0.4, linewidth=0.8)
        
        title = config_name.replace('_', ' ').title()
        ax.set_title(f'{title}\n({samples.shape[0]} samples)', fontsize=11, fontweight='bold')
        ax.set_xlabel('x', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
        y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
        axes[0].set_ylim(y_min, y_max)
        axes[1].set_ylim(y_min, y_max)
        
        plt.suptitle(f'KdV: Ground Truth vs {config_name}', fontsize=12, y=1.02)
        plt.tight_layout()
        
        safe_name = config_name.replace('/', '_').replace('\\', '_')
        plt.savefig(samples_dir / f'{safe_name}.pdf', dpi=150, bbox_inches='tight')
        plt.savefig(samples_dir / f'{safe_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {len(aggregated)} sample comparison plots to {samples_dir}")


def create_summary_report(aggregated: Dict, save_path: Path):
    """Create summary JSON."""
    summary = {
        'dataset': 'KdV (Korteweg-de Vries)',
        'n_samples': ntr,
        'n_x': n_x,
        'epochs': epochs,
        'batch_size': batch_size,
        'n_seeds': n_seeds,
        'note': 'Very limited data (201 samples from single trajectory)',
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
                'density_mse': float(q.density_mse) if q.density_mse is not None else None,
            },
            'training_metrics': {
                'final_train_loss': float(q.final_train_loss) if q.final_train_loss is not None else None,
                'total_train_time': float(q.total_train_time) if q.total_train_time is not None else None,
            }
        }
    
    with open(save_path / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Handle --list-configs
    if args.list_configs:
        all_configs = get_all_configs()
        print("Available configurations:")
        for name in sorted(all_configs.keys()):
            cfg = all_configs[name]
            ot_type = "OT" if cfg.get("use_ot", False) else "no-OT"
            kernel = cfg.get("ot_kernel", "") or ""
            print(f"  {name}: {ot_type}, kernel={kernel}")
        sys.exit(0)
    
    all_configs = get_all_configs()
    
    print("="*60)
    print(f"OT-FFM Experiments on KdV")
    print(f"WARNING: Very limited data (only ~160 training samples)")
    print(f"Configs: {len(all_configs)}, Seeds: {n_seeds}")
    print(f"Output: {spath}")
    if args.config:
        print(f"MODE: Single config ({args.config}, seed={args.seed if args.seed else 'all'})")
    elif args.baselines_only:
        print("MODE: Baselines-only (running DDPM and NCSN only)")
    elif args.load_only:
        print("MODE: Load-only (loading saved results)")
    else:
        print("MODE: Training (running experiments)")
    print("="*60)
    
    if args.config:
        # Single config mode (for parallelization)
        seeds_to_run = [args.seed] if args.seed else random_seeds
        if args.config not in all_configs:
            print(f"ERROR: Config '{args.config}' not found in built-in or external configs.")
            print(f"Available configs: {list(all_configs.keys())}")
            sys.exit(1)
        
        all_results = run_single_config_with_seeds(args.config, seeds_to_run, all_configs[args.config])
        
        # Aggregate and save for single config
        aggregated = aggregate_results(all_results)
        for cname, data in aggregated.items():
            data['mean_quality'].save(spath / cname / 'aggregated_quality.json')
        print(f"\nCompleted {args.config} with seeds {seeds_to_run}")
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
        quality_dict = {name: data['mean_quality'].mean_mse for name, data in aggregated.items()
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
        include_spectrum=True,
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
    print("\nNote: Due to very limited training data (only ~160 samples from")
    print("      a single trajectory), results may be unreliable.")
    print("="*60)

