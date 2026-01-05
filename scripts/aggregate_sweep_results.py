#!/usr/bin/env python3
"""
Aggregate results from multiple experiment directories.

This script:
1. Loads results from both original experiments and sweep experiments
2. Finds the best configuration for each metric across all experiments
3. Generates updated comparison tables and plots
4. Optionally copies/links sweep results into main directories for unified loading

Usage:
    python aggregate_sweep_results.py --dataset econ1_population
    python aggregate_sweep_results.py --dataset rBergomi
    python aggregate_sweep_results.py --dataset stochastic_kdv
    python aggregate_sweep_results.py --dataset stochastic_ns
    python aggregate_sweep_results.py --all  # Process all datasets
"""

import sys
sys.path.append('../')

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import shutil


# =============================================================================
# Configuration: Map datasets to their directories
# =============================================================================

DATASET_CONFIG = {
    "econ1_population": {
        "main_dir": "econ_ot_comprehensive",
        "sweep_dirs": ["econ_signature_sweep"],  # Phase 2 results go to main_dir directly
        "subdirectory": "econ1_population",  # Results are in subdirectory
        "key_metrics": ["autocorrelation_mse", "mean_mse", "variance_mse", "spectrum_mse", "spectrum_mse_log"],
        "lower_is_better": {"autocorrelation_mse", "mean_mse", "variance_mse", "spectrum_mse", 
                           "spectrum_mse_log", "skewness_mse", "kurtosis_mse", "density_mse"},
    },
    "rBergomi": {
        "main_dir": "rBergomi_ot_H0p10",  # Fixed: was rBergomi_ot_comprehensive
        "sweep_dirs": ["rbergomi_signature_sweep"],  # Phase 2 results go to main_dir directly
        "subdirectory": None,
        "key_metrics": ["autocorrelation_mse", "mean_mse", "kurtosis_mse", "spectrum_mse", "spectrum_mse_log"],
        "lower_is_better": {"autocorrelation_mse", "mean_mse", "variance_mse", "spectrum_mse",
                           "spectrum_mse_log", "skewness_mse", "kurtosis_mse", "density_mse"},
    },
    "stochastic_kdv": {
        "main_dir": "stochastic_kdv_ot",
        "sweep_dirs": [],  # Sweep results go to main_dir directly
        "subdirectory": None,
        "key_metrics": ["mean_mse", "variance_mse", "spectrum_mse", "spectrum_mse_log", "autocorrelation_mse"],
        "lower_is_better": {"autocorrelation_mse", "mean_mse", "variance_mse", "spectrum_mse",
                           "spectrum_mse_log", "skewness_mse", "kurtosis_mse", "density_mse"},
    },
    "stochastic_ns": {
        "main_dir": "stochastic_ns_ot",
        "sweep_dirs": [],  # Sweep results go to main_dir directly
        "subdirectory": None,
        "key_metrics": ["spectrum_mse", "spectrum_mse_log", "mean_mse", "variance_mse"],
        "lower_is_better": {"autocorrelation_mse", "mean_mse", "variance_mse", "spectrum_mse",
                           "spectrum_mse_log", "skewness_mse", "kurtosis_mse", "density_mse"},
    },
    "kdv": {
        "main_dir": "kdv_ot",
        "sweep_dirs": [],  # Sweep results go to main_dir directly
        "subdirectory": None,
        "key_metrics": ["spectrum_mse", "spectrum_mse_log", "mean_mse", "variance_mse", "autocorrelation_mse"],
        "lower_is_better": {"autocorrelation_mse", "mean_mse", "variance_mse", "spectrum_mse",
                           "spectrum_mse_log", "skewness_mse", "kurtosis_mse", "density_mse"},
    },
    "navier_stokes": {
        "main_dir": "navier_stokes_ot",
        "sweep_dirs": [],  # Sweep results go to main_dir directly
        "subdirectory": None,
        "key_metrics": ["spectrum_mse", "spectrum_mse_log", "mean_mse", "variance_mse"],
        "lower_is_better": {"autocorrelation_mse", "mean_mse", "variance_mse", "spectrum_mse",
                           "spectrum_mse_log", "skewness_mse", "kurtosis_mse", "density_mse"},
    },
}

# Baseline models to always include
BASELINES = ["DDPM", "NCSN", "independent"]

# Seeds to look for
SEEDS = [1, 2, 4]


@dataclass
class ConfigResult:
    """Results for a single configuration."""
    config_name: str
    source_dir: str
    metrics: Dict[str, float]
    n_seeds: int = 0


def load_quality_metrics(config_dir: Path, seeds: List[int] = SEEDS) -> Optional[Dict[str, float]]:
    """Load and average quality metrics across seeds."""
    all_metrics = []
    
    for seed in seeds:
        seed_dir = config_dir / f"seed_{seed}"
        metrics_file = seed_dir / "quality_metrics.json"
        
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    all_metrics.append(metrics)
            except (json.JSONDecodeError, IOError):
                continue
    
    if not all_metrics:
        return None
    
    # Average numeric metrics
    averaged = {}
    numeric_keys = [
        'mean_mse', 'variance_mse', 'skewness_mse', 'kurtosis_mse',
        'autocorrelation_mse', 'density_mse', 'spectrum_mse', 'spectrum_mse_log',
        'final_train_loss', 'total_train_time', 'mean_path_length', 'mean_grad_variance',
        'convergence_rate', 'final_stability', 'epochs_to_90pct',
        'seasonal_mse', 'seasonal_amplitude_error', 'seasonal_phase_correlation',
    ]
    
    for key in numeric_keys:
        values = [m.get(key) for m in all_metrics if isinstance(m.get(key), (int, float))]
        if values:
            averaged[key] = float(np.mean(values))
    
    # Keep non-numeric metadata from first result
    for key in ['config_name', 'ot_kernel', 'ot_method', 'ot_coupling', 'use_ot']:
        if key in all_metrics[0]:
            averaged[key] = all_metrics[0][key]
    
    averaged['_n_seeds'] = len(all_metrics)
    
    return averaged


def load_results_from_directory(
    base_dir: Path, 
    subdirectory: Optional[str] = None,
    exclude_configs: Optional[set] = None
) -> Dict[str, ConfigResult]:
    """Load all config results from a directory."""
    
    results = {}
    exclude_configs = exclude_configs or set()
    
    search_dir = base_dir / subdirectory if subdirectory else base_dir
    
    if not search_dir.exists():
        print(f"  Directory not found: {search_dir}")
        return results
    
    # Find all config directories (those containing seed_* subdirectories)
    for item in search_dir.iterdir():
        if not item.is_dir():
            continue
        
        config_name = item.name
        
        if config_name in exclude_configs:
            continue
        
        # Check if this looks like a config directory (has seed_* subdirs)
        seed_dirs = list(item.glob("seed_*"))
        if not seed_dirs:
            continue
        
        metrics = load_quality_metrics(item)
        if metrics:
            results[config_name] = ConfigResult(
                config_name=config_name,
                source_dir=str(search_dir),
                metrics=metrics,
                n_seeds=metrics.get('_n_seeds', 0)
            )
    
    return results


def find_best_for_metric(
    all_results: Dict[str, ConfigResult],
    metric: str,
    lower_is_better: bool = True,
    exclude_baselines: bool = False,
    kffm_only: bool = False
) -> Tuple[Optional[str], Optional[float]]:
    """Find the best configuration for a given metric.
    
    Parameters
    ----------
    all_results : Dict[str, ConfigResult]
        All configuration results
    metric : str
        Metric to compare
    lower_is_better : bool
        If True, lower values are better
    exclude_baselines : bool
        If True, exclude DDPM/NCSN/independent from comparison
    kffm_only : bool
        If True, only include configs with use_ot=True and exclude gaussian OT
    """
    
    best_config = None
    best_value = None
    
    for config_name, result in all_results.items():
        if exclude_baselines and config_name in BASELINES:
            continue
        
        # For k-FFM only mode, check use_ot=True and exclude gaussian
        if kffm_only:
            use_ot = result.metrics.get('use_ot', False)
            ot_method = result.metrics.get('ot_method', '')
            if use_ot != True or ot_method == 'gaussian':
                continue
        
        value = result.metrics.get(metric)
        if value is None:
            continue
        
        if best_value is None:
            best_value = value
            best_config = config_name
        elif lower_is_better and value < best_value:
            best_value = value
            best_config = config_name
        elif not lower_is_better and value > best_value:
            best_value = value
            best_config = config_name
    
    return best_config, best_value


def print_comparison_table(
    all_results: Dict[str, ConfigResult],
    key_metrics: List[str],
    lower_is_better: set,
    dataset_name: str
):
    """Print a comparison table showing best configs for each metric."""
    
    print(f"\n{'='*80}")
    print(f"  Results Summary: {dataset_name}")
    print(f"{'='*80}")
    print(f"\nLoaded {len(all_results)} configurations\n")
    
    # Find best for each metric
    print("Best Configuration per Metric:")
    print("-" * 80)
    print(f"{'Metric':<25} {'Best Config':<35} {'Value':<15} {'Source'}")
    print("-" * 80)
    
    for metric in key_metrics:
        is_lower_better = metric in lower_is_better
        best_config, best_value = find_best_for_metric(all_results, metric, is_lower_better)
        
        if best_config and best_value is not None:
            source = all_results[best_config].source_dir.split('/')[-1]
            print(f"{metric:<25} {best_config:<35} {best_value:<15.2e} {source}")
        else:
            print(f"{metric:<25} {'N/A':<35} {'N/A':<15}")
    
    print("-" * 80)
    
    # Compare baselines vs best k-FFM
    print("\nBaseline vs Best k-FFM Comparison:")
    print("-" * 80)
    
    for metric in key_metrics:
        is_lower_better = metric in lower_is_better
        
        # Best k-FFM (only configs with use_ot=True, excluding gaussian)
        best_kffm, best_kffm_val = find_best_for_metric(
            all_results, metric, is_lower_better, kffm_only=True
        )
        
        # Best baseline
        baseline_results = {k: v for k, v in all_results.items() if k in BASELINES}
        best_baseline, best_baseline_val = find_best_for_metric(
            baseline_results, metric, is_lower_better
        )
        
        if best_kffm_val is not None and best_baseline_val is not None:
            if is_lower_better:
                ratio = best_baseline_val / best_kffm_val if best_kffm_val > 0 else float('inf')
                winner = "k-FFM ✓" if best_kffm_val < best_baseline_val else "Baseline ✓"
            else:
                ratio = best_kffm_val / best_baseline_val if best_baseline_val > 0 else float('inf')
                winner = "k-FFM ✓" if best_kffm_val > best_baseline_val else "Baseline ✓"
            
            print(f"{metric:<25}")
            print(f"  Baseline ({best_baseline}): {best_baseline_val:.2e}")
            print(f"  k-FFM ({best_kffm}): {best_kffm_val:.2e}")
            print(f"  → {winner}")
            print()


def generate_summary_json(
    all_results: Dict[str, ConfigResult],
    key_metrics: List[str],
    lower_is_better: set,
    output_path: Path
):
    """Generate a JSON summary of aggregated results."""
    
    summary = {
        "total_configs": len(all_results),
        "configs": {},
        "best_per_metric": {},
    }
    
    # All config metrics
    for config_name, result in all_results.items():
        summary["configs"][config_name] = {
            "source_dir": result.source_dir,
            "n_seeds": result.n_seeds,
            "metrics": result.metrics,
        }
    
    # Best per metric
    for metric in key_metrics:
        is_lower_better = metric in lower_is_better
        best_config, best_value = find_best_for_metric(all_results, metric, is_lower_better)
        # Best k-FFM: only configs with use_ot=True, excluding gaussian
        best_kffm, best_kffm_val = find_best_for_metric(
            all_results, metric, is_lower_better, kffm_only=True
        )
        
        summary["best_per_metric"][metric] = {
            "overall_best": {"config": best_config, "value": best_value},
            "best_kffm": {"config": best_kffm, "value": best_kffm_val},
        }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved aggregated summary to: {output_path}")


def copy_sweep_to_main(
    sweep_dir: Path,
    main_dir: Path,
    subdirectory: Optional[str] = None,
    dry_run: bool = True
):
    """Copy/link sweep results into main directory for unified loading."""
    
    target_dir = main_dir / subdirectory if subdirectory else main_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    
    if not sweep_dir.exists():
        print(f"  Sweep directory not found: {sweep_dir}")
        return
    
    # Handle case where sweep results are in a subdirectory
    if subdirectory and (sweep_dir / subdirectory).exists():
        sweep_search_dir = sweep_dir / subdirectory
    else:
        sweep_search_dir = sweep_dir
    
    copied = 0
    for item in sweep_search_dir.iterdir():
        if not item.is_dir():
            continue
        
        # Check if it's a config directory
        seed_dirs = list(item.glob("seed_*"))
        if not seed_dirs:
            continue
        
        config_name = item.name
        target_config_dir = target_dir / config_name
        
        if target_config_dir.exists():
            print(f"  Skipping {config_name} (already exists in main)")
            continue
        
        if dry_run:
            print(f"  Would copy: {item} -> {target_config_dir}")
        else:
            shutil.copytree(item, target_config_dir)
            print(f"  Copied: {config_name}")
        
        copied += 1
    
    action = "Would copy" if dry_run else "Copied"
    print(f"\n{action} {copied} configurations from sweep to main directory")


def process_dataset(dataset_name: str, outputs_dir: Path, merge: bool = False, dry_run: bool = True):
    """Process a single dataset: load, aggregate, and report results."""
    
    if dataset_name not in DATASET_CONFIG:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available: {list(DATASET_CONFIG.keys())}")
        return
    
    config = DATASET_CONFIG[dataset_name]
    main_dir = outputs_dir / config["main_dir"]
    subdirectory = config["subdirectory"]
    
    print(f"\n{'#'*80}")
    print(f"  Processing: {dataset_name}")
    print(f"{'#'*80}")
    print(f"Main directory: {main_dir}")
    print(f"Subdirectory: {subdirectory or 'None'}")
    print(f"Sweep directories: {config['sweep_dirs']}")
    
    # Load results from main directory
    print(f"\nLoading from main directory...")
    all_results = load_results_from_directory(main_dir, subdirectory)
    print(f"  Found {len(all_results)} configurations")
    
    # Load results from sweep directories
    for sweep_name in config["sweep_dirs"]:
        sweep_dir = outputs_dir / sweep_name
        print(f"\nLoading from sweep directory: {sweep_dir}")
        
        # For sweep directories, check if results are in subdirectory
        sweep_results = load_results_from_directory(sweep_dir, subdirectory)
        if not sweep_results:
            # Try without subdirectory
            sweep_results = load_results_from_directory(sweep_dir, None)
        
        print(f"  Found {len(sweep_results)} configurations")
        
        # Merge, preferring sweep results for duplicates
        for config_name, result in sweep_results.items():
            if config_name in all_results:
                print(f"  Updating {config_name} from sweep (was in main)")
            all_results[config_name] = result
    
    if not all_results:
        print("No results found!")
        return
    
    # Print comparison table
    print_comparison_table(
        all_results,
        config["key_metrics"],
        config["lower_is_better"],
        dataset_name
    )
    
    # Save aggregated summary
    summary_path = main_dir / f"aggregated_results_{dataset_name}.json"
    generate_summary_json(all_results, config["key_metrics"], config["lower_is_better"], summary_path)
    
    # Optionally copy sweep results to main directory
    if merge:
        print("\n" + "="*40)
        print("Merging sweep results into main directory")
        print("="*40)
        for sweep_name in config["sweep_dirs"]:
            sweep_dir = outputs_dir / sweep_name
            copy_sweep_to_main(sweep_dir, main_dir, subdirectory, dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(description="Aggregate results from experiments and sweeps")
    parser.add_argument('--dataset', type=str, default=None,
                        choices=list(DATASET_CONFIG.keys()),
                        help='Dataset to process')
    parser.add_argument('--all', action='store_true',
                        help='Process all datasets')
    parser.add_argument('--outputs-dir', type=str, default='../outputs',
                        help='Base outputs directory')
    parser.add_argument('--merge', action='store_true',
                        help='Copy sweep results into main directories')
    parser.add_argument('--no-dry-run', action='store_true',
                        help='Actually copy files (default is dry-run)')
    
    args = parser.parse_args()
    
    outputs_dir = Path(args.outputs_dir)
    
    if args.all:
        for dataset in DATASET_CONFIG.keys():
            process_dataset(dataset, outputs_dir, args.merge, dry_run=not args.no_dry_run)
    elif args.dataset:
        process_dataset(args.dataset, outputs_dir, args.merge, dry_run=not args.no_dry_run)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python aggregate_sweep_results.py --all")
        print("  python aggregate_sweep_results.py --dataset econ1_population")
        print("  python aggregate_sweep_results.py --dataset econ1_population --merge --no-dry-run")


if __name__ == "__main__":
    main()

