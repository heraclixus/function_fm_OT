"""
Post-processing script for OT-FFM experiments.
Loads saved results and regenerates summary files and plots.

Usage:
    python econ_ot_postprocess.py                           # Default: econ_ot_comprehensive
    python econ_ot_postprocess.py ../outputs/AEMET_ot_comprehensive
    python econ_ot_postprocess.py ../outputs/moGP_ot_comprehensive
    python econ_ot_postprocess.py ../outputs/rBergomi_ot_H0p10
    python econ_ot_postprocess.py ../outputs/Heston_ot_kappa1.0
    python econ_ot_postprocess.py ../outputs/expr_genes_ot_comprehensive
"""

import sys
sys.path.append('../')

import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json

from util.ot_monitoring import (
    TrainingMetrics,
    compare_training_runs,
    plot_convergence_comparison,
)
from util.eval import (
    GenerationQualityMetrics,
    compare_generation_quality,
    print_comprehensive_table,
    compare_spectra_1d,
    compare_convergence,
    print_convergence_table,
    save_all_metrics_summary,
)

# Default seeds (will auto-detect from directories)
DEFAULT_SEEDS = [1, 2, 4]  # 2**i for i in range(3)

# Baseline model names (these are not OT configs, used for filtering)
BASELINE_NAMES = {'DDPM', 'NCSN', 'ddpm', 'ncsn'}


def is_baseline_config(config_name: str) -> bool:
    """Check if a config name corresponds to a baseline model."""
    return config_name in BASELINE_NAMES


def filter_ot_configs(aggregated: dict) -> dict:
    """Filter out baseline configs, keeping only OT configs."""
    return {k: v for k, v in aggregated.items() if not is_baseline_config(k)}


def to_python_float(val):
    """Convert numpy types to native Python float for JSON serialization."""
    if val is None:
        return None
    if isinstance(val, (np.floating, np.integer)):
        return val.item()
    if isinstance(val, np.ndarray):
        return val.item() if val.ndim == 0 else float(val.flat[0])
    return float(val)


def auto_detect_configs(base_dir: Path):
    """Auto-detect OT configurations from directory structure."""
    configs = []
    for item in sorted(base_dir.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it has seed subdirectories
            seed_dirs = list(item.glob("seed_*"))
            if seed_dirs:
                configs.append(item.name)
    return configs


def auto_detect_seeds(config_dir: Path):
    """Auto-detect seed values from directory structure."""
    seeds = []
    for item in sorted(config_dir.iterdir()):
        if item.is_dir() and item.name.startswith("seed_"):
            try:
                seed = int(item.name.replace("seed_", ""))
                seeds.append(seed)
            except ValueError:
                pass
    return seeds


def load_dataset_results(dataset_name: str, dataset_dir: Path):
    """Load saved results for a dataset."""
    aggregated = {}
    
    # Auto-detect configs
    ot_configs = auto_detect_configs(dataset_dir)
    if not ot_configs:
        print(f"  No config directories found in {dataset_dir}")
        return aggregated
    
    print(f"  Found {len(ot_configs)} configs: {ot_configs[:5]}{'...' if len(ot_configs) > 5 else ''}")
    
    for config_name in ot_configs:
        config_dir = dataset_dir / config_name
        
        # Auto-detect seeds
        seeds = auto_detect_seeds(config_dir)
        if not seeds:
            print(f"  Skipping {config_name} (no seed dirs)")
            continue
        
        # Load quality metrics from all seeds and average
        quality_list = []
        training_list = []
        first_seed = seeds[0]
        
        for seed in seeds:
            seed_dir = config_dir / f"seed_{seed}"
            
            # Load quality metrics
            qm_path = seed_dir / 'quality_metrics.json'
            if qm_path.exists():
                qm = GenerationQualityMetrics.load(qm_path)
                quality_list.append(qm)
            
            # Load training metrics
            tm_path = seed_dir / 'training_metrics.json'
            if tm_path.exists():
                tm = TrainingMetrics.load(tm_path)
                training_list.append(tm)
        
        if not quality_list:
            print(f"  Skipping {config_name} (no quality metrics)")
            continue
        
        # Average quality metrics
        mean_quality = GenerationQualityMetrics(
            config_name=quality_list[0].config_name if quality_list else config_name,
            use_ot=quality_list[0].use_ot if quality_list else False,
            ot_kernel=quality_list[0].ot_kernel if quality_list else "",
            ot_method=quality_list[0].ot_method if quality_list else "",
            ot_coupling=quality_list[0].ot_coupling if quality_list else "",
        )
        
        metrics_to_average = [
            'mean_mse', 'variance_mse', 'skewness_mse', 'kurtosis_mse', 
            'autocorrelation_mse', 'density_mse', 
            'final_train_loss', 'total_train_time', 'mean_path_length', 'mean_grad_variance',
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
        
        # Load samples from first seed
        first_seed_dir = config_dir / f"seed_{first_seed}"
        samples_path = first_seed_dir / 'samples_original.pt'
        samples_up_path = first_seed_dir / 'samples_upsampled.pt'
        # Also check for just 'samples.pt' (some scripts use this)
        if not samples_path.exists():
            samples_path = first_seed_dir / 'samples.pt'
        
        samples_original = torch.load(samples_path) if samples_path.exists() else None
        samples_upsampled = torch.load(samples_up_path) if samples_up_path.exists() else None
        
        aggregated[config_name] = {
            'mean_quality': mean_quality,
            'training_metrics': training_list[0] if training_list else None,
            'samples_original': samples_original,
            'samples_upsampled': samples_upsampled,
        }
        
        print(f"  Loaded {config_name}: {len(quality_list)} seeds")
    
    return aggregated


def create_full_summary(all_aggregated, save_path):
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


def auto_detect_datasets(base_dir: Path):
    """Auto-detect dataset subdirectories (for multi-dataset experiments like econ)."""
    datasets = []
    for item in sorted(base_dir.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            # Check if this is a dataset dir (has config subdirs with seed subdirs)
            # or if it's directly a config dir
            has_seed_subdirs = any(
                (item / subdir.name).glob("seed_*") 
                for subdir in item.iterdir() if subdir.is_dir()
            )
            if has_seed_subdirs:
                datasets.append(item.name)
    return datasets


def process_single_directory(spath: Path, include_seasonal: bool = False):
    """Process a single experiment directory (may have multiple datasets or just configs)."""
    
    # Check if this directory has dataset subdirectories or direct config directories
    datasets = auto_detect_datasets(spath)
    
    if datasets:
        # Multi-dataset structure (e.g., econ with econ1, econ2, econ3)
        print(f"Found {len(datasets)} datasets: {datasets}")
        all_aggregated = {}
        
        for dataset_name in datasets:
            print(f"\n--- Loading {dataset_name} ---")
            dataset_dir = spath / dataset_name
            
            # Load ground truth
            gt_path = dataset_dir / 'ground_truth.pt'
            ground_truth = torch.load(gt_path) if gt_path.exists() else None
            
            # Load aggregated results
            aggregated = load_dataset_results(dataset_name, dataset_dir)
            all_aggregated[dataset_name] = aggregated
            
            if not aggregated:
                print(f"  No results found for {dataset_name}")
                continue
            
            process_aggregated_results(aggregated, ground_truth, dataset_dir, include_seasonal)
        
        # Create overall summary
        print("\n" + "="*70)
        print("Creating comprehensive summary...")
        print("="*70)
        
        create_full_summary(all_aggregated, spath)
        
        # Cross-dataset comparison
        all_quality = []
        for dataset_name, aggregated in all_aggregated.items():
            for config_name, data in aggregated.items():
                q = data['mean_quality']
                q.config_name = f"{dataset_name[:6]}_{config_name}"
                all_quality.append(q)
        
        if all_quality:
            compare_generation_quality(all_quality, save_path=spath / 'all_datasets_quality.pdf', top_k=5)
            compare_generation_quality(all_quality, save_path=spath / 'all_datasets_quality_all.pdf', show_all=True)
    
    else:
        # Single dataset structure (configs directly in the directory)
        print("Single dataset structure detected")
        
        # Load ground truth
        gt_path = spath / 'ground_truth.pt'
        # Also check for rescaled version
        if not gt_path.exists():
            gt_path = spath / 'ground_truth_rescaled.pt'
        ground_truth = torch.load(gt_path) if gt_path.exists() else None
        
        # Load aggregated results
        aggregated = load_dataset_results("main", spath)
        
        if aggregated:
            process_aggregated_results(aggregated, ground_truth, spath, include_seasonal)
            
            # Create summary
            summary = {"main": {}}
            for config_name, data in aggregated.items():
                q = data['mean_quality']
                summary["main"][config_name] = {
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
            
            with open(spath / 'experiment_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Saved summary to {spath / 'experiment_summary.json'}")


def plot_individual_sample_comparisons(aggregated, ground_truth, save_dir, n_plot=30):
    """Create individual comparison plots for each config (ground truth vs model)."""
    # Create a subdirectory for sample comparison plots
    samples_dir = save_dir / 'sample_comparisons'
    samples_dir.mkdir(exist_ok=True)
    
    # Determine x grid from ground truth shape
    if ground_truth.ndim == 3:
        ground_truth = ground_truth.squeeze(1)
    n_x = ground_truth.shape[-1]
    x_grid = torch.linspace(0, 1, n_x)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(aggregated)))
    
    for idx, (config_name, data) in enumerate(aggregated.items()):
        samples = data.get('samples_original') or data.get('samples')
        if samples is None:
            continue
        
        if samples.ndim == 3:
            samples = samples.squeeze(1)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
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
        
        title = config_name.replace('_', ' ').title()
        ax.set_title(f'{title}\n({samples.shape[0]} samples)', fontsize=11, fontweight='bold')
        ax.set_xlabel('t', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Match y-axis limits
        y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
        y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
        axes[0].set_ylim(y_min, y_max)
        axes[1].set_ylim(y_min, y_max)
        
        plt.suptitle(f'Ground Truth vs {config_name}', fontsize=12, y=1.02)
        plt.tight_layout()
        
        # Save individual plot
        safe_name = config_name.replace('/', '_').replace('\\', '_')
        plt.savefig(samples_dir / f'{safe_name}.pdf', dpi=150, bbox_inches='tight')
        plt.savefig(samples_dir / f'{safe_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  Saved {len(aggregated)} individual sample comparison plots to {samples_dir}")


def process_aggregated_results(aggregated, ground_truth, save_dir, include_seasonal=False):
    """Process aggregated results and generate visualizations."""
    
    print(f"  Generating visualizations...")
    
    # Filter out baselines for OT-specific plots (training, convergence, spectrum)
    ot_aggregated = filter_ot_configs(aggregated)
    n_baselines = len(aggregated) - len(ot_aggregated)
    if n_baselines > 0:
        baseline_names = [k for k in aggregated.keys() if is_baseline_config(k)]
        print(f"  Found {n_baselines} baseline model(s): {baseline_names}")
        print(f"  Using {len(ot_aggregated)} OT configs for training/convergence/spectrum plots")
    
    # Training comparison (OT configs only - baselines don't have training metrics)
    training_list = [d['training_metrics'] for d in ot_aggregated.values() if d['training_metrics']]
    if training_list:
        try:
            figs = compare_training_runs(training_list, save_path=save_dir / 'training')
            for fig in figs:
                plt.close(fig)
            plot_convergence_comparison(training_list, save_path=save_dir / 'convergence.pdf')
        except Exception as e:
            print(f"  Warning: Could not generate training comparison: {e}")
    
    # Convergence analysis (OT configs only)
    losses_dict = {
        name: d['training_metrics'].train_losses 
        for name, d in ot_aggregated.items() 
        if d['training_metrics'] is not None and d['training_metrics'].train_losses
    }
    if losses_dict:
        try:
            figs, conv_metrics = compare_convergence(
                losses_dict,
                save_path=save_dir / 'convergence_metrics'
            )
            for fig in figs:
                plt.close(fig)
            
            with open(save_dir / 'convergence_metrics.json', 'w') as f:
                conv_json = {
                    k: {kk: float(vv) if vv is not None else None for kk, vv in v.items()}
                    for k, v in conv_metrics.items()
                }
                json.dump(conv_json, f, indent=2)
            print_convergence_table(conv_metrics)
        except Exception as e:
            print(f"  Warning: Could not generate convergence analysis: {e}")
    
    # Spectrum comparison (OT configs only)
    if ground_truth is not None:
        generated_list = [d['samples_original'] for d in ot_aggregated.values() if d['samples_original'] is not None]
        config_names = [k for k, d in ot_aggregated.items() if d['samples_original'] is not None]
        
        if generated_list:
            try:
                fig = compare_spectra_1d(
                    ground_truth,
                    generated_list,
                    config_names=config_names,
                    save_path=save_dir / 'spectrum_comparison.pdf'
                )
                plt.close(fig)
            except Exception as e:
                print(f"  Warning: Could not generate spectrum comparison: {e}")
        
        # Sample comparison plots (all configs including baselines)
        try:
            plot_individual_sample_comparisons(aggregated, ground_truth, save_dir)
        except Exception as e:
            print(f"  Warning: Could not generate sample comparisons: {e}")
    
    # Quality comparison (all configs including baselines)
    quality_list = [d['mean_quality'] for d in aggregated.values()]
    try:
        compare_generation_quality(
            quality_list, 
            save_path=save_dir / 'quality_comparison.pdf',
            include_spectrum=True,
            include_seasonal=include_seasonal,
            top_k=5,
            show_all=False,
        )
        # Also save a version with all configs for reference
        compare_generation_quality(
            quality_list, 
            save_path=save_dir / 'quality_comparison_all.pdf',
            include_spectrum=True,
            include_seasonal=include_seasonal,
            show_all=True,
        )
    except Exception as e:
        print(f"  Warning: Could not generate quality comparison: {e}")
    
    # Save comprehensive metrics (all configs including baselines)
    try:
        save_all_metrics_summary(quality_list, save_dir / 'comprehensive_metrics.json')
    except Exception as e:
        print(f"  Warning: Could not save metrics summary: {e}")
    
    print(f"\n--- Quality Results (including baselines) ---")
    print_comprehensive_table(quality_list)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-process OT-FFM experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python econ_ot_postprocess.py                                    # Default: econ_ot_comprehensive
    python econ_ot_postprocess.py ../outputs/AEMET_ot_comprehensive  # AEMET experiments
    python econ_ot_postprocess.py ../outputs/moGP_ot_comprehensive   # moGP experiments
    python econ_ot_postprocess.py ../outputs/rBergomi_ot_H0p10       # rBergomi experiments
    python econ_ot_postprocess.py ../outputs/Heston_ot_kappa1.0      # Heston experiments
        """
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="../outputs/econ_ot_comprehensive",
        help="Path to experiment output directory (default: ../outputs/econ_ot_comprehensive)"
    )
    parser.add_argument(
        "--seasonal",
        action="store_true",
        help="Include seasonal pattern analysis (for AEMET-like data)"
    )
    
    args = parser.parse_args()
    
    spath = Path(args.directory)
    
    if not spath.exists():
        print(f"Error: Directory not found: {spath}")
        sys.exit(1)
    
    print("="*70)
    print("Post-processing OT-FFM Experiments")
    print(f"Directory: {spath.resolve()}")
    print("="*70)
    
    process_single_directory(spath, include_seasonal=args.seasonal)
    
    plt.close('all')
    
    print("\n" + "="*70)
    print(f"Post-processing complete! Results saved to: {spath}")
    print("="*70)

