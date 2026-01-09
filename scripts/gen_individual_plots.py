#!/usr/bin/env python3
"""
Generate individual bar plots showing best performance per kernel category
for time series datasets (AEMET, expr_genes, Heston, rBergomi).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def compute_convergence_rate(train_losses: List[float]) -> Optional[float]:
    """
    Compute convergence rate from training loss history.
    
    Convergence rate = (log(loss_0) - log(loss_half)) / half_epochs
    
    Higher values indicate faster convergence.
    """
    if not train_losses or len(train_losses) < 2:
        return None
    
    losses = np.array(train_losses)
    n_epochs = len(losses)
    half_idx = max(1, n_epochs // 2)
    
    if losses[0] > 0 and losses[half_idx] > 0:
        return float((np.log(losses[0]) - np.log(losses[half_idx])) / half_idx)
    else:
        return float((losses[0] - losses[half_idx]) / half_idx)


def compute_convergence_for_config(config_dir: Path, seeds: List[int] = [1, 2, 4]) -> Optional[float]:
    """
    Compute average convergence rate for a config across seeds.
    Looks for training_metrics.json files in seed subdirectories.
    """
    convergence_rates = []
    
    for seed in seeds:
        seed_dir = config_dir / f'seed_{seed}'
        training_file = seed_dir / 'training_metrics.json'
        
        if training_file.exists():
            try:
                with open(training_file, 'r') as f:
                    training_data = json.load(f)
                
                train_losses = training_data.get('train_losses', [])
                if train_losses:
                    rate = compute_convergence_rate(train_losses)
                    if rate is not None:
                        convergence_rates.append(rate)
            except (json.JSONDecodeError, KeyError):
                continue
    
    if convergence_rates:
        return float(np.mean(convergence_rates))
    return None


# Define kernel categories and their prefixes
KERNEL_CATEGORIES_ALL = {
    'Signature': ['signature_sinkhorn_reg0.1', 'signature_sinkhorn_reg0.5', 'signature_sinkhorn_reg1.0'],
    'RBF': ['rbf_exact', 'rbf_sinkhorn_reg0.1', 'rbf_sinkhorn_reg0.5', 'rbf_sinkhorn_reg1.0'],
    'Euclidean': ['euclidean_exact', 'euclidean_sinkhorn_reg0.1', 'euclidean_sinkhorn_reg0.5', 'euclidean_sinkhorn_reg1.0'],
    'Gaussian': ['gaussian_ot'],
    'Independent': ['independent'],
}

# Default: use all kernel categories
KERNEL_CATEGORIES = KERNEL_CATEGORIES_ALL.copy()


def get_kernel_categories(ignore_gaussian: bool = True) -> Dict:
    """Get kernel categories, optionally filtering out Gaussian."""
    if ignore_gaussian:
        return {k: v for k, v in KERNEL_CATEGORIES_ALL.items() if k != 'Gaussian'}
    return KERNEL_CATEGORIES_ALL.copy()

# Metrics to plot (excluding skewness and density per user request)
METRICS = ['mean_mse', 'variance_mse', 'kurtosis_mse', 'autocorrelation_mse']
METRIC_LABELS = ['Mean', 'Variance', 'Kurtosis', 'Autocorrelation']

# Color palette - soft, aesthetic pastels with good contrast
COLORS = ['#6BAED6', '#FC8D62', '#8DA0CB', '#66C2A5']  # Soft blue, coral, lavender, mint

# Dataset configurations
# summary_files is a list of files to try in order (first found is used)
DATASETS = {
    # Time series datasets
    'AEMET': {
        'dir': 'AEMET_ot_comprehensive',
        'title': 'AEMET',
        'summary_files': ['comprehensive_metrics.json', 'experiment_summary.json'],
    },
    'expr_genes': {
        'dir': 'expr_genes_ot_comprehensive', 
        'title': 'Gene Expression',
        'summary_files': ['comprehensive_metrics.json', 'experiment_summary.json'],
    },
    'Heston': {
        'dir': 'Heston_ot_kappa1.0',
        'title': 'Heston',
        'summary_files': ['comprehensive_metrics.json', 'experiment_summary.json'],
    },
    'rBergomi': {
        'dir': 'rBergomi_ot_H0p10',
        'title': 'rBergomi',
        'summary_files': ['comprehensive_metrics.json', 'experiment_summary.json'],
    },
    'econ1': {
        'dir': 'econ_ot_comprehensive/econ1_population',
        'title': 'Economy',
        'summary_files': ['comprehensive_metrics.json'],
    },
    'moGP': {
        'dir': 'moGP_ot_comprehensive',
        'title': 'Multi-output GP',
        'summary_files': ['comprehensive_metrics.json', 'experiment_summary.json'],
    },
    # PDE datasets
    'kdv': {
        'dir': 'kdv_ot',
        'title': 'KdV',
        'summary_files': ['comprehensive_metrics.json', 'experiment_summary.json'],
    },
    'ginzburg_landau': {
        'dir': 'ginzburg_landau_ot',
        'title': 'Ginzburg-Landau',
        'summary_files': ['comprehensive_metrics.json', 'experiment_summary.json'],
    },
    'navier_stokes': {
        'dir': 'navier_stokes_ot',
        'title': 'Navier-Stokes',
        'summary_files': ['comprehensive_metrics.json', 'experiment_summary.json'],
    },
    'stochastic_kdv': {
        'dir': 'stochastic_kdv_ot',
        'title': 'Stochastic KdV',
        'summary_files': ['comprehensive_metrics.json', 'experiment_summary.json'],
    },
    'stochastic_ns': {
        'dir': 'stochastic_ns_ot',
        'title': 'Stochastic NS',
        'summary_files': ['comprehensive_metrics.json', 'experiment_summary.json'],
    },
}


def load_experiment_summary(output_dir: Path) -> Optional[Dict]:
    """Load the experiment summary JSON file."""
    summary_file = output_dir / 'experiment_summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            return json.load(f)
    
    # Try comprehensive_metrics.json as fallback
    comprehensive_file = output_dir / 'comprehensive_metrics.json'
    if comprehensive_file.exists():
        with open(comprehensive_file, 'r') as f:
            return json.load(f)
    
    return None


def normalize_configs(configs) -> Dict:
    """
    Normalize configs to a dict format.
    Handles both dict format (experiment_summary.json) and list format (comprehensive_metrics.json).
    """
    if isinstance(configs, dict):
        return configs
    elif isinstance(configs, list):
        # Convert list format to dict format
        # Each item has 'config_name' field, possibly with prefix like 'econ1_population_'
        normalized = {}
        for item in configs:
            config_name = item.get('config_name', '')
            # Remove common prefixes (e.g., 'econ1_population_independent' -> 'independent')
            for prefix in ['econ1_population_', 'econ2_population_', 'econ3_population_']:
                if config_name.startswith(prefix):
                    config_name = config_name[len(prefix):]
                    break
            normalized[config_name] = item
        return normalized
    return {}


def get_best_config_per_category(configs, categories: Dict = KERNEL_CATEGORIES) -> Dict:
    """
    Find the best performing configuration for each kernel category.
    Best is determined by lowest average MSE across all metrics.
    
    Also includes sweep configs (sig_*, rbf_*, euclidean_*) in their respective categories.
    
    Returns dict mapping category name -> best config data
    """
    # Normalize configs to dict format
    configs = normalize_configs(configs)
    
    # Extend categories to include sweep configs
    extended_categories = {cat: list(names) for cat, names in categories.items()}
    
    # Map sweep prefixes to categories
    prefix_to_category = {
        'sig_': 'Signature',
        'signature_': 'Signature',
        'rbf_': 'RBF',
        'euclidean_': 'Euclidean',
        'gaussian_': 'Gaussian',
    }
    
    # Add sweep configs to appropriate categories
    for config_name in configs.keys():
        for prefix, category in prefix_to_category.items():
            if config_name.startswith(prefix) and category in extended_categories:
                if config_name not in extended_categories[category]:
                    extended_categories[category].append(config_name)
                break
    
    best_per_category = {}
    
    for category_name, config_names in extended_categories.items():
        best_config = None
        best_avg_mse = float('inf')
        
        for config_name in config_names:
            if config_name in configs:
                config_data = configs[config_name]
                
                # Get quality metrics
                if 'quality_metrics' in config_data:
                    metrics = config_data['quality_metrics']
                else:
                    # Config itself contains the metrics directly
                    metrics = config_data
                
                # Compute average MSE across the metrics we care about
                mse_values = []
                for metric in METRICS:
                    val = metrics.get(metric)
                    if val is not None and isinstance(val, (int, float)):
                        mse_values.append(val)
                
                if mse_values:
                    avg_mse = np.mean(mse_values)
                    if avg_mse < best_avg_mse:
                        best_avg_mse = avg_mse
                        best_config = {
                            'name': config_name,
                            'metrics': metrics
                        }
        
        if best_config:
            best_per_category[category_name] = best_config
    
    return best_per_category


def compute_ranks(best_configs: Dict, metrics: List[str] = METRICS) -> Dict[str, Dict[str, int]]:
    """
    Compute the rank of each category for each metric.
    Rank 1 = lowest MSE (best), higher rank = worse.
    
    Returns dict: {metric: {category: rank}}
    """
    categories = list(best_configs.keys())
    ranks = {metric: {} for metric in metrics}
    
    for metric in metrics:
        # Get values for all categories (skip None values)
        values = []
        for category in categories:
            config_data = best_configs[category]
            metric_data = config_data['metrics']
            val = metric_data.get(metric)
            if val is not None:
                values.append((category, val))
        
        # Sort by value (ascending - lower is better)
        sorted_values = sorted(values, key=lambda x: x[1])
        
        # Assign ranks
        for rank, (category, _) in enumerate(sorted_values, start=1):
            ranks[metric][category] = rank
    
    return ranks


def create_grouped_bar_plot(
    best_configs: Dict,
    dataset_name: str,
    output_path: Path,
    figsize: Tuple[float, float] = (10, 6)
):
    """
    Create a grouped bar plot showing MSE values for each metric across kernel categories.
    Displays rank labels on top of each bar.
    
    Args:
        best_configs: Dict mapping category name -> best config data
        dataset_name: Name of the dataset for the title
        output_path: Path to save the figure
        figsize: Figure size
    """
    # Setup
    categories = list(best_configs.keys())
    n_categories = len(categories)
    n_metrics = len(METRICS)
    
    # Compute ranks for each metric
    ranks = compute_ranks(best_configs)
    
    # Bar positions
    x = np.arange(n_categories)
    bar_width = 0.18
    
    # Create figure with extra space at bottom for legend
    fig, ax = plt.subplots(figsize=figsize)
    
    # Collect all values to determine y-axis range
    all_values = []
    
    # Plot bars for each metric
    for i, (metric, label) in enumerate(zip(METRICS, METRIC_LABELS)):
        values = []
        for category in categories:
            config_data = best_configs[category]
            metrics_data = config_data['metrics']
            val = metrics_data.get(metric)
            # Use a small positive value for None/0 to work with log scale
            values.append(val if val is not None and val > 0 else np.nan)
        
        # Only add valid values for y-axis range calculation
        all_values.extend([v for v in values if not np.isnan(v)])
        
        offset = (i - n_metrics / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width, label=label, color=COLORS[i], 
                     edgecolor='white', linewidth=0.8)
        
        # Add rank labels on top of each bar
        for j, (bar, category) in enumerate(zip(bars, categories)):
            # Only add rank if category has this metric
            if category in ranks[metric]:
                rank = ranks[metric][category]
                height = bar.get_height()
                if not np.isnan(height):
                    ax.annotate(
                        f'#{rank}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8, fontweight='bold',
                        color='#444444'
                    )
    
    # Extend y-axis upper limit to give room for rank labels
    if all_values:
        max_val = max(all_values)
        min_val = min(all_values)
        ax.set_ylim(min_val * 0.5, max_val * 5)  # Extra headroom in log scale
    
    # Formatting
    ax.set_xlabel('Kernel Category', fontsize=12, fontweight='medium')
    ax.set_ylabel('MSE', fontsize=12, fontweight='medium')
    ax.set_title(f'Average MSE for Statistics for {dataset_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_yscale('log')
    
    # Legend - place below the plot
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.12),
        ncol=4, 
        fontsize=10, 
        framealpha=0.95,
        edgecolor='#cccccc',
        fancybox=True
    )
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#888888')
    ax.set_axisbelow(True)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Tight layout with room for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {output_path}")
    print(f"  Saved: {output_path.with_suffix('.pdf')}")


def create_single_metric_bar_plot(
    best_configs: Dict,
    dataset_name: str,
    metric_key: str,
    metric_label: str,
    output_path: Path,
    color: str = '#6BAED6',
    figsize: Tuple[float, float] = (8, 5),
    lower_is_better: bool = True
):
    """
    Create a bar plot for a single metric across kernel categories.
    Displays rank labels on top of each bar.
    
    Args:
        best_configs: Dict mapping category name -> best config data
        dataset_name: Name of the dataset for the title
        metric_key: Key of the metric to plot (e.g., 'spectrum_mse')
        metric_label: Display label for the metric
        output_path: Path to save the figure
        color: Bar color
        figsize: Figure size
        lower_is_better: If True, rank 1 = lowest value; if False, rank 1 = highest
    """
    # Setup
    categories = list(best_configs.keys())
    n_categories = len(categories)
    
    # Get values for all categories
    values = []
    for category in categories:
        config_data = best_configs[category]
        metrics = config_data['metrics']
        val = metrics.get(metric_key, None)
        values.append(val if val is not None else 0)
    
    # Check if we have valid data
    valid_values = [v for v in values if v is not None and v > 0]
    if not valid_values:
        print(f"  Warning: No valid data for metric '{metric_key}'")
        return
    
    # Compute ranks
    indexed_values = [(i, v) for i, v in enumerate(values) if v is not None and v > 0]
    if lower_is_better:
        sorted_indexed = sorted(indexed_values, key=lambda x: x[1])
    else:
        sorted_indexed = sorted(indexed_values, key=lambda x: x[1], reverse=True)
    
    ranks = {}
    for rank, (idx, _) in enumerate(sorted_indexed, start=1):
        ranks[idx] = rank
    
    # Bar positions
    x = np.arange(n_categories)
    bar_width = 0.6
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bars
    bars = ax.bar(x, values, bar_width, color=color, edgecolor='white', linewidth=0.8)
    
    # Add rank labels on top of each bar
    for i, bar in enumerate(bars):
        if i in ranks:
            rank = ranks[i]
            height = bar.get_height()
            ax.annotate(
                f'#{rank}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=10, fontweight='bold',
                color='#444444'
            )
    
    # Extend y-axis for rank labels
    if valid_values:
        max_val = max(valid_values)
        min_val = min(valid_values)
        ax.set_ylim(min_val * 0.3, max_val * 4)
    
    # Formatting
    ax.set_xlabel('Kernel Category', fontsize=12, fontweight='medium')
    ax.set_ylabel(metric_label, fontsize=12, fontweight='medium')
    ax.set_title(f'{metric_label} for {dataset_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_yscale('log')
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#888888')
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {output_path}")
    print(f"  Saved: {output_path.with_suffix('.pdf')}")


def process_dataset(dataset_key: str, outputs_dir: Path, plots_output_dir: Path, ignore_gaussian: bool = True):
    """Process a single dataset and generate its plot."""
    dataset_info = DATASETS[dataset_key]
    dataset_dir = outputs_dir / dataset_info['dir']
    
    if not dataset_dir.exists():
        print(f"  Warning: Directory not found: {dataset_dir}")
        return False
    
    # First, try to load aggregated results (includes sweep experiments)
    aggregated = load_aggregated_results(dataset_dir, dataset_key)
    
    if aggregated and 'configs' in aggregated:
        print(f"  Using: aggregated_results (includes sweep experiments)")
        # Extract configs from aggregated format
        configs = {}
        for config_name, config_data in aggregated['configs'].items():
            if 'metrics' in config_data:
                configs[config_name] = config_data['metrics']
            else:
                configs[config_name] = config_data
    else:
        # Fallback: Load experiment summary - try files in order from summary_files list
        summary = None
        summary_files = dataset_info.get('summary_files', ['comprehensive_metrics.json', 'experiment_summary.json'])
        
        for summary_file in summary_files:
            summary_path = dataset_dir / summary_file
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                print(f"  Using: {summary_file}")
                break
        
        if summary is None:
            print(f"  Warning: No experiment summary found in {dataset_dir}")
            return False
        
        # Get configs dict - check for nested config key (e.g., econ1_population)
        config_key = dataset_info.get('config_key', None)
        if config_key:
            if config_key in summary:
                configs = summary[config_key]
            else:
                print(f"  Warning: Config key '{config_key}' not found in summary")
                return False
        elif 'configs' in summary:
            configs = summary['configs']
        else:
            print(f"  Warning: No 'configs' key found in summary")
            return False
    
    # Find best config per category (with optional Gaussian filtering)
    categories = get_kernel_categories(ignore_gaussian=ignore_gaussian)
    best_configs = get_best_config_per_category(configs, categories=categories)
    
    if not best_configs:
        print(f"  Warning: No valid configurations found")
        return False
    
    # Fill in missing convergence rates by computing from training_metrics.json
    for category, config_data in best_configs.items():
        metrics = config_data.get('metrics', {})
        if metrics.get('convergence_rate') is None:
            config_name = config_data.get('name', '')
            config_subdir = dataset_dir / config_name
            if config_subdir.exists():
                conv_rate = compute_convergence_for_config(config_subdir)
                if conv_rate is not None:
                    metrics['convergence_rate'] = conv_rate
                    print(f"  Computed convergence_rate for {category}: {conv_rate:.4f}")
    
    # Print which configs were selected
    print(f"  Best configs per category:")
    for category, data in best_configs.items():
        print(f"    {category}: {data['name']}")
    
    # Create main statistics plot
    output_filename = f"{dataset_key}_category_comparison.png"
    output_path = plots_output_dir / output_filename
    
    create_grouped_bar_plot(
        best_configs=best_configs,
        dataset_name=dataset_info['title'],
        output_path=output_path
    )
    
    # Create spectrum MSE plot (if data available)
    # Use spectrum_mse_log for PDE datasets (log scale is more appropriate for energy spectra)
    is_pde = dataset_key in PDE_DATASETS
    spectrum_key = 'spectrum_mse_log' if is_pde else 'spectrum_mse'
    spectrum_label = 'Spectrum MSE (log)' if is_pde else 'Spectrum MSE'
    
    has_spectrum = any(
        best_configs[cat]['metrics'].get(spectrum_key) is not None 
        for cat in best_configs
    )
    if has_spectrum:
        spectrum_output = plots_output_dir / f"{dataset_key}_{spectrum_key}.png"
        create_single_metric_bar_plot(
            best_configs=best_configs,
            dataset_name=dataset_info['title'],
            metric_key=spectrum_key,
            metric_label=spectrum_label,
            output_path=spectrum_output,
            color='#9E9AC8',  # Soft purple
            lower_is_better=True
        )
    
    # Create convergence rate plot (if data available)
    has_convergence = any(
        best_configs[cat]['metrics'].get('convergence_rate') is not None 
        for cat in best_configs
    )
    if has_convergence:
        convergence_output = plots_output_dir / f"{dataset_key}_convergence_rate.png"
        create_single_metric_bar_plot(
            best_configs=best_configs,
            dataset_name=dataset_info['title'],
            metric_key='convergence_rate',
            metric_label='Convergence Rate',
            output_path=convergence_output,
            color='#74C476',  # Soft green
            lower_is_better=False  # Higher convergence rate is better (faster convergence)
        )
    
    return True


def generate_all_plots(outputs_dir: Path = None, plots_output_dir: Path = None, ignore_gaussian: bool = True):
    """Generate plots for all time series datasets."""
    # Default paths
    if outputs_dir is None:
        script_dir = Path(__file__).parent
        outputs_dir = script_dir.parent / 'outputs'
    
    if plots_output_dir is None:
        plots_output_dir = outputs_dir
    
    print("=" * 60)
    print("Generating Individual Category Comparison Plots")
    if ignore_gaussian:
        print("(Ignoring Gaussian kernel category)")
    print("=" * 60)
    
    for dataset_key in DATASETS:
        print(f"\nProcessing {dataset_key}...")
        success = process_dataset(dataset_key, outputs_dir, plots_output_dir, ignore_gaussian=ignore_gaussian)
        if success:
            print(f"  ✓ Successfully generated plot for {dataset_key}")
        else:
            print(f"  ✗ Failed to generate plot for {dataset_key}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


def generate_single_plot(
    dataset_key: str,
    outputs_dir: Path = None,
    plots_output_dir: Path = None,
    ignore_gaussian: bool = True
):
    """Generate plot for a single dataset."""
    if dataset_key not in DATASETS:
        print(f"Error: Unknown dataset '{dataset_key}'")
        print(f"Available datasets: {list(DATASETS.keys())}")
        return
    
    if outputs_dir is None:
        script_dir = Path(__file__).parent
        outputs_dir = script_dir.parent / 'outputs'
    
    if plots_output_dir is None:
        plots_output_dir = outputs_dir
    
    print(f"Generating plot for {dataset_key}...")
    success = process_dataset(dataset_key, outputs_dir, plots_output_dir, ignore_gaussian=ignore_gaussian)
    if success:
        print(f"✓ Successfully generated plot for {dataset_key}")
    else:
        print(f"✗ Failed to generate plot for {dataset_key}")


# =============================================================================
# Training Loss Curve Plots
# =============================================================================

# Colors for kernel categories in loss curve plots
KERNEL_CURVE_COLORS = {
    'Independent': '#1f77b4',   # Blue
    'Signature': '#2ca02c',     # Green
    'RBF': '#ff7f0e',           # Orange
    'Euclidean': '#d62728',     # Red
}


def load_training_losses(config_dir: Path, seeds: List[int] = [1, 2, 4]) -> Optional[List[float]]:
    """
    Load training losses from a config directory.
    Returns averaged losses across available seeds.
    """
    all_losses = []
    max_len = 0
    
    for seed in seeds:
        seed_dir = config_dir / f'seed_{seed}'
        training_file = seed_dir / 'training_metrics.json'
        
        if training_file.exists():
            try:
                with open(training_file, 'r') as f:
                    training_data = json.load(f)
                
                train_losses = training_data.get('train_losses', [])
                if train_losses:
                    all_losses.append(train_losses)
                    max_len = max(max_len, len(train_losses))
            except (json.JSONDecodeError, IOError):
                continue
    
    if not all_losses:
        return None
    
    # Pad shorter sequences and average
    padded = []
    for losses in all_losses:
        if len(losses) < max_len:
            # Pad with last value
            losses = losses + [losses[-1]] * (max_len - len(losses))
        padded.append(losses)
    
    # Return averaged losses
    avg_losses = np.mean(padded, axis=0).tolist()
    return avg_losses


def find_best_config_with_losses(
    dataset_dir: Path,
    category: str,
    config_names: List[str],
    configs: Dict,
    seeds: List[int] = [1, 2, 4]
) -> Tuple[Optional[str], Optional[List[float]], Optional[float]]:
    """
    Find the best config in a category that has training loss data.
    Returns (config_name, losses, convergence_rate).
    """
    best_config = None
    best_losses = None
    best_conv_rate = None
    
    # Extended config names including sweep configs
    extended_names = list(config_names)
    
    # Add sweep configs based on category
    prefix_map = {
        'Signature': ['sig_', 'signature_'],
        'RBF': ['rbf_'],
        'Euclidean': ['euclidean_'],
    }
    
    if category in prefix_map:
        for config_name in configs.keys():
            for prefix in prefix_map[category]:
                if config_name.startswith(prefix) and config_name not in extended_names:
                    # Skip gaussian configs
                    if 'gaussian' in config_name.lower():
                        continue
                    extended_names.append(config_name)
    
    for config_name in extended_names:
        config_subdir = dataset_dir / config_name
        
        if not config_subdir.exists():
            continue
        
        # Load training losses
        losses = load_training_losses(config_subdir, seeds)
        if losses is None:
            continue
        
        # Compute convergence rate
        conv_rate = compute_convergence_rate(losses)
        if conv_rate is None:
            continue
        
        # Select best by convergence rate (higher is better)
        if best_conv_rate is None or conv_rate > best_conv_rate:
            best_conv_rate = conv_rate
            best_losses = losses
            best_config = config_name
    
    return best_config, best_losses, best_conv_rate


def plot_loss_curves(
    dataset_key: str,
    outputs_dir: Path,
    plots_output_dir: Path,
    ignore_gaussian: bool = True,
    figsize: Tuple[float, float] = (10, 6)
) -> bool:
    """
    Plot training loss curves for the best config of each kernel category.
    
    Args:
        dataset_key: Dataset identifier
        outputs_dir: Base outputs directory
        plots_output_dir: Directory to save plots
        ignore_gaussian: If True, exclude Gaussian category
        figsize: Figure size
    
    Returns:
        True if plot was generated successfully
    """
    dataset_info = DATASETS.get(dataset_key)
    if not dataset_info:
        print(f"  Unknown dataset: {dataset_key}")
        return False
    
    dataset_dir = outputs_dir / dataset_info['dir']
    if not dataset_dir.exists():
        print(f"  Directory not found: {dataset_dir}")
        return False
    
    # Load configs
    aggregated = load_aggregated_results(dataset_dir, dataset_key)
    if aggregated and 'configs' in aggregated:
        configs = {}
        for config_name, config_data in aggregated['configs'].items():
            if 'metrics' in config_data:
                configs[config_name] = config_data['metrics']
            else:
                configs[config_name] = config_data
    else:
        # Fallback to summary files
        summary = None
        summary_files = dataset_info.get('summary_files', ['comprehensive_metrics.json', 'experiment_summary.json'])
        for summary_file in summary_files:
            summary_path = dataset_dir / summary_file
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                break
        
        if not summary:
            print(f"  No summary file found for {dataset_key}")
            return False
        
        if 'configs' in summary:
            configs = normalize_configs(summary['configs'])
        else:
            print(f"  No configs found in summary for {dataset_key}")
            return False
    
    # Get kernel categories
    categories = get_kernel_categories(ignore_gaussian=ignore_gaussian)
    
    # Find best config with losses for each category
    category_data = {}  # {category: (config_name, losses, conv_rate)}
    
    for category, config_names in categories.items():
        config_name, losses, conv_rate = find_best_config_with_losses(
            dataset_dir, category, config_names, configs
        )
        if losses is not None:
            category_data[category] = (config_name, losses, conv_rate)
            print(f"    {category}: {config_name} (conv_rate={conv_rate:.4f})")
    
    if not category_data:
        print(f"  No training loss data found for {dataset_key}")
        return False
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    for category, (config_name, losses, conv_rate) in category_data.items():
        color = KERNEL_CURVE_COLORS.get(category, '#333333')
        epochs = np.arange(len(losses))
        
        # Plot with label showing config name
        label = f"{category}"
        ax.plot(epochs, losses, color=color, linewidth=2, label=label, alpha=0.9)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title(f'Training Loss Curves - {dataset_info["title"]}', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    
    # Save
    output_path = plots_output_dir / f"{dataset_key}_loss_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {output_path}")
    return True


def generate_loss_curve_plots(
    outputs_dir: Path = None,
    plots_output_dir: Path = None,
    ignore_gaussian: bool = True,
    dataset_key: str = None
):
    """
    Generate training loss curve plots for datasets.
    
    Args:
        outputs_dir: Base outputs directory
        plots_output_dir: Directory to save plots
        ignore_gaussian: If True, exclude Gaussian category
        dataset_key: If specified, only generate for this dataset
    """
    if outputs_dir is None:
        script_dir = Path(__file__).parent
        outputs_dir = script_dir.parent / 'outputs'
    
    if plots_output_dir is None:
        plots_output_dir = outputs_dir
    
    print("=" * 60)
    print("Generating Training Loss Curve Plots")
    if ignore_gaussian:
        print("(Excluding Gaussian OT)")
    print("=" * 60)
    
    datasets_to_process = [dataset_key] if dataset_key else list(DATASETS.keys())
    
    for ds_key in datasets_to_process:
        if ds_key not in DATASETS:
            print(f"\nSkipping unknown dataset: {ds_key}")
            continue
        
        print(f"\nProcessing {ds_key}...")
        success = plot_loss_curves(ds_key, outputs_dir, plots_output_dir, ignore_gaussian)
        if success:
            print(f"  ✓ Generated loss curve plot for {ds_key}")
        else:
            print(f"  ✗ Failed to generate loss curve plot for {ds_key}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


def generate_combined_loss_curves_plot(
    outputs_dir: Path = None,
    plots_output_dir: Path = None,
    ignore_gaussian: bool = True,
    dpi: int = 300
):
    """
    Generate a combined 5x2 plot with loss curves for all datasets (excluding moGP).
    High resolution for full-page figures.
    
    Args:
        outputs_dir: Base outputs directory
        plots_output_dir: Directory to save plots
        ignore_gaussian: If True, exclude Gaussian category
        dpi: DPI for output (default 300 for high resolution)
    """
    if outputs_dir is None:
        script_dir = Path(__file__).parent
        outputs_dir = script_dir.parent / 'outputs'
    
    if plots_output_dir is None:
        plots_output_dir = outputs_dir
    
    print("=" * 60)
    print("Generating Combined Loss Curves Plot (5x2)")
    if ignore_gaussian:
        print("(Excluding Gaussian OT)")
    print("=" * 60)
    
    # Datasets to include (exclude moGP)
    # Order: sequence datasets first row, then PDE datasets
    datasets_order = [
        # Row 1-2: Sequence datasets
        'AEMET', 'expr_genes',
        'econ1', 'Heston',
        'rBergomi', 'kdv',
        # Row 3-5: PDE datasets
        'navier_stokes', 'stochastic_kdv',
        'stochastic_ns', 'ginzburg_landau',
    ]
    
    # Get kernel categories
    categories = get_kernel_categories(ignore_gaussian=ignore_gaussian)
    
    # Collect data for all datasets
    all_data = {}  # {dataset_key: {category: (config_name, losses, conv_rate)}}
    
    for ds_key in datasets_order:
        if ds_key not in DATASETS:
            print(f"  Skipping unknown dataset: {ds_key}")
            continue
        
        dataset_info = DATASETS[ds_key]
        dataset_dir = outputs_dir / dataset_info['dir']
        
        if not dataset_dir.exists():
            print(f"  Directory not found for {ds_key}: {dataset_dir}")
            continue
        
        # Load configs
        aggregated = load_aggregated_results(dataset_dir, ds_key)
        if aggregated and 'configs' in aggregated:
            configs = {}
            for config_name, config_data in aggregated['configs'].items():
                if 'metrics' in config_data:
                    configs[config_name] = config_data['metrics']
                else:
                    configs[config_name] = config_data
        else:
            # Fallback to summary files
            summary = None
            summary_files = dataset_info.get('summary_files', ['comprehensive_metrics.json', 'experiment_summary.json'])
            for summary_file in summary_files:
                summary_path = dataset_dir / summary_file
                if summary_path.exists():
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)
                    break
            
            if not summary or 'configs' not in summary:
                print(f"  No configs found for {ds_key}")
                continue
            
            configs = normalize_configs(summary['configs'])
        
        # Find best config with losses for each category
        category_data = {}
        for category, config_names in categories.items():
            config_name, losses, conv_rate = find_best_config_with_losses(
                dataset_dir, category, config_names, configs
            )
            if losses is not None:
                category_data[category] = (config_name, losses, conv_rate)
        
        if category_data:
            all_data[ds_key] = category_data
            print(f"  ✓ {ds_key}: {len(category_data)} kernel categories with data")
        else:
            print(f"  ✗ {ds_key}: No loss data found")
    
    if not all_data:
        print("\nNo data found for any dataset!")
        return
    
    # Create 5x2 figure (high resolution)
    fig, axes = plt.subplots(5, 2, figsize=(14, 18))
    axes = axes.flatten()
    
    # Plot each dataset
    plot_idx = 0
    for ds_key in datasets_order:
        if ds_key not in all_data:
            # Leave subplot empty but hide it
            if plot_idx < len(axes):
                axes[plot_idx].set_visible(False)
            plot_idx += 1
            continue
        
        if plot_idx >= len(axes):
            break
        
        ax = axes[plot_idx]
        category_data = all_data[ds_key]
        dataset_title = DATASETS[ds_key]['title']
        
        for category, (config_name, losses, conv_rate) in category_data.items():
            color = KERNEL_CURVE_COLORS.get(category, '#333333')
            epochs = np.arange(len(losses))
            ax.plot(epochs, losses, color=color, linewidth=1.5, label=category, alpha=0.9)
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_title(dataset_title, fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=9)
        
        # Only add legend to first subplot
        if plot_idx == 0:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.95)
        
        plot_idx += 1
    
    # Hide any unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)
    
    # Add a shared legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=11, 
               bbox_to_anchor=(0.5, 0.01), frameon=True, framealpha=0.95)
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])  # Leave room for legend at bottom
    
    # Save with high resolution
    output_path = plots_output_dir / 'all_loss_curves_combined.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Saved: {output_path} (dpi={dpi})")
    print(f"✓ Saved: {output_path.with_suffix('.pdf')}")
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


def generate_combined_plot(outputs_dir: Path = None, plots_output_dir: Path = None, ignore_gaussian: bool = True):
    """Generate a single figure with all datasets as subplots."""
    if outputs_dir is None:
        script_dir = Path(__file__).parent
        outputs_dir = script_dir.parent / 'outputs'
    
    if plots_output_dir is None:
        plots_output_dir = outputs_dir
    
    # Get kernel categories (with optional Gaussian filtering)
    categories = get_kernel_categories(ignore_gaussian=ignore_gaussian)
    
    # Collect data from all datasets
    all_data = {}
    for dataset_key in DATASETS:
        dataset_info = DATASETS[dataset_key]
        dataset_dir = outputs_dir / dataset_info['dir']
        
        if not dataset_dir.exists():
            continue
        
        # Try files in order from summary_files list
        summary = None
        summary_files = dataset_info.get('summary_files', ['comprehensive_metrics.json', 'experiment_summary.json'])
        
        for summary_file in summary_files:
            summary_path = dataset_dir / summary_file
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                break
        
        if summary is None:
            continue
        
        # Handle nested config key
        config_key = dataset_info.get('config_key', None)
        if config_key:
            if config_key not in summary:
                continue
            configs = summary[config_key]
        elif 'configs' in summary:
            configs = summary['configs']
        else:
            continue
        
        best_configs = get_best_config_per_category(configs, categories=categories)
        if best_configs:
            # Fill in missing convergence rates
            for category, config_data in best_configs.items():
                metrics = config_data.get('metrics', {})
                if metrics.get('convergence_rate') is None:
                    config_name = config_data.get('name', '')
                    config_subdir = dataset_dir / config_name
                    if config_subdir.exists():
                        conv_rate = compute_convergence_for_config(config_subdir)
                        if conv_rate is not None:
                            metrics['convergence_rate'] = conv_rate
            
            all_data[dataset_key] = {
                'title': dataset_info['title'],
                'best_configs': best_configs
            }
    
    if not all_data:
        print("No data found for any dataset")
        return
    
    # Create 2x2 subplot figure
    n_datasets = len(all_data)
    n_cols = 2
    n_rows = (n_datasets + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten() if n_datasets > 1 else [axes]
    
    for idx, (dataset_key, data) in enumerate(all_data.items()):
        ax = axes[idx]
        best_configs = data['best_configs']
        categories = list(best_configs.keys())
        n_categories = len(categories)
        n_metrics = len(METRICS)
        
        # Compute ranks for this dataset
        ranks = compute_ranks(best_configs)
        
        x = np.arange(n_categories)
        bar_width = 0.18
        
        all_values = []
        
        for i, (metric, label) in enumerate(zip(METRICS, METRIC_LABELS)):
            values = []
            for category in categories:
                config_data = best_configs[category]
                metrics_data = config_data['metrics']
                val = metrics_data.get(metric)
                values.append(val if val is not None and val > 0 else np.nan)
            
            all_values.extend([v for v in values if not np.isnan(v)])
            
            offset = (i - n_metrics / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, values, bar_width, label=label, color=COLORS[i],
                   edgecolor='white', linewidth=0.8)
            
            # Add rank labels on top of each bar
            for j, (bar, category) in enumerate(zip(bars, categories)):
                if category not in ranks[metric]:
                    continue
                rank = ranks[metric][category]
                height = bar.get_height()
                if np.isnan(height):
                    continue
                ax.annotate(
                    f'#{rank}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),  # 2 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=6, fontweight='bold',
                    color='#444444'
                )
        
        # Extend y-axis for rank labels
        if all_values:
            max_val = max(all_values)
            min_val = min(all_values)
            ax.set_ylim(min_val * 0.5, max_val * 5)
        
        ax.set_xlabel('Kernel Category', fontsize=11)
        ax.set_ylabel('MSE', fontsize=11)
        ax.set_title(f'Average MSE for Statistics for {data["title"]}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10, rotation=15, ha='right')
        ax.set_yscale('log')
        ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#888888')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if idx == 0:
            ax.legend(
                loc='upper center',
                bbox_to_anchor=(1.1, -0.15),
                ncol=4,
                fontsize=9,
                framealpha=0.95,
                edgecolor='#cccccc',
                fancybox=True
            )
    
    # Hide unused axes
    for idx in range(len(all_data), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    output_path = plots_output_dir / 'all_datasets_category_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved combined plot: {output_path}")
    print(f"Saved combined plot: {output_path.with_suffix('.pdf')}")


# =============================================================================
# Baseline Comparison Plots (DDPM, NCSN, FFM, k-FFM)
# =============================================================================

# Baseline model configurations
BASELINE_MODELS = ['DDPM', 'NCSN', 'FFM', 'k-FFM']
BASELINE_COLORS = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']  # Red, Blue, Green, Purple

BASELINE_METRICS_SEQ = ['mean_mse', 'variance_mse', 'autocorrelation_mse']
BASELINE_LABELS_SEQ = ['Mean', 'Variance', 'Autocorr.']

BASELINE_METRICS_PDE = ['mean_mse', 'variance_mse', 'spectrum_mse_log']
BASELINE_LABELS_PDE = ['Mean', 'Variance', 'Spectrum (log)']

# Dataset groups for baseline comparison
SEQUENCE_DATASETS = ['AEMET', 'expr_genes', 'econ1', 'Heston', 'rBergomi']
PDE_DATASETS = ['kdv', 'navier_stokes', 'stochastic_kdv', 'stochastic_ns']


def load_baseline_from_subdirs(dataset_dir: Path, model_name: str, seeds: List[int] = [1, 2, 4]) -> Optional[Dict]:
    """
    Load baseline model metrics from subdirectories when not in summary file.
    Averages metrics across seeds.
    """
    model_dir = dataset_dir / model_name
    if not model_dir.exists():
        return None
    
    all_metrics = []
    for seed in seeds:
        seed_dir = model_dir / f'seed_{seed}'
        quality_file = seed_dir / 'quality_metrics.json'
        if quality_file.exists():
            try:
                with open(quality_file, 'r') as f:
                    metrics = json.load(f)
                    all_metrics.append(metrics)
            except (json.JSONDecodeError, IOError):
                continue
    
    if not all_metrics:
        return None
    
    # Only average numeric metric keys (skip strings like config_name, ot_kernel, etc.)
    numeric_keys = [
        'mean_mse', 'variance_mse', 'skewness_mse', 'kurtosis_mse',
        'autocorrelation_mse', 'density_mse', 'spectrum_mse', 'spectrum_mse_log',
        'final_train_loss', 'total_train_time', 'mean_path_length', 'mean_grad_variance',
        'seasonal_mse', 'seasonal_amplitude_error', 'seasonal_phase_correlation',
        'convergence_rate', 'final_stability', 'epochs_to_90pct'
    ]
    
    # Average metrics across seeds
    avg_metrics = {}
    for key in numeric_keys:
        values = [m.get(key) for m in all_metrics if m.get(key) is not None and isinstance(m.get(key), (int, float))]
        if values:
            avg_metrics[key] = float(np.mean(values))
    
    return avg_metrics if avg_metrics else None


def load_aggregated_results(dataset_dir: Path, dataset_key: str) -> Optional[Dict]:
    """
    Load aggregated results file if it exists.
    
    Aggregated results combine original experiments + sweep experiments and
    contain the best configuration for each metric.
    """
    # Handle special cases where dataset_key differs from file naming
    key_mappings = {
        'econ1': 'econ1_population',
        'econ2': 'econ2_population', 
        'econ3': 'econ3_population',
    }
    file_key = key_mappings.get(dataset_key, dataset_key)
    
    # Try different naming patterns for aggregated results
    possible_names = [
        f'aggregated_results_{file_key}.json',
        f'aggregated_results_{dataset_key}.json',
        'aggregated_results.json',
    ]
    
    for name in possible_names:
        agg_file = dataset_dir / name
        if agg_file.exists():
            try:
                with open(agg_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
    
    # Also check parent directory for nested datasets (e.g., econ_ot_comprehensive/econ1_population)
    parent_dir = dataset_dir.parent
    for name in possible_names:
        agg_file = parent_dir / name
        if agg_file.exists():
            try:
                with open(agg_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
    
    return None


def get_baseline_models_data(dataset_key: str, outputs_dir: Path) -> Optional[Dict]:
    """Get metrics for baseline models (DDPM, NCSN, FFM, k-FFM) for a dataset.
    
    For k-FFM, we select the BEST value for each metric across all kernel variants,
    not just one globally-selected configuration. This ensures that k-FFM represents
    the best achievable performance for each metric.
    
    Priority: aggregated_results > comprehensive_metrics > experiment_summary
    """
    dataset_info = DATASETS.get(dataset_key)
    if not dataset_info:
        return None
    
    dataset_dir = outputs_dir / dataset_info['dir']
    if not dataset_dir.exists():
        return None
    
    # First, try to load aggregated results (includes sweep experiments)
    aggregated = load_aggregated_results(dataset_dir, dataset_key)
    
    if aggregated and 'configs' in aggregated:
        # Use aggregated results - they already combine original + sweep experiments
        configs = aggregated['configs']
        
        # Extract metrics from aggregated format (metrics are nested under 'metrics' key)
        normalized_configs = {}
        for config_name, config_data in configs.items():
            if 'metrics' in config_data:
                normalized_configs[config_name] = config_data['metrics']
            else:
                normalized_configs[config_name] = config_data
        
        configs = normalized_configs
    else:
        # Fallback: Load from experiment summary
        summary_files = dataset_info.get('summary_files', ['comprehensive_metrics.json', 'experiment_summary.json'])
        summary = None
        for summary_file in summary_files:
            summary_path = dataset_dir / summary_file
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                break
        
        if not summary:
            return None
        
        # Get configs
        config_key = dataset_info.get('config_key', None)
        if config_key:
            if config_key not in summary:
                return None
            configs = summary[config_key]
        elif 'configs' in summary:
            configs = summary['configs']
        else:
            return None
        
        configs = normalize_configs(configs)
    
    result = {}
    
    # Get DDPM and NCSN baselines - first try from configs, then from subdirectories
    for model in ['DDPM', 'NCSN']:
        if model in configs:
            config_data = configs[model]
            if 'quality_metrics' in config_data:
                result[model] = config_data['quality_metrics']
            else:
                result[model] = config_data
        else:
            # Fallback: try to load from subdirectory
            subdir_metrics = load_baseline_from_subdirs(dataset_dir, model)
            if subdir_metrics:
                result[model] = subdir_metrics
    
    # Get FFM (independent coupling)
    if 'independent' in configs:
        config_data = configs['independent']
        if 'quality_metrics' in config_data:
            result['FFM'] = config_data['quality_metrics']
        else:
            result['FFM'] = config_data
    
    # Get k-FFM: for each metric, find the BEST value across all OT kernel types
    # This ensures k-FFM represents the best achievable performance per metric
    # Include all possible OT config names (original + sweep experiments)
    # NOTE: gaussian_ot is EXCLUDED - focusing on kernel-based OT methods
    ot_config_names = [
        # Original configs (excluding gaussian_ot)
        'signature_sinkhorn_reg0.1', 'signature_sinkhorn_reg0.5', 'signature_sinkhorn_reg1.0',
        'rbf_exact', 'rbf_sinkhorn_reg0.1', 'rbf_sinkhorn_reg0.5', 'rbf_sinkhorn_reg1.0',
        'euclidean_exact', 'euclidean_sinkhorn_reg0.1', 'euclidean_sinkhorn_reg0.5', 'euclidean_sinkhorn_reg1.0',
        'signature_sinkhorn_barycentric', 'rbf_sinkhorn_barycentric',
    ]
    
    # Also include any config that looks like an OT config (from sweeps)
    # Exclude gaussian configs
    for config_name in configs.keys():
        if config_name not in ot_config_names and config_name not in ['DDPM', 'NCSN', 'independent', 'gaussian_ot']:
            # Skip gaussian configs
            if config_name.startswith('gaussian'):
                continue
            # Check if it's an OT config (has use_ot=True or ot_kernel set)
            config_data = configs[config_name]
            if isinstance(config_data, dict):
                # Also check ot_method is not gaussian
                ot_method = config_data.get('ot_method', '')
                if ot_method == 'gaussian':
                    continue
                if config_data.get('use_ot') == True or config_data.get('ot_kernel'):
                    ot_config_names.append(config_name)
                # Also check for sweep configs (sig_*, rbf_*, euclidean_*)
                elif config_name.startswith(('sig_', 'rbf_', 'euclidean_', 'gp_')):
                    ot_config_names.append(config_name)
    
    # All metrics we might want to compare
    all_metrics = [
        'mean_mse', 'variance_mse', 'kurtosis_mse', 'skewness_mse', 
        'autocorrelation_mse', 'spectrum_mse', 'spectrum_mse_log', 'density_mse'
    ]
    
    # Collect all metrics from all OT configs
    ot_metrics_list = []
    for config_name in ot_config_names:
        if config_name in configs:
            config_data = configs[config_name]
            if 'quality_metrics' in config_data:
                ot_metrics_list.append(config_data['quality_metrics'])
            else:
                ot_metrics_list.append(config_data)
    
    if ot_metrics_list:
        # Build k-FFM metrics by taking the best (minimum) value for each metric
        kffm_metrics = {}
        for metric in all_metrics:
            values = [m.get(metric) for m in ot_metrics_list 
                     if m.get(metric) is not None and isinstance(m.get(metric), (int, float))]
            if values:
                kffm_metrics[metric] = min(values)
        
        if kffm_metrics:
            result['k-FFM'] = kffm_metrics
    
    return result if result else None


def generate_baseline_comparison_plot(
    dataset_keys: List[str],
    metrics: List[str],
    metric_labels: List[str],
    title: str,
    output_path: Path,
    outputs_dir: Path
):
    """Generate baseline comparison bar plot across datasets."""
    
    # Abbreviations for dataset names
    DATASET_ABBREVIATIONS = {
        'Navier-Stokes': 'NS',
        'Stochastic KdV': 'Sto. KdV',
        'Stochastic NS': 'Sto. NS',
        '\\shortstack{Navier-\\\\Stokes}': 'NS',
        '\\shortstack{Stoch.\\\\KdV}': 'Sto. KdV',
        '\\shortstack{Stoch.\\\\NS}': 'Sto. NS',
        'Gene Expr.': 'Gene',
    }
    
    def abbreviate_title(title: str) -> str:
        """Abbreviate dataset title for plotting."""
        return DATASET_ABBREVIATIONS.get(title, title)
    
    # Collect data from all datasets
    all_data = {}
    for dataset_key in dataset_keys:
        data = get_baseline_models_data(dataset_key, outputs_dir)
        if data:
            all_data[dataset_key] = {
                'title': abbreviate_title(DATASETS[dataset_key]['title']),
                'models': data
            }
    
    if not all_data:
        print(f"No baseline data found for plot: {output_path}")
        return
    
    n_datasets = len(all_data)
    n_metrics = len(metrics)
    
    # Vertical layout (n_metrics x 1) for better presentation in double-column paper
    fig_width = 7  # Single column width
    fig_height = 3.2 * n_metrics  # Height per subplot
    fig, axes = plt.subplots(n_metrics, 1, figsize=(fig_width, fig_height))
    if n_metrics == 1:
        axes = [axes]
    
    dataset_titles = [all_data[k]['title'] for k in all_data.keys()]
    x = np.arange(n_datasets)
    bar_width = 0.18
    
    for ax_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[ax_idx]
        
        # First, compute ranks for each dataset
        dataset_keys_list = list(all_data.keys())
        ranks_per_dataset = {}  # {dataset_key: {model: rank}}
        
        for dataset_key in dataset_keys_list:
            models_data = all_data[dataset_key]['models']
            # Get values for all models
            model_values = []
            for model in BASELINE_MODELS:
                if model in models_data:
                    val = models_data[model].get(metric)
                    if val is not None:
                        model_values.append((model, val))
            # Sort by value (lower is better for MSE)
            model_values.sort(key=lambda x: x[1])
            ranks_per_dataset[dataset_key] = {m: rank + 1 for rank, (m, _) in enumerate(model_values)}
        
        # Now plot bars with rank annotations
        for model_idx, model in enumerate(BASELINE_MODELS):
            values = []
            for dataset_key in dataset_keys_list:
                models = all_data[dataset_key]['models']
                if model in models:
                    val = models[model].get(metric)
                    values.append(val if val is not None else np.nan)
                else:
                    values.append(np.nan)
            
            offset = (model_idx - len(BASELINE_MODELS) / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, values, bar_width, 
                         label=model, color=BASELINE_COLORS[model_idx],
                         alpha=0.85, edgecolor='white', linewidth=0.5)
            
            # Add rank annotations (just number, no #)
            for bar, dataset_key in zip(bars, dataset_keys_list):
                if not np.isnan(bar.get_height()):
                    rank = ranks_per_dataset.get(dataset_key, {}).get(model)
                    if rank:
                        ax.annotate(f'#{rank}',
                                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                   xytext=(0, 2), textcoords='offset points',
                                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                                   color='#333333')
        
        ax.set_ylabel(f'{label} MSE', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_titles, rotation=0, ha='center', fontsize=10)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, linestyle='--', color='gray')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.tick_params(axis='y', labelsize=10)
        
        # Extend y-axis to make room for rank labels
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin * 0.8, ymax * 3)
        
        # Only show x-label on bottom subplot
        if ax_idx < n_metrics - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Dataset', fontsize=11)
    
    # Add single legend at top of first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc='upper right', ncol=4,
                   fontsize=9, frameon=True, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


def generate_baseline_stacked_plot(
    dataset_keys: List[str],
    metrics: List[str],
    metric_labels: List[str],
    title: str,
    output_path: Path,
    outputs_dir: Path
):
    """Generate stacked bar plot for baseline comparison (like AEMET_category_comparison)."""
    
    # Collect data from all datasets
    all_data = {}
    for dataset_key in dataset_keys:
        data = get_baseline_models_data(dataset_key, outputs_dir)
        if data:
            all_data[dataset_key] = {
                'title': DATASETS[dataset_key]['title'],
                'models': data
            }
    
    if not all_data:
        print(f"No baseline data found for plot: {output_path}")
        return
    
    dataset_keys_with_data = list(all_data.keys())
    dataset_titles = [all_data[k]['title'] for k in dataset_keys_with_data]
    n_datasets = len(dataset_titles)
    n_metrics = len(metrics)
    
    fig, ax = plt.subplots(figsize=(max(10, n_datasets * 2), 6))
    
    x = np.arange(n_datasets)
    bar_width = 0.18
    
    # Compute ranks for each metric
    def compute_ranks_for_metric(metric_key):
        ranks = {}
        for dataset_key in dataset_keys_with_data:
            models = all_data[dataset_key]['models']
            model_values = []
            for model in BASELINE_MODELS:
                if model in models:
                    val = models[model].get(metric_key)
                    if val is not None:
                        model_values.append((model, val))
            
            # Sort by value (lower is better for MSE)
            model_values.sort(key=lambda x: x[1])
            for rank, (model, _) in enumerate(model_values, 1):
                if dataset_key not in ranks:
                    ranks[dataset_key] = {}
                ranks[dataset_key][model] = rank
        return ranks
    
    all_values = []
    
    for metric_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ranks = compute_ranks_for_metric(metric)
        
        for model_idx, model in enumerate(BASELINE_MODELS):
            values = []
            for dataset_key in dataset_keys_with_data:
                models_data = all_data[dataset_key]['models']
                if model in models_data:
                    val = models_data[model].get(metric)
                    values.append(val if val is not None else np.nan)
                else:
                    values.append(np.nan)
            
            all_values.extend([v for v in values if not np.isnan(v)])
            
            # Position: group by dataset, then by metric, then by model
            group_offset = metric_idx * (len(BASELINE_MODELS) * bar_width + 0.1)
            model_offset = model_idx * bar_width
            positions = x * (n_metrics * len(BASELINE_MODELS) * bar_width + 0.5) + group_offset + model_offset
            
            bars = ax.bar(positions, values, bar_width,
                         label=model if metric_idx == 0 else '',
                         color=BASELINE_COLORS[model_idx],
                         alpha=0.85, edgecolor='white', linewidth=0.5)
            
            # Add rank labels
            for bar, dataset_key in zip(bars, dataset_keys_with_data):
                if not np.isnan(bar.get_height()):
                    rank = ranks.get(dataset_key, {}).get(model)
                    if rank:
                        ax.annotate(f'#{rank}',
                                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                   xytext=(0, 3), textcoords='offset points',
                                   ha='center', va='bottom', fontsize=7, fontweight='bold',
                                   color='#333333')
    
    # Set up x-axis - one tick per dataset
    tick_positions = []
    for i in range(n_datasets):
        center = i * (n_metrics * len(BASELINE_MODELS) * bar_width + 0.5) + (n_metrics * len(BASELINE_MODELS) * bar_width) / 2 - bar_width / 2
        tick_positions.append(center)
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(dataset_titles, rotation=30, ha='right', fontsize=10)
    
    ax.set_ylabel('MSE', fontsize=11)
    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_yscale('log')
    
    # Adjust y-axis limits
    if all_values:
        min_val = min(all_values)
        max_val = max(all_values)
        ax.set_ylim(min_val * 0.3, max_val * 10)
    
    ax.grid(True, alpha=0.3, linestyle='--', color='gray', axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add metric labels at top
    # This is complex, so skip for now and just use legend
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(BASELINE_MODELS)], labels[:len(BASELINE_MODELS)],
             loc='upper center', bbox_to_anchor=(0.5, -0.15),
             ncol=len(BASELINE_MODELS), fontsize=10, frameon=False)
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")


def generate_all_baseline_plots(outputs_dir: Path = None, plots_output_dir: Path = None):
    """Generate all baseline comparison plots."""
    if outputs_dir is None:
        script_dir = Path(__file__).parent
        outputs_dir = script_dir.parent / 'outputs'
    
    if plots_output_dir is None:
        plots_output_dir = outputs_dir
    
    print("=" * 60)
    print("Generating Baseline Comparison Plots")
    print("=" * 60)
    
    # Sequence datasets - individual metric subplots
    print("\n1. Sequence datasets baseline comparison (subplots)...")
    generate_baseline_comparison_plot(
        dataset_keys=SEQUENCE_DATASETS,
        metrics=BASELINE_METRICS_SEQ,
        metric_labels=BASELINE_LABELS_SEQ,
        title='Baseline Comparison: Sequence Datasets',
        output_path=plots_output_dir / 'baseline_comparison_seq.png',
        outputs_dir=outputs_dir
    )
    
    # PDE datasets - individual metric subplots
    print("\n2. PDE datasets baseline comparison (subplots)...")
    generate_baseline_comparison_plot(
        dataset_keys=PDE_DATASETS,
        metrics=BASELINE_METRICS_PDE,
        metric_labels=BASELINE_LABELS_PDE,
        title='Baseline Comparison: PDE Datasets',
        output_path=plots_output_dir / 'baseline_comparison_pde.png',
        outputs_dir=outputs_dir
    )
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate category comparison plots for time series datasets'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        choices=list(DATASETS.keys()) + ['all'],
        default='all',
        help='Dataset to process (default: all)'
    )
    parser.add_argument(
        '--combined', '-c',
        action='store_true',
        help='Generate a combined plot with all datasets'
    )
    parser.add_argument(
        '--outputs-dir', '-o',
        type=str,
        default=None,
        help='Path to outputs directory'
    )
    parser.add_argument(
        '--plots-dir', '-p',
        type=str,
        default=None,
        help='Path to save plots (default: same as outputs-dir)'
    )
    parser.add_argument(
        '--ignore-gaussian',
        action='store_true',
        default=True,
        help='Ignore Gaussian kernel category (default: True)'
    )
    parser.add_argument(
        '--include-gaussian',
        action='store_true',
        default=False,
        help='Include Gaussian kernel category (overrides --ignore-gaussian)'
    )
    parser.add_argument(
        '--baseline',
        action='store_true',
        help='Generate baseline comparison plots (DDPM, NCSN, FFM, k-FFM)'
    )
    parser.add_argument(
        '--loss-curves',
        action='store_true',
        help='Generate training loss curve plots (best config per kernel category)'
    )
    parser.add_argument(
        '--loss-curves-combined',
        action='store_true',
        help='Generate combined 5x2 loss curves plot (all datasets except moGP, high resolution)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for high resolution plots (default: 300)'
    )
    
    args = parser.parse_args()
    
    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else None
    plots_dir = Path(args.plots_dir) if args.plots_dir else None
    
    # --include-gaussian overrides --ignore-gaussian
    ignore_gaussian = not args.include_gaussian
    
    if args.loss_curves_combined:
        # Generate combined 5x2 loss curves plot
        generate_combined_loss_curves_plot(outputs_dir, plots_dir, ignore_gaussian=ignore_gaussian, dpi=args.dpi)
    elif args.loss_curves:
        # Generate individual loss curve plots
        if args.dataset == 'all':
            generate_loss_curve_plots(outputs_dir, plots_dir, ignore_gaussian=ignore_gaussian)
        else:
            generate_loss_curve_plots(outputs_dir, plots_dir, ignore_gaussian=ignore_gaussian, dataset_key=args.dataset)
    elif args.baseline:
        generate_all_baseline_plots(outputs_dir, plots_dir)
    elif args.combined:
        generate_combined_plot(outputs_dir, plots_dir, ignore_gaussian=ignore_gaussian)
    elif args.dataset == 'all':
        generate_all_plots(outputs_dir, plots_dir, ignore_gaussian=ignore_gaussian)
    else:
        generate_single_plot(args.dataset, outputs_dir, plots_dir, ignore_gaussian=ignore_gaussian)

