"""
Plot hyperparameter sensitivity using violin plots.

For each dataset, creates a violin plot showing the distribution of mean_mse
across all hyperparameter configurations for each kernel type.

Usage:
    python plot_hyperparam_sensitivity.py
    python plot_hyperparam_sensitivity.py --dataset aemet
    python plot_hyperparam_sensitivity.py --output_dir ../outputs/sensitivity_plots
"""

import sys
sys.path.append('../')

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# =============================================================================
# Configuration
# =============================================================================

# Dataset categories
PDE_DATASETS = ["kdv", "navier_stokes", "stochastic_kdv", "stochastic_ns"]
SEQUENCE_DATASETS = ["aemet", "expr_genes", "economy", "heston", "rbergomi"]
ALL_DATASETS = PDE_DATASETS + SEQUENCE_DATASETS

# Dataset to output directory mapping
DATASET_OUTPUT_DIRS = {
    "kdv": "kdv_ot",
    "navier_stokes": "navier_stokes_ot",
    "stochastic_kdv": "stochastic_kdv_ot",
    "stochastic_ns": "stochastic_ns_ot",
    "aemet": "AEMET_ot_comprehensive",
    "expr_genes": "expr_genes_ot_comprehensive",
    "economy": "econ_ot_comprehensive",
    "heston": "Heston_ot_kappa1.0",
    "rbergomi": "rBergomi_ot_H0p10",
}

# Pretty names for display
DATASET_DISPLAY_NAMES = {
    "kdv": "KdV",
    "navier_stokes": "Navier-Stokes",
    "stochastic_kdv": "Stochastic KdV",
    "stochastic_ns": "Stochastic NS",
    "aemet": "AEMET",
    "expr_genes": "Gene Expression",
    "economy": "Economy",
    "heston": "Heston",
    "rbergomi": "rBergomi",
}

KERNEL_DISPLAY_NAMES = {
    "none": "None\n(Independent)",
    "euclidean": "Euclidean",
    "rbf": "RBF",
    "signature": "Signature",
}

KERNEL_COLORS = {
    "none": "#7f7f7f",      # gray
    "euclidean": "#1f77b4",  # blue
    "rbf": "#ff7f0e",        # orange
    "signature": "#2ca02c",  # green
}

# =============================================================================
# Data Loading Functions
# =============================================================================

def categorize_config(config_name: str, config_data: Dict) -> Optional[str]:
    """
    Determine which kernel category a config belongs to.
    Returns: 'none', 'signature', 'rbf', 'euclidean', or None if not categorizable.
    """
    # Skip baselines
    if config_name in ['DDPM', 'NCSN']:
        return None
    
    # Independent = no OT
    if config_name == 'independent' or not config_data.get('use_ot', False):
        return 'none'
    
    # Skip gaussian OT (different method)
    if config_data.get('ot_method') == 'gaussian':
        return None
    
    # Use ot_kernel field (most reliable)
    ot_kernel = config_data.get('ot_kernel', '').lower()
    if ot_kernel == 'signature':
        return 'signature'
    elif ot_kernel == 'rbf':
        return 'rbf'
    elif ot_kernel == 'euclidean':
        return 'euclidean'
    
    # Fallback: infer from config name
    name_lower = config_name.lower()
    if 'signature' in name_lower or name_lower.startswith('sig_'):
        return 'signature'
    elif 'rbf' in name_lower:
        return 'rbf'
    elif 'euclidean' in name_lower:
        return 'euclidean'
    elif 'independent' in name_lower:
        return 'none'
    
    return None


def load_dataset_results(dataset: str, outputs_dir: Path) -> Dict[str, List[float]]:
    """
    Load all experiment results for a dataset and organize by kernel category.
    
    Returns: Dict[kernel_category, List[mean_mse_values]]
    """
    dataset_dir = outputs_dir / DATASET_OUTPUT_DIRS.get(dataset, dataset)
    
    if not dataset_dir.exists():
        print(f"  Warning: Directory not found: {dataset_dir}")
        return {}
    
    # Collect all configs
    all_configs = {}
    
    # Search patterns for result files
    search_dirs = [dataset_dir]
    
    # For economy, also search subdirectories
    if dataset == "economy":
        for subdir in ["econ1_population", "econ2_gdp", "econ3_labor"]:
            subpath = dataset_dir / subdir
            if subpath.exists():
                search_dirs.append(subpath)
    
    for search_dir in search_dirs:
        # Try aggregated_results files
        for agg_file in search_dir.glob("aggregated_results*.json"):
            try:
                with open(agg_file, 'r') as f:
                    data = json.load(f)
                if 'configs' in data:
                    for name, cfg in data['configs'].items():
                        if name not in all_configs:
                            if 'metrics' in cfg:
                                all_configs[name] = cfg['metrics']
                            else:
                                all_configs[name] = cfg
            except (json.JSONDecodeError, IOError) as e:
                print(f"  Warning: Could not load {agg_file}: {e}")
                continue
        
        # Try comprehensive_metrics.json
        metrics_file = search_dir / "comprehensive_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                if 'configs' in data:
                    cfg_list = data['configs']
                    if isinstance(cfg_list, list):
                        for cfg in cfg_list:
                            name = cfg.get('config_name', '')
                            # Remove dataset prefix if present
                            for prefix in ['econ1_population_', 'econ2_gdp_', 'econ3_labor_']:
                                if name.startswith(prefix):
                                    name = name[len(prefix):]
                                    break
                            if name and name not in all_configs:
                                all_configs[name] = cfg
                    elif isinstance(cfg_list, dict):
                        for name, cfg in cfg_list.items():
                            if name not in all_configs:
                                all_configs[name] = cfg
            except (json.JSONDecodeError, IOError) as e:
                print(f"  Warning: Could not load {metrics_file}: {e}")
        
        # Also search for individual quality_metrics.json files in subdirectories
        for qm_file in search_dir.glob("*/quality_metrics.json"):
            try:
                with open(qm_file, 'r') as f:
                    cfg = json.load(f)
                name = cfg.get('config_name', qm_file.parent.name)
                if name not in all_configs:
                    all_configs[name] = cfg
            except (json.JSONDecodeError, IOError):
                continue
    
    # Organize by kernel category
    kernel_results = defaultdict(list)
    
    for config_name, config_data in all_configs.items():
        category = categorize_config(config_name, config_data)
        if category is None:
            continue
        
        # Get mean_mse value
        mean_mse = config_data.get('mean_mse')
        if mean_mse is None or not isinstance(mean_mse, (int, float)):
            continue
        
        # Skip extreme outliers (likely failed runs)
        if mean_mse > 1e10 or mean_mse < 0:
            continue
        
        kernel_results[category].append(mean_mse)
    
    return dict(kernel_results)


# =============================================================================
# Plotting Functions
# =============================================================================

def create_violin_plot(
    kernel_data: Dict[str, List[float]],
    dataset: str,
    output_path: Path,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Create a violin plot for a single dataset.
    """
    # Order kernels consistently
    kernel_order = ['none', 'euclidean', 'rbf', 'signature']
    
    # Filter to kernels that have data
    kernels = [k for k in kernel_order if k in kernel_data and len(kernel_data[k]) > 0]
    
    if not kernels:
        print(f"  No data found for {dataset}")
        return False
    
    # Prepare data
    data = [kernel_data[k] for k in kernels]
    positions = list(range(len(kernels)))
    colors = [KERNEL_COLORS[k] for k in kernels]
    labels = [KERNEL_DISPLAY_NAMES[k] for k in kernels]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create violin plot
    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
    
    # Customize colors
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    # Customize mean and median lines
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)
    parts['cbars'].set_color('black')
    parts['cmaxes'].set_color('black')
    parts['cmins'].set_color('black')
    
    # Add scatter points for individual values
    for i, (k, vals) in enumerate(zip(kernels, data)):
        # Add jitter
        jitter = np.random.uniform(-0.1, 0.1, len(vals))
        ax.scatter(
            [i + j for j in jitter], 
            vals, 
            c=colors[i], 
            alpha=0.4, 
            s=20, 
            edgecolors='none'
        )
    
    # Customize axes
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Mean MSE', fontsize=12)
    ax.set_xlabel('Kernel Type', fontsize=12)
    
    # Title
    display_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    ax.set_title(f'Hyperparameter Sensitivity: {display_name}', fontsize=14, fontweight='bold')
    
    # Add count annotations
    for i, (k, vals) in enumerate(zip(kernels, data)):
        ax.annotate(
            f'n={len(vals)}',
            xy=(i, ax.get_ylim()[1]),
            ha='center',
            va='bottom',
            fontsize=9,
            color='gray'
        )
    
    # Log scale if values span multiple orders of magnitude
    all_vals = [v for vals in data for v in vals]
    if len(all_vals) > 0:
        val_range = max(all_vals) / (min(all_vals) + 1e-10)
        if val_range > 100:
            ax.set_yscale('log')
            ax.set_ylabel('Mean MSE (log scale)', fontsize=12)
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Legend for mean/median
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Mean'),
        Line2D([0], [0], color='black', linewidth=1.5, label='Median'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def create_combined_plot(
    all_data: Dict[str, Dict[str, List[float]]],
    output_path: Path,
    dataset_type: str = 'all',
):
    """
    Create a combined plot with subplots for multiple datasets.
    """
    if dataset_type == 'pde':
        datasets = [d for d in PDE_DATASETS if d in all_data]
        title = 'Hyperparameter Sensitivity: PDE Datasets'
    elif dataset_type == 'sequence':
        datasets = [d for d in SEQUENCE_DATASETS if d in all_data]
        title = 'Hyperparameter Sensitivity: Sequence Datasets'
    else:
        datasets = [d for d in ALL_DATASETS if d in all_data]
        title = 'Hyperparameter Sensitivity: All Datasets'
    
    if not datasets:
        print(f"  No data for {dataset_type} datasets")
        return False
    
    # Determine grid size
    n_datasets = len(datasets)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    kernel_order = ['none', 'euclidean', 'rbf', 'signature']
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        kernel_data = all_data[dataset]
        
        # Filter to kernels that have data
        kernels = [k for k in kernel_order if k in kernel_data and len(kernel_data[k]) > 0]
        
        if not kernels:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(DATASET_DISPLAY_NAMES.get(dataset, dataset))
            continue
        
        data = [kernel_data[k] for k in kernels]
        positions = list(range(len(kernels)))
        colors = [KERNEL_COLORS[k] for k in kernels]
        labels = [KERNEL_DISPLAY_NAMES[k].replace('\n', ' ') for k in kernels]
        
        # Create violin plot
        parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        
        parts['cmeans'].set_color('red')
        parts['cmeans'].set_linewidth(2)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)
        parts['cbars'].set_color('black')
        parts['cmaxes'].set_color('black')
        parts['cmins'].set_color('black')
        
        # Scatter points
        for i, vals in enumerate(data):
            jitter = np.random.uniform(-0.1, 0.1, len(vals))
            ax.scatter([i + j for j in jitter], vals, c=colors[i], alpha=0.3, s=15, edgecolors='none')
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Mean MSE', fontsize=10)
        ax.set_title(DATASET_DISPLAY_NAMES.get(dataset, dataset), fontsize=11, fontweight='bold')
        
        # Log scale if needed
        all_vals = [v for vals in data for v in vals]
        if len(all_vals) > 0:
            val_range = max(all_vals) / (min(all_vals) + 1e-10)
            if val_range > 100:
                ax.set_yscale('log')
        
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    # Hide unused subplots
    for idx in range(len(datasets), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def create_5x2_combined_plot(
    all_data: Dict[str, Dict[str, List[float]]],
    output_path: Path,
    dpi: int = 300,
):
    """
    Create a 4x2 combined violin plot for all datasets (excluding moGP).
    High resolution for full-page figures.
    """
    # Order datasets: sequence first, then PDE (8 datasets for 4x2)
    # Note: expr_genes excluded (missing data)
    datasets_order = [
        # Row 1: Sequence datasets
        'aemet', 'economy',
        # Row 2: Sequence datasets
        'heston', 'rbergomi',
        # Row 3: PDE datasets
        'kdv', 'navier_stokes',
        # Row 4: PDE datasets
        'stochastic_kdv', 'stochastic_ns',
    ]
    
    # Filter to datasets with data
    datasets_with_data = [d for d in datasets_order if d and d in all_data]
    
    if not datasets_with_data:
        print("  No data for 4x2 combined plot")
        return False
    
    # Create 4x2 figure
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    axes = axes.flatten()
    
    kernel_order = ['none', 'euclidean', 'rbf', 'signature']
    
    for idx, dataset in enumerate(datasets_order):
        ax = axes[idx]
        
        if dataset is None or dataset not in all_data:
            ax.set_visible(False)
            continue
        
        kernel_data = all_data[dataset]
        
        # Filter to kernels that have data
        kernels = [k for k in kernel_order if k in kernel_data and len(kernel_data[k]) > 0]
        
        if not kernels:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(DATASET_DISPLAY_NAMES.get(dataset, dataset), fontsize=12, fontweight='bold')
            continue
        
        data = [kernel_data[k] for k in kernels]
        positions = list(range(len(kernels)))
        colors = [KERNEL_COLORS[k] for k in kernels]
        labels = [KERNEL_DISPLAY_NAMES[k].replace('\n', ' ') for k in kernels]
        
        # Create violin plot
        parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        
        parts['cmeans'].set_color('red')
        parts['cmeans'].set_linewidth(2)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)
        parts['cbars'].set_color('black')
        parts['cmaxes'].set_color('black')
        parts['cmins'].set_color('black')
        
        # Scatter points
        for i, vals in enumerate(data):
            jitter = np.random.uniform(-0.1, 0.1, len(vals))
            ax.scatter([i + j for j in jitter], vals, c=colors[i], alpha=0.3, s=12, edgecolors='none')
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Mean MSE', fontsize=10)
        ax.set_title(DATASET_DISPLAY_NAMES.get(dataset, dataset), fontsize=12, fontweight='bold')
        
        # Log scale if needed
        all_vals = [v for vals in data for v in vals]
        if len(all_vals) > 0:
            val_range = max(all_vals) / (min(all_vals) + 1e-10)
            if val_range > 100:
                ax.set_yscale('log')
        
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add count annotations
        for i, (k, vals) in enumerate(zip(kernels, data)):
            ymax = ax.get_ylim()[1]
            ax.annotate(
                f'n={len(vals)}',
                xy=(i, ymax),
                ha='center',
                va='bottom',
                fontsize=8,
                color='gray'
            )
    
    # Hide unused subplots
    for idx in range(len(datasets_order), len(axes)):
        axes[idx].set_visible(False)
    
    # Add legend at bottom
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_elements = [
        Patch(facecolor=KERNEL_COLORS['none'], edgecolor='black', alpha=0.7, label='None (Independent)'),
        Patch(facecolor=KERNEL_COLORS['euclidean'], edgecolor='black', alpha=0.7, label='Euclidean'),
        Patch(facecolor=KERNEL_COLORS['rbf'], edgecolor='black', alpha=0.7, label='RBF'),
        Patch(facecolor=KERNEL_COLORS['signature'], edgecolor='black', alpha=0.7, label='Signature'),
        Line2D([0], [0], color='red', linewidth=2, label='Mean'),
        Line2D([0], [0], color='black', linewidth=1.5, label='Median'),
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=10,
               bbox_to_anchor=(0.5, 0.01), frameon=True, framealpha=0.95)
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])  # Leave room for legend
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Plot hyperparameter sensitivity using violin plots'
    )
    parser.add_argument('--dataset', type=str, default=None,
                        help='Specific dataset to plot (default: all)')
    parser.add_argument('--outputs_dir', type=str, default='../outputs',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='../outputs/sensitivity_plots',
                        help='Directory to save plots')
    parser.add_argument('--combined-only', action='store_true',
                        help='Only generate combined plots')
    parser.add_argument('--full-page', action='store_true',
                        help='Generate 5x2 full-page combined plot (high resolution)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for high resolution plots (default: 300)')
    
    args = parser.parse_args()
    
    outputs_dir = Path(args.outputs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Hyperparameter Sensitivity Plots")
    print("=" * 60)
    
    # Determine which datasets to process
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = ALL_DATASETS
    
    # Load all data
    all_data = {}
    for dataset in datasets:
        print(f"\nLoading {dataset}...")
        kernel_data = load_dataset_results(dataset, outputs_dir)
        
        if kernel_data:
            all_data[dataset] = kernel_data
            for kernel, values in kernel_data.items():
                print(f"  {kernel}: {len(values)} configs, "
                      f"mean_mse range: [{min(values):.2e}, {max(values):.2e}]")
        else:
            print(f"  No data found")
    
    if not all_data:
        print("\nNo data found for any dataset!")
        return
    
    # Generate 4x2 full-page plot if requested
    if args.full_page:
        print("\nGenerating 4x2 full-page combined plot...")
        fullpage_path = output_dir / "sensitivity_4x2_combined.png"
        if create_5x2_combined_plot(all_data, fullpage_path, dpi=args.dpi):
            print(f"  ✓ Saved: {fullpage_path} (dpi={args.dpi})")
            print(f"  ✓ Saved: {fullpage_path.with_suffix('.pdf')}")
        print("\nDone!")
        return
    
    # Generate individual plots
    if not args.combined_only:
        print("\nGenerating individual plots...")
        for dataset in all_data:
            output_path = output_dir / f"sensitivity_{dataset}.png"
            success = create_violin_plot(all_data[dataset], dataset, output_path)
            if success:
                print(f"  Saved: {output_path}")
    
    # Generate combined plots
    print("\nGenerating combined plots...")
    
    # PDE datasets
    pde_path = output_dir / "sensitivity_pde_combined.png"
    if create_combined_plot(all_data, pde_path, 'pde'):
        print(f"  Saved: {pde_path}")
    
    # Sequence datasets
    seq_path = output_dir / "sensitivity_sequence_combined.png"
    if create_combined_plot(all_data, seq_path, 'sequence'):
        print(f"  Saved: {seq_path}")
    
    # All datasets
    all_path = output_dir / "sensitivity_all_combined.png"
    if create_combined_plot(all_data, all_path, 'all'):
        print(f"  Saved: {all_path}")
    
    print("\nDone!")
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
