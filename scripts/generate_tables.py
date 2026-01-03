#!/usr/bin/env python3
"""
Generate LaTeX tables showing best performance per kernel category
for all datasets.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def compute_convergence_rate(train_losses: List[float]) -> Optional[float]:
    """
    Compute convergence rate from training loss history.
    
    Convergence rate = (log(loss_0) - log(loss_half)) / half_epochs
    
    This measures the average logarithmic decrease in loss during the first half 
    of training. Higher values indicate faster convergence.
    
    Parameters
    ----------
    train_losses : List[float]
        Training loss values per epoch
        
    Returns
    -------
    float or None
        Convergence rate, or None if cannot be computed
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
    
    Looks for training_metrics.json files in seed subdirectories and computes
    convergence rate from the train_losses array.
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

# Metrics to include in the table
# Full set of all metrics
METRICS_ALL = ['mean_mse', 'variance_mse', 'kurtosis_mse', 'skewness_mse', 'autocorrelation_mse', 'convergence_rate', 'spectrum_mse']
METRIC_HEADERS_ALL = ['Mean', 'Variance', 'Kurtosis', 'Skewness', 'Autocorr.', 'Conv. Rate', 'Spectrum']

# Sequence datasets: no spectrum (not applicable)
METRICS_SEQUENCE = ['mean_mse', 'variance_mse', 'kurtosis_mse', 'skewness_mse', 'autocorrelation_mse', 'convergence_rate']
METRIC_HEADERS_SEQUENCE = ['Mean', 'Variance', 'Kurtosis', 'Skewness', 'Autocorr.', 'Conv. Rate']

# PDE datasets: no autocorrelation, kurtosis, or skewness (not applicable/meaningful for 2D fields)
METRICS_PDE = ['mean_mse', 'variance_mse', 'convergence_rate', 'spectrum_mse']
METRIC_HEADERS_PDE = ['Mean', 'Variance', 'Conv. Rate', 'Spectrum']

# Default for backwards compatibility
METRICS = METRICS_ALL
METRIC_HEADERS = METRIC_HEADERS_ALL

# Dataset configurations
DATASETS = {
    # Time series / Sequence datasets
    'AEMET': {
        'dir': 'AEMET_ot_comprehensive',
        'title': 'AEMET',
        'summary_files': ['comprehensive_metrics.json', 'experiment_summary.json'],
        'category': 'sequence',
    },
    'expr_genes': {
        'dir': 'expr_genes_ot_comprehensive', 
        'title': 'Gene Expr.',
        'summary_files': ['comprehensive_metrics.json', 'experiment_summary.json'],
        'category': 'sequence',
    },
    'econ1': {
        'dir': 'econ_ot_comprehensive/econ1_population',
        'title': 'Economy',
        'summary_files': ['comprehensive_metrics.json'],
        'category': 'sequence',
    },
    'Heston': {
        'dir': 'Heston_ot_kappa1.0',
        'title': 'Heston',
        'summary_files': ['comprehensive_metrics.json', 'experiment_summary.json'],
        'category': 'sequence',
    },
    'rBergomi': {
        'dir': 'rBergomi_ot_H0p10',
        'title': 'rBergomi',
        'summary_files': ['comprehensive_metrics.json', 'experiment_summary.json'],
        'category': 'sequence',
    },
    'moGP': {
        'dir': 'moGP_ot_comprehensive',
        'title': 'Multi-GP',
        'summary_files': ['comprehensive_metrics.json', 'experiment_summary.json'],
        'category': 'sequence',
    },
    # PDE datasets
    'kdv': {
        'dir': 'kdv_ot',
        'title': 'KdV',
        'summary_files': ['comprehensive_metrics.json', 'experiment_summary.json'],
        'category': 'pde',
    },
    'navier_stokes': {
        'dir': 'navier_stokes_ot',
        'title': '\\shortstack{Navier-\\\\Stokes}',
        'summary_files': ['comprehensive_metrics.json', 'experiment_summary.json'],
        'category': 'pde',
    },
    'stochastic_kdv': {
        'dir': 'stochastic_kdv_ot',
        'title': '\\shortstack{Stoch.\\\\KdV}',
        'summary_files': ['comprehensive_metrics.json', 'experiment_summary.json'],
        'category': 'pde',
    },
    'ginzburg_landau': {
        'dir': 'ginzburg_landau_ot',
        'title': '\\shortstack{Ginzburg-\\\\Landau}',
        'summary_files': ['comprehensive_metrics.json', 'experiment_summary.json'],
        'category': 'pde',
    },
    'stochastic_ns': {
        'dir': 'stochastic_ns_ot',
        'title': 'Stoch. NS',
        'summary_files': ['comprehensive_metrics.json', 'experiment_summary.json'],
        'category': 'pde',
    },
}

# Dataset groups
SEQUENCE_DATASETS = ['AEMET', 'expr_genes', 'econ1', 'Heston', 'rBergomi', 'moGP']
PDE_DATASETS = ['kdv', 'navier_stokes', 'stochastic_kdv', 'ginzburg_landau', 'stochastic_ns']


def normalize_configs(configs) -> Dict:
    """
    Normalize configs to a dict format.
    Handles both dict format (experiment_summary.json) and list format (comprehensive_metrics.json).
    """
    if isinstance(configs, dict):
        return configs
    elif isinstance(configs, list):
        normalized = {}
        for item in configs:
            config_name = item.get('config_name', '')
            # Remove common prefixes
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
    Best is determined by lowest average MSE across core metrics.
    """
    configs = normalize_configs(configs)
    best_per_category = {}
    
    # Core metrics for determining "best" (exclude convergence_rate and spectrum)
    core_metrics = ['mean_mse', 'variance_mse', 'kurtosis_mse', 'skewness_mse', 'autocorrelation_mse']
    
    for category_name, config_names in categories.items():
        best_config = None
        best_avg_mse = float('inf')
        
        for config_name in config_names:
            if config_name in configs:
                config_data = configs[config_name]
                
                if 'quality_metrics' in config_data:
                    metrics = config_data['quality_metrics']
                else:
                    metrics = config_data
                
                mse_values = []
                for metric in core_metrics:
                    val = metrics.get(metric)
                    if val is not None:
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


def format_value(value, precision: int = 2) -> str:
    """Format a value for LaTeX table using consistent scientific notation (a.aa × 10^{-b})."""
    if value is None:
        return '--'
    
    if value == 0:
        return '$0$'
    
    abs_val = abs(value)
    
    # Always use scientific notation: a.aa × 10^{b}
    exp = int(np.floor(np.log10(abs_val)))
    mantissa = value / (10 ** exp)
    
    # Format mantissa with 2 decimal places
    return f'${mantissa:.{precision}f} \\times 10^{{{exp}}}$'


def load_dataset_data(
    dataset_key: str, 
    outputs_dir: Path, 
    compute_missing_convergence: bool = True,
    ignore_gaussian: bool = True
) -> Optional[Dict]:
    """Load data for a single dataset.
    
    Parameters
    ----------
    dataset_key : str
        Key into DATASETS dict
    outputs_dir : Path
        Base outputs directory
    compute_missing_convergence : bool
        If True and convergence_rate is missing, attempt to compute it from 
        training_metrics.json files in the config directories
    ignore_gaussian : bool
        If True, exclude Gaussian kernel category from results
    """
    dataset_info = DATASETS[dataset_key]
    dataset_dir = outputs_dir / dataset_info['dir']
    
    if not dataset_dir.exists():
        return None
    
    summary_files = dataset_info.get('summary_files', ['comprehensive_metrics.json', 'experiment_summary.json'])
    
    for summary_file in summary_files:
        summary_path = dataset_dir / summary_file
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            break
    else:
        return None
    
    config_key = dataset_info.get('config_key', None)
    if config_key:
        if config_key not in summary:
            return None
        configs = summary[config_key]
    elif 'configs' in summary:
        configs = summary['configs']
    else:
        return None
    
    # Get best configs per category (with optional Gaussian filtering)
    categories = get_kernel_categories(ignore_gaussian=ignore_gaussian)
    best_configs = get_best_config_per_category(configs, categories=categories)
    
    # If requested, try to compute missing convergence rates on-the-fly
    if compute_missing_convergence:
        for category, config_data in best_configs.items():
            metrics = config_data.get('metrics', {})
            if metrics.get('convergence_rate') is None:
                # Try to compute from training_metrics.json
                config_name = config_data.get('name', '')
                config_subdir = dataset_dir / config_name
                
                if config_subdir.exists():
                    conv_rate = compute_convergence_for_config(config_subdir)
                    if conv_rate is not None:
                        metrics['convergence_rate'] = conv_rate
    
    return best_configs


def find_best_per_metric(all_data: Dict) -> Dict[str, Dict[str, str]]:
    """
    Find the best (lowest) value for each metric across all datasets and kernels.
    Returns dict: {metric: {(dataset, kernel): value}}
    """
    best = {metric: {'value': float('inf'), 'keys': []} for metric in METRICS}
    
    for dataset_key, data in all_data.items():
        for kernel, config in data['best_configs'].items():
            metrics = config['metrics']
            for metric in METRICS:
                val = metrics.get(metric)
                if val is not None:
                    # For convergence_rate, higher is better
                    if metric == 'convergence_rate':
                        if val > best[metric]['value'] or best[metric]['value'] == float('inf'):
                            best[metric]['value'] = val
                            best[metric]['keys'] = [(dataset_key, kernel)]
                        elif val == best[metric]['value']:
                            best[metric]['keys'].append((dataset_key, kernel))
                    else:
                        # For MSE metrics, lower is better
                        if val < best[metric]['value']:
                            best[metric]['value'] = val
                            best[metric]['keys'] = [(dataset_key, kernel)]
                        elif val == best[metric]['value']:
                            best[metric]['keys'].append((dataset_key, kernel))
    
    return best


def generate_latex_table(outputs_dir: Path = None, output_file: Path = None, ignore_gaussian: bool = True) -> str:
    """Generate a LaTeX table with all datasets and kernel categories."""
    if outputs_dir is None:
        script_dir = Path(__file__).parent
        outputs_dir = script_dir.parent / 'outputs'
    
    if output_file is None:
        output_file = outputs_dir / 'table1.tex'
    
    # Collect data from all datasets
    all_data = {}
    for dataset_key in DATASETS:
        best_configs = load_dataset_data(dataset_key, outputs_dir, ignore_gaussian=ignore_gaussian)
        if best_configs:
            all_data[dataset_key] = {
                'title': DATASETS[dataset_key]['title'],
                'best_configs': best_configs
            }
    
    if not all_data:
        print("No data found")
        return ""
    
    # Find best values for highlighting
    best_per_metric = find_best_per_metric(all_data)
    
    # Kernel order (Independent first as N/A baseline, then others)
    kernel_order = ['Independent', 'Signature', 'RBF', 'Euclidean']
    if not ignore_gaussian:
        kernel_order.append('Gaussian')
    
    # Build LaTeX table (table* for ICML double-column format)
    lines = []
    
    # Table header
    n_metrics = len(METRIC_HEADERS)
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Performance comparison across datasets and OT kernel types. Best values per dataset are in \textbf{bold}.}')
    lines.append(r'\label{tab:performance_comparison}')
    lines.append(r'\resizebox{\textwidth}{!}{%')
    lines.append(r'\begin{tabular}{ll' + 'r' * n_metrics + '}')
    lines.append(r'\toprule')
    
    # Column headers
    header = 'Dataset & Kernel & ' + ' & '.join(METRIC_HEADERS) + r' \\'
    lines.append(header)
    lines.append(r'\midrule')
    
    # Data rows
    for dataset_key, data in all_data.items():
        dataset_title = data['title']
        best_configs = data['best_configs']
        
        # Find best value for each metric within this dataset
        dataset_best = {}
        for metric in METRICS:
            best_val = float('inf')
            if metric == 'convergence_rate':
                best_val = float('-inf')
            for kernel in kernel_order:
                if kernel in best_configs:
                    val = best_configs[kernel]['metrics'].get(metric)
                    if val is not None:
                        if metric == 'convergence_rate':
                            if val > best_val:
                                best_val = val
                        else:
                            if val < best_val:
                                best_val = val
            dataset_best[metric] = best_val
        
        available_kernels = [k for k in kernel_order if k in best_configs]
        
        for i, kernel in enumerate(available_kernels):
            config = best_configs[kernel]
            metrics = config['metrics']
            
            # Dataset column (multirow for first row)
            if i == 0:
                dataset_col = f'\\multirow{{5}}{{*}}{{{dataset_title}}}'
            else:
                dataset_col = ''
            
            # Kernel column (N/A for Independent)
            kernel_col = 'N/A' if kernel == 'Independent' else kernel
            
            # Metric values
            values = []
            for metric in METRICS:
                val = metrics.get(metric)
                formatted = format_value(val)
                
                # Bold if best in dataset
                if val is not None and val == dataset_best[metric]:
                    formatted = f'\\textbf{{{formatted}}}'
                
                values.append(formatted)
            
            row = f'{dataset_col} & {kernel_col} & ' + ' & '.join(values) + r' \\'
            lines.append(row)
        
        # Add midrule between datasets (except after last)
        if dataset_key != list(all_data.keys())[-1]:
            lines.append(r'\midrule')
    
    # Table footer
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'}')
    lines.append(r'\end{table*}')
    
    # Join and save
    table_content = '\n'.join(lines)
    
    with open(output_file, 'w') as f:
        f.write(table_content)
    
    print(f"Generated: {output_file}")
    return table_content


def generate_subtable(
    outputs_dir: Path,
    dataset_keys: List[str],
    caption: str,
    label: str,
    output_file: Path,
    metrics: List[str] = None,
    metric_headers: List[str] = None,
    ignore_gaussian: bool = True,
    single_column: bool = False
) -> str:
    """Generate a LaTeX table for a specific subset of datasets.
    
    Parameters
    ----------
    outputs_dir : Path
        Path to outputs directory
    dataset_keys : List[str]
        List of dataset keys to include
    caption : str
        LaTeX caption
    label : str
        LaTeX label
    output_file : Path
        Output file path
    metrics : List[str], optional
        List of metric keys to include. Defaults to METRICS_ALL.
    metric_headers : List[str], optional
        List of metric headers for LaTeX. Defaults to METRIC_HEADERS_ALL.
    ignore_gaussian : bool
        If True, exclude Gaussian kernel category from table
    single_column : bool
        If True, generate a single-column table (table instead of table*)
    """
    if metrics is None:
        metrics = METRICS_ALL
    if metric_headers is None:
        metric_headers = METRIC_HEADERS_ALL
    
    # Collect data from specified datasets
    all_data = {}
    for dataset_key in dataset_keys:
        if dataset_key not in DATASETS:
            continue
        best_configs = load_dataset_data(dataset_key, outputs_dir, ignore_gaussian=ignore_gaussian)
        if best_configs:
            all_data[dataset_key] = {
                'title': DATASETS[dataset_key]['title'],
                'best_configs': best_configs
            }
    
    if not all_data:
        print(f"No data found for subtable: {label}")
        return ""
    
    kernel_order = ['Independent', 'Signature', 'RBF', 'Euclidean']
    if not ignore_gaussian:
        kernel_order.append('Gaussian')
    
    # Build LaTeX table
    lines = []
    n_metrics = len(metric_headers)
    
    if single_column:
        # Single-column table (fits in one column of double-column layout)
        lines.append(r'\begin{table}[t]')
        lines.append(r'\centering')
        lines.append(r'\small')
        lines.append(f'\\caption{{{caption}}}')
        lines.append(f'\\label{{{label}}}')
        lines.append(r'\begin{tabular}{ll' + 'r' * n_metrics + '}')
    else:
        # Double-column table (spans full page width)
        lines.append(r'\begin{table*}[t]')
        lines.append(r'\centering')
        lines.append(f'\\caption{{{caption}}}')
        lines.append(f'\\label{{{label}}}')
        lines.append(r'\resizebox{\textwidth}{!}{%')
        lines.append(r'\begin{tabular}{ll' + 'r' * n_metrics + '}')
    lines.append(r'\toprule')
    
    header = 'Dataset & Kernel & ' + ' & '.join(metric_headers) + r' \\'
    lines.append(header)
    lines.append(r'\midrule')
    
    dataset_keys_with_data = [k for k in dataset_keys if k in all_data]
    
    for dataset_key in dataset_keys_with_data:
        data = all_data[dataset_key]
        dataset_title = data['title']
        best_configs = data['best_configs']
        
        # Find best value for each metric within this dataset
        dataset_best = {}
        for metric in metrics:
            best_val = float('inf')
            if metric == 'convergence_rate':
                best_val = float('-inf')
            for kernel in kernel_order:
                if kernel in best_configs:
                    val = best_configs[kernel]['metrics'].get(metric)
                    if val is not None:
                        if metric == 'convergence_rate':
                            if val > best_val:
                                best_val = val
                        else:
                            if val < best_val:
                                best_val = val
            dataset_best[metric] = best_val
        
        available_kernels = [k for k in kernel_order if k in best_configs]
        
        for i, kernel in enumerate(available_kernels):
            config = best_configs[kernel]
            config_metrics = config['metrics']
            
            if i == 0:
                dataset_col = f'\\multirow{{5}}{{*}}{{{dataset_title}}}'
            else:
                dataset_col = ''
            
            kernel_col = 'N/A' if kernel == 'Independent' else kernel
            
            values = []
            for metric in metrics:
                val = config_metrics.get(metric)
                formatted = format_value(val)
                
                if val is not None and val == dataset_best[metric]:
                    formatted = f'\\textbf{{{formatted}}}'
                
                values.append(formatted)
            
            row = f'{dataset_col} & {kernel_col} & ' + ' & '.join(values) + r' \\'
            lines.append(row)
        
        if dataset_key != dataset_keys_with_data[-1]:
            lines.append(r'\midrule')
    
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    if single_column:
        lines.append(r'\end{table}')
    else:
        lines.append(r'}')  # Close resizebox
        lines.append(r'\end{table*}')
    
    table_content = '\n'.join(lines)
    
    with open(output_file, 'w') as f:
        f.write(table_content)
    
    print(f"Generated: {output_file}")
    return table_content


def generate_sequence_table(outputs_dir: Path = None, output_file: Path = None, ignore_gaussian: bool = True) -> str:
    """Generate a LaTeX table for sequence/time-series datasets (no spectrum column)."""
    if outputs_dir is None:
        script_dir = Path(__file__).parent
        outputs_dir = script_dir.parent / 'outputs'
    
    if output_file is None:
        output_file = outputs_dir / 'table_sequence.tex'
    
    return generate_subtable(
        outputs_dir=outputs_dir,
        dataset_keys=SEQUENCE_DATASETS,
        caption='Performance comparison for sequence datasets. Best values per dataset are in \\textbf{bold}.',
        label='tab:sequence_performance',
        output_file=output_file,
        metrics=METRICS_SEQUENCE,
        metric_headers=METRIC_HEADERS_SEQUENCE,
        ignore_gaussian=ignore_gaussian
    )


def generate_pde_table(outputs_dir: Path = None, output_file: Path = None, ignore_gaussian: bool = True) -> str:
    """Generate a LaTeX table for PDE datasets (single-column, fewer metrics)."""
    if outputs_dir is None:
        script_dir = Path(__file__).parent
        outputs_dir = script_dir.parent / 'outputs'
    
    if output_file is None:
        output_file = outputs_dir / 'table_pde.tex'
    
    return generate_subtable(
        outputs_dir=outputs_dir,
        dataset_keys=PDE_DATASETS,
        caption='Performance comparison for PDE datasets. Best values per dataset are in \\textbf{bold}.',
        label='tab:pde_performance',
        output_file=output_file,
        metrics=METRICS_PDE,
        metric_headers=METRIC_HEADERS_PDE,
        ignore_gaussian=ignore_gaussian,
        single_column=True  # PDE table fits in single column
    )


def generate_all_tables(outputs_dir: Path = None, ignore_gaussian: bool = True) -> None:
    """Generate all tables (full, sequence, PDE)."""
    if outputs_dir is None:
        script_dir = Path(__file__).parent
        outputs_dir = script_dir.parent / 'outputs'
    
    print("=" * 60)
    print("Generating LaTeX Tables")
    if ignore_gaussian:
        print("(Ignoring Gaussian kernel category)")
    print("=" * 60)
    
    print("\n1. Full table (all datasets)...")
    generate_latex_table(outputs_dir, ignore_gaussian=ignore_gaussian)
    
    print("\n2. Sequence datasets table...")
    generate_sequence_table(outputs_dir, ignore_gaussian=ignore_gaussian)
    
    print("\n3. PDE datasets table...")
    generate_pde_table(outputs_dir, ignore_gaussian=ignore_gaussian)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


def generate_compact_table(outputs_dir: Path = None, output_file: Path = None, ignore_gaussian: bool = True) -> str:
    """Generate a more compact LaTeX table with only key metrics."""
    if outputs_dir is None:
        script_dir = Path(__file__).parent
        outputs_dir = script_dir.parent / 'outputs'
    
    if output_file is None:
        output_file = outputs_dir / 'table1_compact.tex'
    
    # Collect data
    all_data = {}
    for dataset_key in DATASETS:
        best_configs = load_dataset_data(dataset_key, outputs_dir, ignore_gaussian=ignore_gaussian)
        if best_configs:
            all_data[dataset_key] = {
                'title': DATASETS[dataset_key]['title'],
                'best_configs': best_configs
            }
    
    if not all_data:
        print("No data found")
        return ""
    
    # Compact metrics
    compact_metrics = ['mean_mse', 'variance_mse', 'autocorrelation_mse', 'spectrum_mse']
    compact_headers = ['Mean', 'Variance', 'Autocorr.', 'Spectrum']
    
    kernel_order = ['Independent', 'Signature', 'RBF', 'Euclidean']
    if not ignore_gaussian:
        kernel_order.append('Gaussian')
    
    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Performance comparison (compact).}')
    lines.append(r'\label{tab:performance_compact}')
    lines.append(r'\resizebox{\textwidth}{!}{%')
    lines.append(r'\begin{tabular}{ll' + 'r' * len(compact_headers) + '}')
    lines.append(r'\toprule')
    lines.append('Dataset & Kernel & ' + ' & '.join(compact_headers) + r' \\')
    lines.append(r'\midrule')
    
    for dataset_key, data in all_data.items():
        dataset_title = data['title']
        best_configs = data['best_configs']
        
        available_kernels = [k for k in kernel_order if k in best_configs]
        
        # Find best per metric in this dataset
        dataset_best = {}
        for metric in compact_metrics:
            best_val = float('inf')
            for kernel in available_kernels:
                val = best_configs[kernel]['metrics'].get(metric)
                if val is not None and val < best_val:
                    best_val = val
            dataset_best[metric] = best_val
        
        for i, kernel in enumerate(available_kernels):
            config = best_configs[kernel]
            metrics = config['metrics']
            
            dataset_col = f'\\multirow{{5}}{{*}}{{{dataset_title}}}' if i == 0 else ''
            kernel_col = 'N/A' if kernel == 'Independent' else kernel
            
            values = []
            for metric in compact_metrics:
                val = metrics.get(metric)
                formatted = format_value(val)
                if val is not None and val == dataset_best[metric]:
                    formatted = f'\\textbf{{{formatted}}}'
                values.append(formatted)
            
            lines.append(f'{dataset_col} & {kernel_col} & ' + ' & '.join(values) + r' \\')
        
        if dataset_key != list(all_data.keys())[-1]:
            lines.append(r'\midrule')
    
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'}')
    lines.append(r'\end{table*}')
    
    table_content = '\n'.join(lines)
    
    with open(output_file, 'w') as f:
        f.write(table_content)
    
    print(f"Generated: {output_file}")
    return table_content


# =============================================================================
# Baseline Comparison Tables (DDPM, NCSN, FFM, k-FFM)
# =============================================================================

# Baseline model configurations to look for
BASELINE_MODELS = ['DDPM', 'NCSN']

# Metrics for baseline comparison
BASELINE_METRICS_SEQ = ['mean_mse', 'variance_mse', 'skewness_mse', 'kurtosis_mse', 'autocorrelation_mse']
BASELINE_HEADERS_SEQ = ['Mean', 'Var.', 'Skew.', 'Kurt.', 'Autocorr.']

BASELINE_METRICS_PDE = ['mean_mse', 'variance_mse', 'spectrum_mse']
BASELINE_HEADERS_PDE = ['Mean', 'Variance', 'Spectrum']


def get_baseline_models_data(dataset_key: str, outputs_dir: Path) -> Optional[Dict]:
    """
    Get metrics for baseline models (DDPM, NCSN, FFM, k-FFM) for a dataset.
    
    Returns dict with keys: 'DDPM', 'NCSN', 'FFM', 'k-FFM'
    """
    dataset_info = DATASETS.get(dataset_key)
    if not dataset_info:
        return None
    
    dataset_dir = outputs_dir / dataset_info['dir']
    if not dataset_dir.exists():
        return None
    
    # Load experiment summary
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
    
    # Get DDPM and NCSN baselines
    for model in BASELINE_MODELS:
        if model in configs:
            config_data = configs[model]
            if 'quality_metrics' in config_data:
                result[model] = config_data['quality_metrics']
            else:
                result[model] = config_data
    
    # Get FFM (independent coupling)
    if 'independent' in configs:
        config_data = configs['independent']
        if 'quality_metrics' in config_data:
            result['FFM'] = config_data['quality_metrics']
        else:
            result['FFM'] = config_data
    
    # Get k-FFM (best across all OT kernel types, excluding Gaussian and Independent)
    ot_categories = {
        'Signature': ['signature_sinkhorn_reg0.1', 'signature_sinkhorn_reg0.5', 'signature_sinkhorn_reg1.0'],
        'RBF': ['rbf_exact', 'rbf_sinkhorn_reg0.1', 'rbf_sinkhorn_reg0.5', 'rbf_sinkhorn_reg1.0'],
        'Euclidean': ['euclidean_exact', 'euclidean_sinkhorn_reg0.1', 'euclidean_sinkhorn_reg0.5', 'euclidean_sinkhorn_reg1.0'],
    }
    
    best_ot_config = None
    best_ot_avg_mse = float('inf')
    core_metrics = ['mean_mse', 'variance_mse', 'kurtosis_mse', 'skewness_mse', 'autocorrelation_mse']
    
    for category_name, config_names in ot_categories.items():
        for config_name in config_names:
            if config_name in configs:
                config_data = configs[config_name]
                if 'quality_metrics' in config_data:
                    metrics = config_data['quality_metrics']
                else:
                    metrics = config_data
                
                mse_values = [metrics.get(m) for m in core_metrics if metrics.get(m) is not None]
                if mse_values:
                    avg_mse = np.mean(mse_values)
                    if avg_mse < best_ot_avg_mse:
                        best_ot_avg_mse = avg_mse
                        best_ot_config = metrics
    
    if best_ot_config:
        result['k-FFM'] = best_ot_config
    
    return result if result else None


def generate_baseline_comparison_table(
    outputs_dir: Path,
    dataset_keys: List[str],
    metrics: List[str],
    metric_headers: List[str],
    caption: str,
    label: str,
    output_file: Path,
    single_column: bool = False
) -> str:
    """Generate baseline comparison table (DDPM, NCSN, FFM, k-FFM)."""
    
    # Collect data
    all_data = {}
    for dataset_key in dataset_keys:
        data = get_baseline_models_data(dataset_key, outputs_dir)
        if data:
            all_data[dataset_key] = {
                'title': DATASETS[dataset_key]['title'],
                'models': data
            }
    
    if not all_data:
        print(f"No baseline data found for: {label}")
        return ""
    
    model_order = ['DDPM', 'NCSN', 'FFM', 'k-FFM']
    
    # Build table
    lines = []
    n_metrics = len(metric_headers)
    
    if single_column:
        lines.append(r'\begin{table}[t]')
        lines.append(r'\centering')
        lines.append(r'\small')
        lines.append(f'\\caption{{{caption}}}')
        lines.append(f'\\label{{{label}}}')
        lines.append(r'\begin{tabular}{ll' + 'r' * n_metrics + '}')
    else:
        lines.append(r'\begin{table*}[t]')
        lines.append(r'\centering')
        lines.append(f'\\caption{{{caption}}}')
        lines.append(f'\\label{{{label}}}')
        lines.append(r'\resizebox{\textwidth}{!}{%')
        lines.append(r'\begin{tabular}{ll' + 'r' * n_metrics + '}')
    
    lines.append(r'\toprule')
    header = 'Dataset & Model & ' + ' & '.join(metric_headers) + r' \\'
    lines.append(header)
    lines.append(r'\midrule')
    
    dataset_keys_with_data = [k for k in dataset_keys if k in all_data]
    
    for dataset_key in dataset_keys_with_data:
        data = all_data[dataset_key]
        dataset_title = data['title']
        models = data['models']
        
        # Find best value for each metric within this dataset
        dataset_best = {}
        for metric in metrics:
            best_val = float('inf')
            for model in model_order:
                if model in models:
                    val = models[model].get(metric)
                    if val is not None and val < best_val:
                        best_val = val
            dataset_best[metric] = best_val if best_val != float('inf') else None
        
        available_models = [m for m in model_order if m in models]
        n_models = len(available_models)
        
        for i, model in enumerate(available_models):
            model_metrics = models[model]
            
            if i == 0:
                dataset_col = f'\\multirow{{{n_models}}}{{*}}{{{dataset_title}}}'
            else:
                dataset_col = ''
            
            values = []
            for metric in metrics:
                val = model_metrics.get(metric)
                formatted = format_value(val)
                
                if val is not None and dataset_best.get(metric) is not None and val == dataset_best[metric]:
                    formatted = f'\\textbf{{{formatted}}}'
                
                values.append(formatted)
            
            row = f'{dataset_col} & {model} & ' + ' & '.join(values) + r' \\'
            lines.append(row)
        
        if dataset_key != dataset_keys_with_data[-1]:
            lines.append(r'\midrule')
    
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    
    if single_column:
        lines.append(r'\end{table}')
    else:
        lines.append(r'}')
        lines.append(r'\end{table*}')
    
    table_content = '\n'.join(lines)
    
    with open(output_file, 'w') as f:
        f.write(table_content)
    
    print(f"Generated: {output_file}")
    return table_content


def generate_baseline_seq_table(outputs_dir: Path = None, output_file: Path = None) -> str:
    """Generate baseline comparison table for sequence datasets."""
    if outputs_dir is None:
        script_dir = Path(__file__).parent
        outputs_dir = script_dir.parent / 'outputs'
    
    if output_file is None:
        output_file = outputs_dir / 'baseline_comp_seq.tex'
    
    return generate_baseline_comparison_table(
        outputs_dir=outputs_dir,
        dataset_keys=SEQUENCE_DATASETS,
        metrics=BASELINE_METRICS_SEQ,
        metric_headers=BASELINE_HEADERS_SEQ,
        caption='Baseline comparison for sequence datasets. Best values per dataset are in \\textbf{bold}.',
        label='tab:baseline_seq',
        output_file=output_file,
        single_column=False
    )


def generate_baseline_pde_table(outputs_dir: Path = None, output_file: Path = None) -> str:
    """Generate baseline comparison table for PDE datasets."""
    if outputs_dir is None:
        script_dir = Path(__file__).parent
        outputs_dir = script_dir.parent / 'outputs'
    
    if output_file is None:
        output_file = outputs_dir / 'baseline_comp_pde.tex'
    
    return generate_baseline_comparison_table(
        outputs_dir=outputs_dir,
        dataset_keys=PDE_DATASETS,
        metrics=BASELINE_METRICS_PDE,
        metric_headers=BASELINE_HEADERS_PDE,
        caption='Baseline comparison for PDE datasets. Best values per dataset are in \\textbf{bold}.',
        label='tab:baseline_pde',
        output_file=output_file,
        single_column=True  # Fewer columns, fits in single column
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate LaTeX tables for experiment results')
    parser.add_argument('--outputs-dir', '-o', type=str, default=None, help='Path to outputs directory')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    parser.add_argument(
        '--type', '-t',
        type=str,
        choices=['all', 'full', 'sequence', 'pde', 'compact', 'baseline_seq', 'baseline_pde', 'baseline'],
        default='all',
        help='Type of table to generate (default: all)'
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
    
    args = parser.parse_args()
    
    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else None
    output_file = Path(args.output) if args.output else None
    
    # --include-gaussian overrides --ignore-gaussian
    ignore_gaussian = not args.include_gaussian
    
    if args.type == 'all':
        generate_all_tables(outputs_dir, ignore_gaussian=ignore_gaussian)
        # Also generate baseline tables
        print("\n4. Baseline sequence comparison...")
        generate_baseline_seq_table(outputs_dir)
        print("\n5. Baseline PDE comparison...")
        generate_baseline_pde_table(outputs_dir)
    elif args.type == 'full':
        generate_latex_table(outputs_dir, output_file, ignore_gaussian=ignore_gaussian)
    elif args.type == 'sequence':
        generate_sequence_table(outputs_dir, output_file, ignore_gaussian=ignore_gaussian)
    elif args.type == 'pde':
        generate_pde_table(outputs_dir, output_file, ignore_gaussian=ignore_gaussian)
    elif args.type == 'compact':
        generate_compact_table(outputs_dir, output_file, ignore_gaussian=ignore_gaussian)
    elif args.type == 'baseline_seq':
        generate_baseline_seq_table(outputs_dir, output_file)
    elif args.type == 'baseline_pde':
        generate_baseline_pde_table(outputs_dir, output_file)
    elif args.type == 'baseline':
        generate_baseline_seq_table(outputs_dir)
        generate_baseline_pde_table(outputs_dir)

