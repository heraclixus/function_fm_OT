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


# Metrics to include in the table
# Full set of all metrics (excluding kurtosis, skewness, and convergence_rate for cleaner tables)
METRICS_ALL = ['mean_mse', 'variance_mse', 'autocorrelation_mse', 'spectrum_mse']
METRIC_HEADERS_ALL = ['Mean', 'Variance', 'Autocorr.', 'Spectrum']

# Sequence datasets: no spectrum (not applicable)
METRICS_SEQUENCE = ['mean_mse', 'variance_mse', 'autocorrelation_mse']
METRIC_HEADERS_SEQUENCE = ['Mean', 'Variance', 'Autocorr.']

# Sequence datasets FULL: including skewness and kurtosis
METRICS_SEQUENCE_FULL = ['mean_mse', 'variance_mse', 'skewness_mse', 'kurtosis_mse', 'autocorrelation_mse']
METRIC_HEADERS_SEQUENCE_FULL = ['Mean', 'Variance', 'Skewness', 'Kurtosis', 'Autocorr.']

# PDE datasets: no autocorrelation, kurtosis, or skewness (not applicable/meaningful for 2D fields)
# Use spectrum_mse_log for PDE datasets (log scale is more appropriate for energy spectra)
METRICS_PDE = ['mean_mse', 'variance_mse', 'spectrum_mse_log']
METRIC_HEADERS_PDE = ['Mean', 'Variance', 'Spectrum (log)']

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
SEQUENCE_DATASETS = ['AEMET', 'expr_genes', 'econ1', 'Heston', 'rBergomi']
PDE_DATASETS = ['kdv', 'navier_stokes', 'stochastic_kdv', 'stochastic_ns']


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
    
    Also includes sweep configs (sig_*, rbf_*, euclidean_*, gp_*_kernel_*) in their respective categories.
    """
    configs = normalize_configs(configs)
    
    # Extend categories to include sweep configs
    extended_categories = {cat: list(names) for cat, names in categories.items()}
    
    # Map ot_kernel values to categories
    ot_kernel_to_category = {
        'signature': 'Signature',
        'rbf': 'RBF',
        'euclidean': 'Euclidean',
    }
    
    # Add configs to appropriate categories based on their ot_kernel field
    for config_name, config_data in configs.items():
        # Skip baseline models
        if config_name in ['DDPM', 'NCSN', 'independent']:
            continue
        
        # Get the actual metrics data (might be nested or flattened)
        if isinstance(config_data, dict):
            if 'metrics' in config_data and isinstance(config_data['metrics'], dict):
                actual_config = config_data['metrics']
            elif 'quality_metrics' in config_data:
                actual_config = config_data['quality_metrics']
            else:
                actual_config = config_data
        else:
            continue
        
        # Get OT-related fields
        ot_kernel = actual_config.get('ot_kernel', '')
        use_ot = actual_config.get('use_ot', False)
        ot_method = actual_config.get('ot_method', '')
        
        # Skip non-OT configs and gaussian OT
        if not use_ot or ot_method == 'gaussian':
            continue
        
        # Categorize by ot_kernel field (most reliable)
        if ot_kernel and ot_kernel.lower() in ot_kernel_to_category:
            category = ot_kernel_to_category[ot_kernel.lower()]
            if category in extended_categories and config_name not in extended_categories[category]:
                extended_categories[category].append(config_name)
        # Fallback: try to infer from config name
        elif 'signature' in config_name.lower() or config_name.startswith('sig_'):
            if 'Signature' in extended_categories and config_name not in extended_categories['Signature']:
                extended_categories['Signature'].append(config_name)
        elif 'rbf' in config_name.lower():
            if 'RBF' in extended_categories and config_name not in extended_categories['RBF']:
                extended_categories['RBF'].append(config_name)
        elif 'euclidean' in config_name.lower():
            if 'Euclidean' in extended_categories and config_name not in extended_categories['Euclidean']:
                extended_categories['Euclidean'].append(config_name)
    
    best_per_category = {}
    
    # All metrics we care about
    all_metrics = [
        'mean_mse', 'variance_mse', 'skewness_mse', 'kurtosis_mse',
        'autocorrelation_mse', 'spectrum_mse', 'spectrum_mse_log', 
        'density_mse', 'convergence_rate'
    ]
    
    # Metrics where higher is better
    higher_is_better = ['convergence_rate']
    
    for category_name, config_names in extended_categories.items():
        # Collect all valid configs for this category
        category_metrics_list = []
        
        for config_name in config_names:
            if config_name in configs:
                config_data = configs[config_name]
                
                # Handle different data formats
                if 'metrics' in config_data and isinstance(config_data['metrics'], dict):
                    metrics = config_data['metrics']
                elif 'quality_metrics' in config_data:
                    metrics = config_data['quality_metrics']
                else:
                    metrics = config_data
                
                # Skip configs that aren't k-FFM (for non-Independent categories)
                if category_name != 'Independent':
                    use_ot = metrics.get('use_ot', config_data.get('use_ot', False))
                    ot_method = metrics.get('ot_method', config_data.get('ot_method', ''))
                    if use_ot != True or ot_method == 'gaussian':
                        continue
                
                category_metrics_list.append(metrics)
        
        if category_metrics_list:
            # Build best metrics by taking best value for each metric
            best_metrics = {}
            for metric in all_metrics:
                values = [m.get(metric) for m in category_metrics_list 
                         if m.get(metric) is not None and isinstance(m.get(metric), (int, float))]
                if values:
                    if metric in higher_is_better:
                        best_metrics[metric] = max(values)
                    else:
                        best_metrics[metric] = min(values)
            
            if best_metrics:
                best_per_category[category_name] = {
                    'name': f'{category_name}_best',
                    'metrics': best_metrics
                }
    
    return best_per_category


def format_value(value, precision: int = 2, rank: int = 0) -> str:
    """Format a value for LaTeX table using consistent scientific notation (a.aa × 10^{-b}).
    
    Parameters
    ----------
    value : float or None
        The value to format
    precision : int
        Number of decimal places for mantissa
    rank : int
        0 = no highlighting, 1 = best (green), 2 = second best (orange)
    """
    if value is None:
        return '--'
    
    if value == 0:
        inner = '0'
    else:
        abs_val = abs(value)
        # Always use scientific notation: a.aa × 10^{b}
        exp = int(np.floor(np.log10(abs_val)))
        mantissa = value / (10 ** exp)
        inner = f'{mantissa:.{precision}f} \\times 10^{{{exp}}}'
    
    if rank == 1:
        # Best: green bold
        return f'\\textcolor{{ForestGreen}}{{$\\mathbf{{{inner}}}$}}'
    elif rank == 2:
        # Second best: orange bold
        return f'\\textcolor{{Orange}}{{$\\mathbf{{{inner}}}$}}'
    else:
        return f'${inner}$'


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
        
    Priority: aggregated_results > comprehensive_metrics > experiment_summary
    """
    dataset_info = DATASETS[dataset_key]
    dataset_dir = outputs_dir / dataset_info['dir']
    
    if not dataset_dir.exists():
        return None
    
    # First, try to load aggregated results (includes sweep experiments)
    aggregated = load_aggregated_results(dataset_dir, dataset_key)
    
    if aggregated and 'configs' in aggregated:
        # Extract configs from aggregated format (metrics are nested under 'metrics' key)
        configs = {}
        for config_name, config_data in aggregated['configs'].items():
            if 'metrics' in config_data:
                configs[config_name] = config_data['metrics']
            else:
                configs[config_name] = config_data
    else:
        # Fallback: load from summary files
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
    """Generate a LaTeX table with all datasets and kernel categories.
    
    Features:
    - Best values in green, second best in orange
    - Compact formatting with scriptsize
    """
    if outputs_dir is None:
        script_dir = Path(__file__).parent
        outputs_dir = script_dir.parent / 'outputs'
    
    if output_file is None:
        output_file = outputs_dir / 'table1.tex'
    
    higher_is_better = ['convergence_rate']
    
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
    
    # Kernel order (Independent first as N/A baseline, then others)
    kernel_order = ['Independent', 'Signature', 'RBF', 'Euclidean']
    if not ignore_gaussian:
        kernel_order.append('Gaussian')
    
    # Build LaTeX table with compact formatting
    lines = []
    n_metrics = len(METRIC_HEADERS)
    
    lines.append(r'\begin{table}[t]')
    lines.append(r'\scriptsize')
    lines.append(r'\setlength{\tabcolsep}{3pt}')
    lines.append(r'\centering')
    lines.append(r'\caption{OT kernel comparison across all datasets. Lower is better. Best is \textcolor{ForestGreen}{$\mathbf{green}$}, second best is \textcolor{Orange}{$\mathbf{orange}$}.}')
    lines.append(r'\label{tab:performance_comparison}')
    lines.append(r'\resizebox{\columnwidth}{!}{%')
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
        
        available_kernels = [k for k in kernel_order if k in best_configs]
        n_kernels = len(available_kernels)
        
        # Find 1st and 2nd best values for each metric within this dataset
        metric_ranks = {}  # {metric: {kernel: rank}}
        for metric in METRICS:
            is_higher_better = metric in higher_is_better
            
            # Collect (kernel, value) pairs
            kernel_values = []
            for kernel in available_kernels:
                val = best_configs[kernel]['metrics'].get(metric)
                if val is not None:
                    kernel_values.append((kernel, val))
            
            # Sort by value
            if is_higher_better:
                kernel_values.sort(key=lambda x: x[1], reverse=True)
            else:
                kernel_values.sort(key=lambda x: x[1])
            
            # Assign ranks
            metric_ranks[metric] = {}
            for rank_idx, (kernel, _) in enumerate(kernel_values):
                metric_ranks[metric][kernel] = rank_idx + 1
        
        # Generate rows for each kernel
        for i, kernel in enumerate(available_kernels):
            config = best_configs[kernel]
            config_metrics = config['metrics']
            
            # Dataset column (multirow for first row)
            if i == 0:
                dataset_col = f'\\multirow{{{n_kernels}}}{{*}}{{{dataset_title}}}'
            else:
                dataset_col = ''
            
            # Kernel column (N/A for Independent)
            kernel_col = 'N/A' if kernel == 'Independent' else kernel
            
            # Metric values
            values = []
            for metric in METRICS:
                val = config_metrics.get(metric)
                rank = metric_ranks.get(metric, {}).get(kernel, 0)
                formatted = format_value(val, rank=rank if rank <= 2 else 0)
                values.append(formatted)
            
            row = f'{dataset_col}\n & {kernel_col} & ' + ' & '.join(values) + r' \\'
            lines.append(row)
        
        # Add midrule between datasets (except after last)
        if dataset_key != list(all_data.keys())[-1]:
            lines.append(r'\midrule')
    
    # Table footer
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'} % end resizebox')
    lines.append(r'\end{table}')
    
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
    single_column: bool = False,
    higher_is_better: List[str] = None
) -> str:
    """Generate a LaTeX table for a specific subset of datasets.
    
    Features:
    - Best values in green, second best in orange
    - Compact formatting with scriptsize
    
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
    higher_is_better : List[str], optional
        List of metric keys where higher is better (default: ['convergence_rate'])
    """
    if metrics is None:
        metrics = METRICS_ALL
    if metric_headers is None:
        metric_headers = METRIC_HEADERS_ALL
    if higher_is_better is None:
        higher_is_better = ['convergence_rate']
    
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
    
    # Build LaTeX table with compact formatting
    lines = []
    n_metrics = len(metric_headers)
    
    lines.append(r'\begin{table}[t]')
    lines.append(r'\scriptsize')
    lines.append(r'\setlength{\tabcolsep}{3pt}')
    lines.append(r'\centering')
    lines.append(f'\\caption{{{caption}}}')
    lines.append(f'\\label{{{label}}}')
    lines.append(r'\resizebox{\columnwidth}{!}{%')
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
        
        available_kernels = [k for k in kernel_order if k in best_configs]
        n_kernels = len(available_kernels)
        
        # Find 1st and 2nd best values for each metric within this dataset
        metric_ranks = {}  # {metric: {kernel: rank}}
        for metric in metrics:
            is_higher_better = metric in higher_is_better
            
            # Collect (kernel, value) pairs
            kernel_values = []
            for kernel in available_kernels:
                val = best_configs[kernel]['metrics'].get(metric)
                if val is not None:
                    kernel_values.append((kernel, val))
            
            # Sort by value
            if is_higher_better:
                kernel_values.sort(key=lambda x: x[1], reverse=True)
            else:
                kernel_values.sort(key=lambda x: x[1])
            
            # Assign ranks
            metric_ranks[metric] = {}
            for rank_idx, (kernel, _) in enumerate(kernel_values):
                metric_ranks[metric][kernel] = rank_idx + 1  # 1 = best, 2 = second best
        
        # Generate rows for each kernel
        for i, kernel in enumerate(available_kernels):
            config = best_configs[kernel]
            config_metrics = config['metrics']
            
            if i == 0:
                dataset_col = f'\\multirow{{{n_kernels}}}{{*}}{{{dataset_title}}}'
            else:
                dataset_col = ''
            
            kernel_col = 'N/A' if kernel == 'Independent' else kernel
            
            values = []
            for metric in metrics:
                val = config_metrics.get(metric)
                rank = metric_ranks.get(metric, {}).get(kernel, 0)
                formatted = format_value(val, rank=rank if rank <= 2 else 0)
                values.append(formatted)
            
            row = f'{dataset_col}\n & {kernel_col} & ' + ' & '.join(values) + r' \\'
            lines.append(row)
        
        if dataset_key != dataset_keys_with_data[-1]:
            lines.append(r'\midrule')
    
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'} % end resizebox')
    lines.append(r'\end{table}')
    
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
        caption='OT kernel comparison for sequence datasets. Lower is better. Best is \\textcolor{ForestGreen}{$\\mathbf{green}$}, second best is \\textcolor{Orange}{$\\mathbf{orange}$}.',
        label='tab:sequence_kernel',
        output_file=output_file,
        metrics=METRICS_SEQUENCE,
        metric_headers=METRIC_HEADERS_SEQUENCE,
        ignore_gaussian=ignore_gaussian,
        higher_is_better=['convergence_rate']
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
        caption='OT kernel comparison for PDE datasets. Lower is better. Best is \\textcolor{ForestGreen}{$\\mathbf{green}$}, second best is \\textcolor{Orange}{$\\mathbf{orange}$}.',
        label='tab:pde_kernel',
        output_file=output_file,
        metrics=METRICS_PDE,
        metric_headers=METRIC_HEADERS_PDE,
        ignore_gaussian=ignore_gaussian,
        single_column=True,
        higher_is_better=['convergence_rate']
    )


def generate_sequence_table_full(outputs_dir: Path = None, output_file: Path = None, ignore_gaussian: bool = True) -> str:
    """Generate a LaTeX table for sequence datasets with ALL metrics (including skewness, kurtosis)."""
    if outputs_dir is None:
        script_dir = Path(__file__).parent
        outputs_dir = script_dir.parent / 'outputs'
    
    if output_file is None:
        output_file = outputs_dir / 'table_sequence_full.tex'
    
    return generate_subtable(
        outputs_dir=outputs_dir,
        dataset_keys=SEQUENCE_DATASETS,
        caption='OT kernel comparison for sequence datasets (full metrics). Lower is better. Best is \\textcolor{ForestGreen}{$\\mathbf{green}$}, second best is \\textcolor{Orange}{$\\mathbf{orange}$}.',
        label='tab:sequence_kernel_full',
        output_file=output_file,
        metrics=METRICS_SEQUENCE_FULL,
        metric_headers=METRIC_HEADERS_SEQUENCE_FULL,
        ignore_gaussian=ignore_gaussian,
        higher_is_better=['convergence_rate']
    )


def generate_convergence_rate_table(outputs_dir: Path = None, output_file: Path = None, ignore_gaussian: bool = True) -> str:
    """
    Generate a LaTeX table for convergence rate only.
    Rows are datasets, columns are kernel types, entries are best convergence rate.
    Higher convergence rate is better.
    """
    if outputs_dir is None:
        script_dir = Path(__file__).parent
        outputs_dir = script_dir.parent / 'outputs'
    
    if output_file is None:
        output_file = outputs_dir / 'table_convergence_rate.tex'
    
    # Collect data from all datasets
    all_data = {}
    all_datasets = SEQUENCE_DATASETS + PDE_DATASETS
    
    for dataset_key in all_datasets:
        if dataset_key not in DATASETS:
            continue
        best_configs = load_dataset_data(dataset_key, outputs_dir, ignore_gaussian=ignore_gaussian)
        if best_configs:
            all_data[dataset_key] = {
                'title': DATASETS[dataset_key]['title'],
                'best_configs': best_configs
            }
    
    if not all_data:
        print("No data found for convergence rate table")
        return ""
    
    # Kernel order (columns)
    kernel_order = ['Independent', 'Signature', 'RBF', 'Euclidean']
    if not ignore_gaussian:
        kernel_order.append('Gaussian')
    
    # Build LaTeX table
    lines = []
    n_kernels = len(kernel_order)
    
    lines.append(r'\begin{table}[t]')
    lines.append(r'\scriptsize')
    lines.append(r'\setlength{\tabcolsep}{3pt}')
    lines.append(r'\centering')
    lines.append(r'\caption{Convergence rate comparison (higher is better). Best is \textcolor{ForestGreen}{$\mathbf{green}$}, second best is \textcolor{Orange}{$\mathbf{orange}$}.}')
    lines.append(r'\label{tab:convergence_rate}')
    lines.append(r'\resizebox{\columnwidth}{!}{%')
    lines.append(r'\begin{tabular}{l' + 'r' * n_kernels + '}')
    lines.append(r'\toprule')
    
    # Header row: Dataset & kernel names
    header_kernels = [('N/A' if k == 'Independent' else k) for k in kernel_order]
    header = 'Dataset & ' + ' & '.join(header_kernels) + r' \\'
    lines.append(header)
    lines.append(r'\midrule')
    
    # Separate sequence and PDE datasets with midrule
    seq_keys = [k for k in SEQUENCE_DATASETS if k in all_data]
    pde_keys = [k for k in PDE_DATASETS if k in all_data]
    
    def add_dataset_rows(dataset_keys_subset, add_midrule_after=False):
        for dataset_key in dataset_keys_subset:
            data = all_data[dataset_key]
            dataset_title = data['title']
            best_configs = data['best_configs']
            
            # Get convergence rates for all kernels
            kernel_values = {}
            for kernel in kernel_order:
                if kernel in best_configs:
                    val = best_configs[kernel]['metrics'].get('convergence_rate')
                    if val is not None and isinstance(val, (int, float)):
                        kernel_values[kernel] = val
            
            # Compute ranks (higher is better for convergence rate)
            sorted_kernels = sorted(kernel_values.items(), key=lambda x: x[1], reverse=True)
            ranks = {k: i + 1 for i, (k, _) in enumerate(sorted_kernels)}
            
            # Format values
            values = []
            for kernel in kernel_order:
                if kernel in kernel_values:
                    val = kernel_values[kernel]
                    rank = ranks.get(kernel, 0)
                    formatted = format_value(val, rank=rank if rank <= 2 else 0)
                else:
                    formatted = '--'
                values.append(formatted)
            
            row = f'{dataset_title} & ' + ' & '.join(values) + r' \\'
            lines.append(row)
        
        if add_midrule_after and dataset_keys_subset:
            lines.append(r'\midrule')
    
    # Add sequence datasets
    add_dataset_rows(seq_keys, add_midrule_after=bool(pde_keys))
    
    # Add PDE datasets
    add_dataset_rows(pde_keys, add_midrule_after=False)
    
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'} % end resizebox')
    lines.append(r'\end{table}')
    
    table_content = '\n'.join(lines)
    
    with open(output_file, 'w') as f:
        f.write(table_content)
    
    print(f"Generated: {output_file}")
    return table_content


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
                is_best = val is not None and val == dataset_best[metric]
                formatted = format_value(val, rank=1 if is_best else 0)
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

# Metrics for baseline comparison (only key metrics for single-column table)
BASELINE_METRICS_SEQ = ['mean_mse', 'variance_mse', 'autocorrelation_mse']
BASELINE_HEADERS_SEQ = ['Mean', 'Var.', 'Autocorr.']

BASELINE_METRICS_PDE = ['mean_mse', 'variance_mse', 'spectrum_mse_log']
BASELINE_HEADERS_PDE = ['Mean', 'Variance', 'Spectrum (log)']


def get_baseline_models_data(dataset_key: str, outputs_dir: Path) -> Optional[Dict]:
    """
    Get metrics for baseline models (DDPM, NCSN, FFM, k-FFM) for a dataset.
    
    For k-FFM, we select the BEST value for each metric across all kernel variants,
    not just one globally-selected configuration.
    
    Returns dict with keys: 'DDPM', 'NCSN', 'FFM', 'k-FFM'
    
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
        # Use aggregated results - extract configs with proper format
        configs = {}
        for config_name, config_data in aggregated['configs'].items():
            if 'metrics' in config_data:
                configs[config_name] = config_data['metrics']
            else:
                configs[config_name] = config_data
    else:
        # Fallback: Load experiment summary
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
    
    # Get k-FFM: for each metric, find the BEST value across all OT kernel types
    # Focus on kernel-based OT methods (exclude gaussian_ot)
    ot_config_names = [
        # Signature kernel configs
        'signature_sinkhorn_reg0.1', 'signature_sinkhorn_reg0.5', 'signature_sinkhorn_reg1.0',
        'signature_sinkhorn_barycentric',
        # RBF kernel configs
        'rbf_exact', 'rbf_sinkhorn_reg0.1', 'rbf_sinkhorn_reg0.5', 'rbf_sinkhorn_reg1.0',
        'rbf_sinkhorn_barycentric',
        # Euclidean kernel configs
        'euclidean_exact', 'euclidean_sinkhorn_reg0.1', 'euclidean_sinkhorn_reg0.5', 'euclidean_sinkhorn_reg1.0',
        # NOTE: gaussian_ot is excluded - focusing on kernel methods
    ]
    
    # Also include any config that looks like an OT kernel config (from sweeps)
    # Exclude gaussian_* configs to focus on kernel methods
    # IMPORTANT: Must verify use_ot=True for all sweep configs
    for config_name in configs.keys():
        if config_name not in ot_config_names and config_name not in ['DDPM', 'NCSN', 'independent', 'gaussian_ot']:
            if config_name.startswith('gaussian'):
                continue
            
            config_data = configs[config_name]
            if not isinstance(config_data, dict):
                continue
            
            # Get the actual metrics (might be nested under 'metrics' in aggregated results)
            actual_data = config_data.get('metrics', config_data)
            if 'quality_metrics' in actual_data:
                actual_data = actual_data['quality_metrics']
            
            # Only include if use_ot=True and not gaussian method
            use_ot = actual_data.get('use_ot', config_data.get('use_ot', False))
            ot_method = actual_data.get('ot_method', config_data.get('ot_method', ''))
            
            if use_ot == True and ot_method != 'gaussian':
                ot_config_names.append(config_name)
    
    # All metrics we might want to compare
    all_metrics = [
        'mean_mse', 'variance_mse', 'kurtosis_mse', 'skewness_mse', 
        'autocorrelation_mse', 'spectrum_mse', 'spectrum_mse_log', 'density_mse',
        'convergence_rate'
    ]
    
    # Collect all metrics from all OT configs (verify use_ot=True)
    ot_metrics_list = []
    for config_name in ot_config_names:
        if config_name in configs:
            config_data = configs[config_name]
            
            # Get metrics (might be nested)
            if 'metrics' in config_data:
                metrics = config_data['metrics']
            elif 'quality_metrics' in config_data:
                metrics = config_data['quality_metrics']
            else:
                metrics = config_data
            
            # Double-check use_ot=True (some configs might have slipped through)
            use_ot = metrics.get('use_ot', config_data.get('use_ot', False))
            ot_method = metrics.get('ot_method', config_data.get('ot_method', ''))
            
            if use_ot == True and ot_method != 'gaussian':
                ot_metrics_list.append(metrics)
    
    if ot_metrics_list:
        # Build k-FFM metrics by taking the best value for each metric
        # For convergence_rate, higher is better; for all others, lower is better
        kffm_metrics = {}
        for metric in all_metrics:
            values = [m.get(metric) for m in ot_metrics_list 
                     if m.get(metric) is not None and isinstance(m.get(metric), (int, float))]
            if values:
                if metric == 'convergence_rate':
                    kffm_metrics[metric] = max(values)  # Higher is better
                else:
                    kffm_metrics[metric] = min(values)  # Lower is better
        
        if kffm_metrics:
            result['k-FFM'] = kffm_metrics
    
    return result if result else None


def generate_baseline_comparison_table(
    outputs_dir: Path,
    dataset_keys: List[str],
    metrics: List[str],
    metric_headers: List[str],
    caption: str,
    label: str,
    output_file: Path,
    single_column: bool = False,
    higher_is_better: List[str] = None
) -> str:
    """Generate baseline comparison table (DDPM, NCSN, FFM, k-FFM).
    
    Features:
    - Best values in green, second best in orange
    - k-FFM rows have gray background
    - Compact formatting with scriptsize
    """
    if higher_is_better is None:
        higher_is_better = ['convergence_rate']
    
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
    
    # Table preamble with compact formatting
    lines.append(r'\begin{table}[t]')
    lines.append(r'\scriptsize')
    lines.append(r'\setlength{\tabcolsep}{3pt}')
    lines.append(r'\centering')
    lines.append(f'\\caption{{{caption}}}')
    lines.append(f'\\label{{{label}}}')
    lines.append(r'\resizebox{\columnwidth}{!}{%')
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
        
        available_models = [m for m in model_order if m in models]
        n_models = len(available_models)
        
        # Find 1st and 2nd best values for each metric within this dataset
        metric_ranks = {}  # {metric: {model: rank}}
        for metric in metrics:
            is_higher_better = metric in higher_is_better
            
            # Collect (model, value) pairs
            model_values = []
            for model in available_models:
                val = models[model].get(metric)
                if val is not None:
                    model_values.append((model, val))
            
            # Sort by value
            if is_higher_better:
                model_values.sort(key=lambda x: x[1], reverse=True)
            else:
                model_values.sort(key=lambda x: x[1])
            
            # Assign ranks
            metric_ranks[metric] = {}
            for rank_idx, (model, _) in enumerate(model_values):
                metric_ranks[metric][model] = rank_idx + 1  # 1 = best, 2 = second best
        
        # Generate rows for each model
        for i, model in enumerate(available_models):
            model_metrics = models[model]
            
            if i == 0:
                dataset_col = f'\\multirow{{{n_models}}}{{*}}{{{dataset_title}}}'
            else:
                dataset_col = ''
            
            # Build value cells
            values = []
            for metric in metrics:
                val = model_metrics.get(metric)
                rank = metric_ranks.get(metric, {}).get(model, 0)
                formatted = format_value(val, rank=rank if rank <= 2 else 0)
                
                # Add gray background for k-FFM
                if model == 'k-FFM':
                    formatted = f'\\cellcolor[gray]{{0.9}}{formatted}'
                
                values.append(formatted)
            
            row = f'{dataset_col}\n & {model} & ' + ' & '.join(values) + r' \\'
            lines.append(row)
        
        if dataset_key != dataset_keys_with_data[-1]:
            lines.append(r'\midrule')
    
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'} % end resizebox')
    lines.append(r'\end{table}')
    
    table_content = '\n'.join(lines)
    
    with open(output_file, 'w') as f:
        f.write(table_content)
    
    print(f"Generated: {output_file}")
    return table_content


def generate_baseline_seq_table(outputs_dir: Path = None, output_file: Path = None) -> str:
    """Generate baseline comparison table for sequence datasets (single-column, key metrics only)."""
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
        caption='Baseline comparison for sequence datasets. Lower is better. Best is \\textcolor{ForestGreen}{$\\mathbf{green}$}, second best is \\textcolor{Orange}{$\\mathbf{orange}$}.',
        label='tab:baseline_seq',
        output_file=output_file,
        single_column=True
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
        caption='Baseline comparison for PDE datasets. Lower is better (higher for Conv.\\ Rate). Best is \\textcolor{ForestGreen}{$\\mathbf{green}$}, second best is \\textcolor{Orange}{$\\mathbf{orange}$}.',
        label='tab:baseline_pde',
        output_file=output_file,
        single_column=True,
        higher_is_better=['convergence_rate']
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate LaTeX tables for experiment results')
    parser.add_argument('--outputs-dir', '-o', type=str, default=None, help='Path to outputs directory')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    parser.add_argument(
        '--type', '-t',
        type=str,
        choices=['all', 'full', 'sequence', 'pde', 'compact', 'baseline_seq', 'baseline_pde', 'baseline', 'convergence'],
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
    parser.add_argument(
        '--generate-full',
        action='store_true',
        default=False,
        help='Generate tables with all columns (including skewness, kurtosis) for sequence datasets, plus convergence rate table'
    )
    
    args = parser.parse_args()
    
    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else None
    output_file = Path(args.output) if args.output else None
    
    # --include-gaussian overrides --ignore-gaussian
    ignore_gaussian = not args.include_gaussian
    
    if args.generate_full:
        # Generate full tables with all metrics
        print("=" * 60)
        print("Generating Full LaTeX Tables (with all metrics)")
        if ignore_gaussian:
            print("(Ignoring Gaussian kernel category)")
        print("=" * 60)
        
        print("\n1. Full sequence table (with skewness, kurtosis)...")
        generate_sequence_table_full(outputs_dir, ignore_gaussian=ignore_gaussian)
        
        print("\n2. Convergence rate table...")
        generate_convergence_rate_table(outputs_dir, ignore_gaussian=ignore_gaussian)
        
        print("\n" + "=" * 60)
        print("Done!")
        print("=" * 60)
    elif args.type == 'all':
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
    elif args.type == 'convergence':
        generate_convergence_rate_table(outputs_dir, output_file, ignore_gaussian=ignore_gaussian)

