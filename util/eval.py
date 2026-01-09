import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, skew, kurtosis
from scipy.signal import periodogram, welch
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json


# =============================================================================
# Pointwise Statistics (as in Table 1 of the FFM paper)
# =============================================================================

def compute_pointwise_mean(samples: torch.Tensor) -> torch.Tensor:
    """
    Compute pointwise mean across samples.
    
    Parameters
    ----------
    samples : Tensor, shape (n_samples, n_points) or (n_samples, n_channels, n_points)
        Generated or real samples
        
    Returns
    -------
    mean : Tensor, shape (n_points,) or (n_channels, n_points)
        Pointwise mean
    """
    if samples.ndim == 3:
        return samples.mean(dim=0)
    return samples.mean(dim=0)


def compute_pointwise_variance(samples: torch.Tensor) -> torch.Tensor:
    """Compute pointwise variance across samples."""
    if samples.ndim == 3:
        return samples.var(dim=0)
    return samples.var(dim=0)


def compute_pointwise_skewness(samples: torch.Tensor) -> np.ndarray:
    """Compute pointwise skewness across samples."""
    samples_np = samples.cpu().numpy()
    if samples_np.ndim == 3:
        # (n_samples, n_channels, n_points) -> compute along axis 0
        return skew(samples_np, axis=0)
    return skew(samples_np, axis=0)


def compute_pointwise_kurtosis(samples: torch.Tensor) -> np.ndarray:
    """Compute pointwise kurtosis across samples."""
    samples_np = samples.cpu().numpy()
    if samples_np.ndim == 3:
        return kurtosis(samples_np, axis=0)
    return kurtosis(samples_np, axis=0)


def compute_autocorrelation(samples: torch.Tensor, lag: int = 1) -> torch.Tensor:
    """
    Compute mean autocorrelation at given lag across samples.
    
    Parameters
    ----------
    samples : Tensor, shape (n_samples, n_points)
        Time series samples (1D sequences only, not 2D spatial data)
    lag : int
        Lag for autocorrelation
        
    Returns
    -------
    autocorr : Tensor
        Mean autocorrelation at the given lag, or NaN if input is not 1D sequence data
    """
    # Handle channel dimension for 1D sequences: (n_samples, 1, n_points) -> (n_samples, n_points)
    if samples.ndim == 3 and samples.shape[1] == 1:
        samples = samples.squeeze(1)
    
    # If still 3D after squeeze, it's 2D spatial data - autocorrelation doesn't apply
    if samples.ndim != 2:
        return torch.tensor(float('nan'))
    
    n_samples, n_points = samples.shape
    
    if lag >= n_points:
        return torch.tensor(0.0)
    
    # Compute autocorrelation for each sample
    autocorrs = []
    for i in range(n_samples):
        x = samples[i]
        x_centered = x - x.mean()
        
        # Autocorrelation at lag
        numerator = (x_centered[:-lag] * x_centered[lag:]).sum()
        denominator = (x_centered ** 2).sum()
        
        if denominator > 0:
            autocorrs.append((numerator / denominator).item())
    
    return torch.tensor(np.mean(autocorrs))


def compute_all_pointwise_statistics(
    real: torch.Tensor,
    generated: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute MSE between pointwise statistics of real and generated samples.
    
    This matches Table 1 in the FFM paper.
    
    Parameters
    ----------
    real : Tensor
        Real samples, shape (n_samples, n_points) or (n_samples, 1, n_points)
        For 2D spatial data: (n_samples, H, W) or (n_samples, 1, H, W)
    generated : Tensor
        Generated samples, same shape as real
        
    Returns
    -------
    stats : dict
        Dictionary with MSE for each statistic:
        - mean_mse
        - variance_mse
        - skewness_mse
        - kurtosis_mse
        - autocorrelation_mse (None for 2D spatial data)
    """
    # Detect if this is 2D spatial data (e.g., Navier-Stokes)
    # After removing channel dim, 2D spatial data will be (n_samples, H, W)
    is_2d_spatial = False
    
    # Handle 4D tensors: (n_samples, C, H, W) -> squeeze channel
    if real.ndim == 4:
        real = real.squeeze(1)
        is_2d_spatial = True
    if generated.ndim == 4:
        generated = generated.squeeze(1)
        is_2d_spatial = True
    
    # Handle 3D tensors:
    # - (n_samples, 1, n_points) is 1D data with channel -> squeeze to (n_samples, n_points)
    # - (n_samples, H, W) with H > 1 is 2D spatial data -> keep as is
    if real.ndim == 3:
        if real.shape[1] == 1:
            real = real.squeeze(1)  # 1D data with channel
        else:
            is_2d_spatial = True  # 2D spatial data
    
    if generated.ndim == 3:
        if generated.shape[1] == 1:
            generated = generated.squeeze(1)
        else:
            is_2d_spatial = True
    
    # Final safety check: if either tensor is still 3D, it's definitely 2D spatial
    if real.ndim == 3 or generated.ndim == 3:
        is_2d_spatial = True
    
    # Ensure float type
    real = real.float()
    generated = generated.float()
    
    # For 2D spatial data, flatten to (n_samples, H*W) for statistics
    if is_2d_spatial:
        real_flat = real.reshape(real.shape[0], -1)
        generated_flat = generated.reshape(generated.shape[0], -1)
    else:
        real_flat = real
        generated_flat = generated
    
    # Compute pointwise statistics
    real_mean = compute_pointwise_mean(real_flat)
    gen_mean = compute_pointwise_mean(generated_flat)
    mean_mse = ((real_mean - gen_mean) ** 2).mean().item()
    
    real_var = compute_pointwise_variance(real_flat)
    gen_var = compute_pointwise_variance(generated_flat)
    variance_mse = ((real_var - gen_var) ** 2).mean().item()
    
    real_skew = compute_pointwise_skewness(real_flat)
    gen_skew = compute_pointwise_skewness(generated_flat)
    skewness_mse = np.mean((real_skew - gen_skew) ** 2)
    
    real_kurt = compute_pointwise_kurtosis(real_flat)
    gen_kurt = compute_pointwise_kurtosis(generated_flat)
    kurtosis_mse = np.mean((real_kurt - gen_kurt) ** 2)
    
    # Autocorrelation only makes sense for 1D sequential data
    if is_2d_spatial:
        autocorrelation_mse = None
    else:
        real_autocorr = compute_autocorrelation(real_flat)
        gen_autocorr = compute_autocorrelation(generated_flat)
        # Handle NaN (returned when data is not valid for autocorrelation)
        if torch.isnan(real_autocorr) or torch.isnan(gen_autocorr):
            autocorrelation_mse = None
        else:
            autocorrelation_mse = ((real_autocorr - gen_autocorr) ** 2).item()
    
    return {
        'mean_mse': mean_mse,
        'variance_mse': variance_mse,
        'skewness_mse': skewness_mse,
        'kurtosis_mse': kurtosis_mse,
        'autocorrelation_mse': autocorrelation_mse,
    }


# =============================================================================
# Spectrum Analysis for 1D Time Series
# =============================================================================

def compute_power_spectrum_1d(
    samples: torch.Tensor,
    fs: float = 1.0,
    method: str = 'welch',
    nperseg: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average power spectral density for 1D time series samples.
    
    Parameters
    ----------
    samples : Tensor, shape (n_samples, n_points)
        Time series samples (1D sequences only)
    fs : float
        Sampling frequency
    method : str
        'welch' for Welch's method (smoother), 'periodogram' for raw periodogram
    nperseg : int, optional
        Length of each segment for Welch's method
        
    Returns
    -------
    freqs : ndarray
        Frequency values
    psd : ndarray
        Average power spectral density
        
    Raises
    ------
    ValueError
        If input is 2D spatial data (3D tensor after squeeze)
    """
    # Handle channel dimension for 1D sequences: (n_samples, 1, n_points) -> (n_samples, n_points)
    if samples.ndim == 3 and samples.shape[1] == 1:
        samples = samples.squeeze(1)
    
    # If still 3D, it's 2D spatial data - not valid for 1D spectrum
    if samples.ndim != 2:
        raise ValueError(f"compute_power_spectrum_1d expects 2D input (n_samples, n_points), "
                        f"got {samples.ndim}D tensor with shape {samples.shape}. "
                        f"Use compute_energy_spectrum_2d for 2D spatial data.")
    
    samples_np = samples.cpu().numpy()
    n_samples, n_points = samples_np.shape
    
    if nperseg is None:
        nperseg = min(256, n_points // 4) if n_points > 16 else n_points
    
    psds = []
    for i in range(n_samples):
        if method == 'welch':
            freqs, psd = welch(samples_np[i], fs=fs, nperseg=nperseg)
        else:
            freqs, psd = periodogram(samples_np[i], fs=fs)
        psds.append(psd)
    
    avg_psd = np.mean(psds, axis=0)
    return freqs, avg_psd


def spectrum_mse_1d(
    real: torch.Tensor,
    generated: torch.Tensor,
    fs: float = 1.0,
    method: str = 'welch',
    log_scale: bool = True,
) -> float:
    """
    Compute MSE between power spectra of real and generated samples.
    
    Parameters
    ----------
    real : Tensor
        Real samples
    generated : Tensor
        Generated samples
    fs : float
        Sampling frequency
    method : str
        'welch' or 'periodogram'
    log_scale : bool
        If True, compute MSE in log scale (better for power spectra)
        
    Returns
    -------
    mse : float
        Spectrum MSE
    """
    freqs_real, psd_real = compute_power_spectrum_1d(real, fs=fs, method=method)
    freqs_gen, psd_gen = compute_power_spectrum_1d(generated, fs=fs, method=method)
    
    # Ensure same length (use minimum)
    min_len = min(len(psd_real), len(psd_gen))
    psd_real = psd_real[:min_len]
    psd_gen = psd_gen[:min_len]
    
    if log_scale:
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        psd_real = np.log10(psd_real + eps)
        psd_gen = np.log10(psd_gen + eps)
    
    return float(np.mean((psd_real - psd_gen) ** 2))


# =============================================================================
# Helper: Select Best Per Kernel Category
# =============================================================================

def select_best_per_category(
    config_names: List[str],
    samples_list: List[torch.Tensor],
    quality_metrics_list: List = None,
    metric_key: str = 'mean_mse',
) -> Tuple[List[str], List[torch.Tensor], List[str]]:
    """
    Select the best configuration from each kernel category.
    
    Categories: Independent, Gaussian OT, Euclidean OT, RBF OT, Signature OT
    
    Parameters
    ----------
    config_names : List[str]
        List of configuration names
    samples_list : List[Tensor]
        List of generated samples for each config
    quality_metrics_list : List, optional
        List of quality metrics objects. If None, cannot select best.
    metric_key : str
        Which quality metric to use for selection
        
    Returns
    -------
    selected_names : List[str]
        Simplified display names (e.g., "Euclidean OT")
    selected_samples : List[Tensor]
        Samples from selected configs
    original_names : List[str]
        Original config names for the selected entries
    """
    # Define kernel categories
    categories = {
        'independent': {'display': 'Independent', 'entries': []},
        'gaussian': {'display': 'Gaussian OT', 'entries': []},
        'euclidean': {'display': 'Euclidean OT', 'entries': []},
        'rbf': {'display': 'RBF OT', 'entries': []},
        'signature': {'display': 'Signature OT', 'entries': []},
    }
    
    # Group by category
    for i, name in enumerate(config_names):
        name_lower = name.lower()
        if 'independent' in name_lower:
            categories['independent']['entries'].append(i)
        elif 'gaussian' in name_lower:
            categories['gaussian']['entries'].append(i)
        elif 'signature' in name_lower:
            categories['signature']['entries'].append(i)
        elif 'rbf' in name_lower:
            categories['rbf']['entries'].append(i)
        elif 'euclidean' in name_lower:
            categories['euclidean']['entries'].append(i)
    
    selected_names = []
    selected_samples = []
    original_names = []
    
    # Order: Independent, Gaussian, Euclidean, RBF, Signature
    for cat_key in ['independent', 'gaussian', 'euclidean', 'rbf', 'signature']:
        cat = categories[cat_key]
        if not cat['entries']:
            continue
        
        if len(cat['entries']) == 1:
            idx = cat['entries'][0]
            selected_names.append(cat['display'])
            selected_samples.append(samples_list[idx])
            original_names.append(config_names[idx])
        else:
            # Select best by quality metric
            if quality_metrics_list is not None:
                best_idx = None
                best_val = float('inf')
                for idx in cat['entries']:
                    qm = quality_metrics_list[idx]
                    val = getattr(qm, metric_key, None)
                    if val is not None and val < best_val:
                        best_val = val
                        best_idx = idx
                if best_idx is not None:
                    selected_names.append(cat['display'])
                    selected_samples.append(samples_list[best_idx])
                    original_names.append(config_names[best_idx])
            else:
                # Just take the first one if no quality metrics
                idx = cat['entries'][0]
                selected_names.append(cat['display'])
                selected_samples.append(samples_list[idx])
                original_names.append(config_names[idx])
    
    return selected_names, selected_samples, original_names


def compare_spectra_1d(
    real: torch.Tensor,
    generated: torch.Tensor,
    config_names: Optional[List[str]] = None,
    fs: float = 1.0,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
    top_k: int = 5,
    save_top_k: bool = True,
    quality_metrics_list: List = None,
    save_best_per_category: bool = True,
) -> plt.Figure:
    """
    Plot spectrum comparison between real and generated samples.
    Also generates a top-5 version and a best-per-category version.
    
    Parameters
    ----------
    real : Tensor
        Real samples
    generated : Tensor or List[Tensor]
        Generated samples (single or multiple configurations)
    config_names : List[str], optional
        Names for each generated configuration
    fs : float
        Sampling frequency
    save_path : Path, optional
        Where to save the figure
    top_k : int
        Number of top configurations for the additional top-k plot
    save_top_k : bool
        Whether to generate an additional top-k plot
    quality_metrics_list : List, optional
        Quality metrics for best-per-category selection
    save_best_per_category : bool
        Whether to generate a best-per-category plot (5 entries: one per kernel type)
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    # Compute real spectrum
    freqs_real, psd_real = compute_power_spectrum_1d(real, fs=fs)
    
    # Handle single or multiple generated samples
    if isinstance(generated, torch.Tensor):
        generated = [generated]
        config_names = config_names or ['Generated']
    elif config_names is None:
        config_names = [f'Config {i+1}' for i in range(len(generated))]
    
    # Compute spectrum MSE for each configuration
    spectrum_mses = []
    all_psd_gen = []
    for gen in generated:
        freqs_gen, psd_gen = compute_power_spectrum_1d(gen, fs=fs)
        all_psd_gen.append((freqs_gen, psd_gen))
        # Compute spectrum MSE (returns a float)
        mse = spectrum_mse_1d(real, gen, fs=fs)
        spectrum_mses.append(mse)
    
    # =========================================================================
    # Plot 1: All configurations
    # =========================================================================
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.semilogy(freqs_real, psd_real, 'k-', linewidth=2.5, label='Ground Truth', alpha=0.9)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(generated)))
    
    for i, (name, color) in enumerate(zip(config_names, colors)):
        freqs_gen, psd_gen = all_psd_gen[i]
        ax.semilogy(freqs_gen, psd_gen, '--', color=color, linewidth=1.5, label=name, alpha=0.7)
    
    ax.set_xlabel('Frequency', fontsize=11)
    ax.set_ylabel('Power Spectral Density', fontsize=11)
    ax.set_title('Power Spectrum Comparison (All Configurations)', fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if str(save_path).endswith('.pdf'):
            plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        print(f"Saved spectrum comparison to {save_path}")
    
    # =========================================================================
    # Plot 2: Top-K configurations (by spectrum MSE)
    # =========================================================================
    if save_top_k and save_path and len(generated) > 1:
        # Sort by spectrum MSE (ascending - lower is better)
        sorted_indices = np.argsort(spectrum_mses)[:top_k]
        
        fig2, ax2 = plt.subplots(figsize=figsize)
        
        ax2.semilogy(freqs_real, psd_real, 'k-', linewidth=2.5, label='Ground Truth', alpha=0.9)
        
        # Use distinct colors for top-k (green to red gradient)
        top_colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_indices)))
        
        for rank, (idx, color) in enumerate(zip(sorted_indices, top_colors)):
            freqs_gen, psd_gen = all_psd_gen[idx]
            mse = spectrum_mses[idx]
            label = f"#{rank+1} {config_names[idx]} (MSE={mse:.2e})"
            ax2.semilogy(freqs_gen, psd_gen, '-', color=color, linewidth=2, label=label, alpha=0.85)
        
        ax2.set_xlabel('Frequency', fontsize=11)
        ax2.set_ylabel('Power Spectral Density', fontsize=11)
        ax2.set_title(f'Power Spectrum Comparison (Top {min(top_k, len(generated))} by Spectrum MSE)', fontsize=12)
        ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=True)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save top-k plot
        base_path = str(save_path).replace('.pdf', '').replace('.png', '')
        plt.savefig(f"{base_path}_top{top_k}.pdf", dpi=150, bbox_inches='tight')
        plt.savefig(f"{base_path}_top{top_k}.png", dpi=150, bbox_inches='tight')
        print(f"Saved top-{top_k} spectrum comparison to {base_path}_top{top_k}.pdf")
        plt.close(fig2)
    
    # =========================================================================
    # Plot 3: Best per kernel category (5 entries)
    # =========================================================================
    if save_best_per_category and save_path and len(generated) > 1:
        # Select best from each category
        selected_names, selected_samples, _ = select_best_per_category(
            config_names, generated, quality_metrics_list, metric_key='mean_mse'
        )
        
        if len(selected_samples) > 0:
            fig3, ax3 = plt.subplots(figsize=figsize)
            
            ax3.semilogy(freqs_real, psd_real, 'k-', linewidth=2.5, label='Ground Truth', alpha=0.9)
            
            # Distinct colors for 5 categories
            cat_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            for i, (name, sample) in enumerate(zip(selected_names, selected_samples)):
                freqs_gen, psd_gen = compute_power_spectrum_1d(sample, fs=fs)
                color = cat_colors[i % len(cat_colors)]
                ax3.semilogy(freqs_gen, psd_gen, '-', color=color, linewidth=2, label=name, alpha=0.85)
            
            ax3.set_xlabel('Frequency', fontsize=12)
            ax3.set_ylabel('Power Spectral Density', fontsize=12)
            ax3.set_title('Power Spectrum Comparison (Best per Kernel Type)', fontsize=13)
            ax3.legend(loc='upper right', fontsize=10, frameon=True)
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save best-per-category plot
            base_path = str(save_path).replace('.pdf', '').replace('.png', '')
            plt.savefig(f"{base_path}_simplified.pdf", dpi=150, bbox_inches='tight')
            plt.savefig(f"{base_path}_simplified.png", dpi=150, bbox_inches='tight')
            print(f"Saved simplified spectrum comparison to {base_path}_simplified.pdf")
            plt.close(fig3)
    
    return fig


# =============================================================================
# Seasonal Pattern Analysis
# =============================================================================

def extract_seasonal_component(
    samples: torch.Tensor,
    period: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract seasonal (periodic) component from time series samples.
    
    Parameters
    ----------
    samples : Tensor, shape (n_samples, n_points)
        Time series samples (1D sequences only)
    period : int, optional
        Expected period. If None, will be estimated from data.
        
    Returns
    -------
    seasonal : Tensor, shape (n_points,)
        Average seasonal pattern
    residual_var : Tensor
        Variance of residuals after removing seasonal component
        
    Raises
    ------
    ValueError
        If input is 2D spatial data (3D tensor after squeeze)
    """
    # Handle channel dimension for 1D sequences: (n_samples, 1, n_points) -> (n_samples, n_points)
    if samples.ndim == 3 and samples.shape[1] == 1:
        samples = samples.squeeze(1)
    
    # If still 3D, it's 2D spatial data - seasonal patterns don't apply
    if samples.ndim != 2:
        raise ValueError(f"extract_seasonal_component expects 2D input (n_samples, n_points), "
                        f"got {samples.ndim}D tensor with shape {samples.shape}. "
                        f"Seasonal analysis is not applicable to 2D spatial data.")
    
    n_samples, n_points = samples.shape
    
    if period is None:
        # Estimate period from autocorrelation
        mean_sample = samples.mean(dim=0)
        mean_sample = mean_sample - mean_sample.mean()
        
        # Autocorrelation
        autocorr = torch.zeros(n_points // 2)
        for lag in range(1, n_points // 2):
            autocorr[lag] = (mean_sample[:-lag] * mean_sample[lag:]).sum() / (mean_sample ** 2).sum()
        
        # Find first peak after initial decay
        for i in range(2, len(autocorr) - 1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                period = i
                break
        
        if period is None:
            period = n_points  # No clear periodicity
    
    # Compute seasonal average
    seasonal = samples.mean(dim=0)
    
    # Compute residual variance
    residuals = samples - seasonal.unsqueeze(0)
    residual_var = residuals.var()
    
    return seasonal, residual_var


def seasonal_pattern_mse(
    real: torch.Tensor,
    generated: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute metrics comparing seasonal patterns.
    
    Parameters
    ----------
    real : Tensor
        Real samples
    generated : Tensor
        Generated samples
        
    Returns
    -------
    metrics : dict
        Dictionary with seasonal pattern metrics:
        - seasonal_mse: MSE between seasonal patterns
        - amplitude_error: Error in seasonal amplitude
        - phase_correlation: Correlation between seasonal phases
    """
    if real.ndim == 3:
        real = real.squeeze(1)
    if generated.ndim == 3:
        generated = generated.squeeze(1)
    
    seasonal_real, _ = extract_seasonal_component(real)
    seasonal_gen, _ = extract_seasonal_component(generated)
    
    # Ensure same length
    min_len = min(len(seasonal_real), len(seasonal_gen))
    seasonal_real = seasonal_real[:min_len]
    seasonal_gen = seasonal_gen[:min_len]
    
    # Seasonal pattern MSE
    seasonal_mse = float(((seasonal_real - seasonal_gen) ** 2).mean().item())
    
    # Amplitude error (peak-to-peak)
    amp_real = seasonal_real.max() - seasonal_real.min()
    amp_gen = seasonal_gen.max() - seasonal_gen.min()
    amplitude_error = float(abs(amp_real - amp_gen).item())
    
    # Phase correlation
    seasonal_real_centered = seasonal_real - seasonal_real.mean()
    seasonal_gen_centered = seasonal_gen - seasonal_gen.mean()
    
    numerator = (seasonal_real_centered * seasonal_gen_centered).sum()
    denominator = torch.sqrt((seasonal_real_centered ** 2).sum() * (seasonal_gen_centered ** 2).sum())
    
    if denominator > 0:
        phase_correlation = float((numerator / denominator).item())
    else:
        phase_correlation = 0.0
    
    return {
        'seasonal_mse': seasonal_mse,
        'amplitude_error': amplitude_error,
        'phase_correlation': phase_correlation,
    }


def compare_seasonal_patterns(
    real: torch.Tensor,
    generated_list: List[torch.Tensor],
    config_names: List[str],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> Tuple[plt.Figure, Dict[str, Dict[str, float]]]:
    """
    Compare seasonal patterns across configurations with numerical metrics.
    
    Parameters
    ----------
    real : Tensor
        Real samples
    generated_list : List[Tensor]
        List of generated samples from different configurations
    config_names : List[str]
        Names for each configuration
    save_path : Path, optional
        Where to save the figure
        
    Returns
    -------
    fig : matplotlib.Figure
    metrics : Dict[str, Dict[str, float]]
        Seasonal metrics for each configuration
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Extract seasonal patterns
    seasonal_real, _ = extract_seasonal_component(real)
    n_points = len(seasonal_real)
    x_grid = np.linspace(0, 1, n_points)
    
    # Plot seasonal patterns
    ax = axes[0]
    ax.plot(x_grid, seasonal_real.numpy(), 'k-', linewidth=2, label='Ground Truth')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(generated_list)))
    metrics = {}
    
    for gen, name, color in zip(generated_list, config_names, colors):
        seasonal_gen, _ = extract_seasonal_component(gen)
        ax.plot(x_grid[:len(seasonal_gen)], seasonal_gen.numpy(), '--', 
                color=color, linewidth=1.5, label=name, alpha=0.8)
        
        # Compute metrics
        metrics[name] = seasonal_pattern_mse(real, gen)
    
    ax.set_xlabel('Normalized Time', fontsize=11)
    ax.set_ylabel('Seasonal Component', fontsize=11)
    ax.set_title('Seasonal Pattern Comparison', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot metrics as bar chart
    ax = axes[1]
    metric_names = ['seasonal_mse', 'amplitude_error']
    x = np.arange(len(config_names))
    width = 0.35
    
    for i, metric_name in enumerate(metric_names):
        values = [metrics[name][metric_name] for name in config_names]
        ax.bar(x + i * width, values, width, label=metric_name.replace('_', ' ').title(), 
               color=plt.cm.Set2(i / len(metric_names)))
    
    ax.set_xlabel('Configuration', fontsize=11)
    ax.set_ylabel('Error', fontsize=11)
    ax.set_title('Seasonal Pattern Metrics', fontsize=12)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(config_names, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='best', fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if str(save_path).endswith('.pdf'):
            plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        print(f"Saved seasonal comparison to {save_path}")
    
    return fig, metrics


# =============================================================================
# Convergence Analysis
# =============================================================================

def compute_convergence_metrics(
    losses: List[float],
    window_size: int = 10,
) -> Dict[str, float]:
    """
    Compute numerical metrics for training convergence.
    
    Parameters
    ----------
    losses : List[float]
        Training loss values per epoch
    window_size : int
        Window size for computing final stability
        
    Returns
    -------
    metrics : dict
        Dictionary with convergence metrics:
        - final_loss: Final training loss
        - best_loss: Best (minimum) training loss
        - convergence_rate: Average rate of loss decrease (early epochs)
        - final_stability: Variance of loss in final window (lower = more stable)
        - epochs_to_90pct: Epochs to reach 90% of total improvement
    """
    losses = np.array(losses)
    n_epochs = len(losses)
    
    if n_epochs == 0:
        return {
            'final_loss': None,
            'best_loss': None,
            'convergence_rate': None,
            'final_stability': None,
            'epochs_to_90pct': None,
        }
    
    final_loss = float(losses[-1])
    best_loss = float(np.min(losses))
    
    # Convergence rate: average log decrease in first half
    half_idx = max(1, n_epochs // 2)
    if losses[0] > 0 and losses[half_idx] > 0:
        convergence_rate = float((np.log(losses[0]) - np.log(losses[half_idx])) / half_idx)
    else:
        convergence_rate = float((losses[0] - losses[half_idx]) / half_idx)
    
    # Final stability: variance in last window
    final_window = losses[-min(window_size, n_epochs):]
    final_stability = float(np.var(final_window))
    
    # Epochs to 90% improvement
    total_improvement = losses[0] - best_loss
    threshold = losses[0] - 0.9 * total_improvement
    
    epochs_to_90pct = n_epochs  # Default if never reached
    for i, loss in enumerate(losses):
        if loss <= threshold:
            epochs_to_90pct = i + 1
            break
    
    return {
        'final_loss': final_loss,
        'best_loss': best_loss,
        'convergence_rate': convergence_rate,
        'final_stability': final_stability,
        'epochs_to_90pct': epochs_to_90pct,
    }


def compare_convergence(
    losses_dict: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    figsize_curves: Tuple[int, int] = (10, 6),
    figsize_bars: Tuple[int, int] = (10, 5),
) -> Tuple[List[plt.Figure], Dict[str, Dict[str, float]]]:
    """
    Compare convergence across configurations with numerical metrics.
    Creates 3 separate plots: loss curves, final loss comparison, convergence metrics.
    
    Parameters
    ----------
    losses_dict : Dict[str, List[float]]
        Dictionary mapping config names to loss lists
    save_path : Path, optional
        Base path for saving figures. Will create:
        - {save_path}_curves.pdf - Training loss curves
        - {save_path}_final_loss.pdf - Final loss comparison (sorted ascending)
        - {save_path}_metrics.pdf - Convergence metrics (sorted by conv rate)
        
    Returns
    -------
    figs : List[matplotlib.Figure]
        List of 3 figures
    metrics : Dict[str, Dict[str, float]]
        Convergence metrics for each configuration
    """
    config_names = list(losses_dict.keys())
    colors_map = {name: plt.cm.tab10(i / len(config_names)) for i, name in enumerate(config_names)}
    
    # Compute metrics for all configs
    metrics = {}
    for name in config_names:
        metrics[name] = compute_convergence_metrics(losses_dict[name])
    
    figs = []
    
    # =========================================================================
    # Plot 1: Training Loss Curves (with legend outside)
    # =========================================================================
    fig1, ax = plt.subplots(figsize=figsize_curves)
    
    for name in config_names:
        losses = losses_dict[name]
        ax.semilogy(losses, color=colors_map[name], label=name, linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss (log scale)', fontsize=11)
    ax.set_title('Training Loss Curves', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Legend outside the plot on the right
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=True)
    
    plt.tight_layout()
    figs.append(fig1)
    
    if save_path:
        base_path = str(save_path).replace('.pdf', '').replace('.png', '')
        fig1.savefig(f"{base_path}_curves.pdf", dpi=150, bbox_inches='tight')
        fig1.savefig(f"{base_path}_curves.png", dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {base_path}_curves.pdf")
    
    # =========================================================================
    # Plot 2: Final Loss Comparison (sorted ascending - lower is better)
    # =========================================================================
    fig2, ax = plt.subplots(figsize=figsize_bars)
    
    # Sort by final loss (ascending)
    sorted_by_loss = sorted(
        [(name, metrics[name]['final_loss'] or float('inf')) for name in config_names],
        key=lambda x: x[1]
    )
    sorted_names = [x[0] for x in sorted_by_loss]
    final_losses = [x[1] for x in sorted_by_loss]
    
    # Use color gradient: green (best) to red (worst)
    n_configs = len(sorted_names)
    bar_colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, n_configs))
    
    bars = ax.bar(range(n_configs), final_losses, color=bar_colors)
    
    # Add rank labels
    for i, (bar, loss) in enumerate(zip(bars, final_losses)):
        ax.annotate(f"#{i+1}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Configuration (sorted by final loss)', fontsize=11)
    ax.set_ylabel('Final Loss', fontsize=11)
    ax.set_title('Final Loss Comparison (Lower is Better)', fontsize=12)
    ax.set_xticks(range(n_configs))
    ax.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    figs.append(fig2)
    
    if save_path:
        fig2.savefig(f"{base_path}_final_loss.pdf", dpi=150, bbox_inches='tight')
        fig2.savefig(f"{base_path}_final_loss.png", dpi=150, bbox_inches='tight')
        print(f"Saved final loss comparison to {base_path}_final_loss.pdf")
    
    # =========================================================================
    # Plot 3: Convergence Metrics (sorted by convergence rate - higher is better)
    # =========================================================================
    fig3, ax = plt.subplots(figsize=figsize_bars)
    
    # Sort by convergence rate (descending - higher is better)
    sorted_by_conv = sorted(
        [(name, metrics[name]['convergence_rate'] or 0) for name in config_names],
        key=lambda x: x[1],
        reverse=True
    )
    sorted_names_conv = [x[0] for x in sorted_by_conv]
    
    x = np.arange(len(sorted_names_conv))
    width = 0.25
    
    conv_rates = [metrics[name]['convergence_rate'] or 0 for name in sorted_names_conv]
    stabilities = [metrics[name]['final_stability'] or 0 for name in sorted_names_conv]
    epochs_90 = [metrics[name]['epochs_to_90pct'] or 0 for name in sorted_names_conv]
    
    # Normalize for visualization
    max_conv = max(conv_rates) if max(conv_rates) > 0 else 1
    max_stab = max(stabilities) if max(stabilities) > 0 else 1
    max_epochs = max(epochs_90) if max(epochs_90) > 0 else 1
    
    ax.bar(x - width, [c / max_conv for c in conv_rates], width, 
           label='Conv. Rate (norm)', color='#1f77b4', alpha=0.8)
    ax.bar(x, [1 - s / max_stab for s in stabilities], width, 
           label='Stability (norm)', color='#ff7f0e', alpha=0.8)
    ax.bar(x + width, [1 - e / max_epochs for e in epochs_90], width, 
           label='Speed (norm)', color='#2ca02c', alpha=0.8)
    
    ax.set_xlabel('Configuration (sorted by convergence rate)', fontsize=11)
    ax.set_ylabel('Normalized Score (higher=better)', fontsize=11)
    ax.set_title('Convergence Metrics (Sorted by Convergence Rate)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_names_conv, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    figs.append(fig3)
    
    if save_path:
        fig3.savefig(f"{base_path}_metrics.pdf", dpi=150, bbox_inches='tight')
        fig3.savefig(f"{base_path}_metrics.png", dpi=150, bbox_inches='tight')
        print(f"Saved convergence metrics to {base_path}_metrics.pdf")
    
    return figs, metrics


def print_convergence_table(metrics: Dict[str, Dict[str, float]]):
    """Print a formatted table of convergence metrics."""
    print("\n" + "="*90)
    print("CONVERGENCE METRICS COMPARISON")
    print("="*90)
    
    print(f"{'Config':<25} {'Final Loss':<14} {'Best Loss':<14} {'Conv Rate':<12} {'Stability':<12} {'Ep to 90%':<10}")
    print("-"*90)
    
    for name, m in metrics.items():
        final = f"{m['final_loss']:.2e}" if m['final_loss'] is not None else "N/A"
        best = f"{m['best_loss']:.2e}" if m['best_loss'] is not None else "N/A"
        rate = f"{m['convergence_rate']:.4f}" if m['convergence_rate'] is not None else "N/A"
        stab = f"{m['final_stability']:.2e}" if m['final_stability'] is not None else "N/A"
        epochs = str(m['epochs_to_90pct']) if m['epochs_to_90pct'] is not None else "N/A"
        
        print(f"{name[:24]:<25} {final:<14} {best:<14} {rate:<12} {stab:<12} {epochs:<10}")
    
    print("="*90 + "\n")


def compare_convergence_simplified(
    losses_dict: Dict[str, List[float]],
    quality_metrics_dict: Dict[str, float] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Compare convergence showing only best from each kernel category.
    
    Shows 5 entries: Independent, Gaussian OT, Euclidean OT, RBF OT, Signature OT
    
    Parameters
    ----------
    losses_dict : Dict[str, List[float]]
        Dictionary mapping config names to loss lists
    quality_metrics_dict : Dict[str, float], optional
        Dictionary mapping config names to a quality metric (for selecting best)
    save_path : Path, optional
        Where to save figure
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    # Define kernel categories
    categories = {
        'independent': {'display': 'Independent', 'best_name': None, 'best_val': float('inf')},
        'gaussian': {'display': 'Gaussian OT', 'best_name': None, 'best_val': float('inf')},
        'euclidean': {'display': 'Euclidean OT', 'best_name': None, 'best_val': float('inf')},
        'rbf': {'display': 'RBF OT', 'best_name': None, 'best_val': float('inf')},
        'signature': {'display': 'Signature OT', 'best_name': None, 'best_val': float('inf')},
    }
    
    # Find best from each category
    for name in losses_dict.keys():
        name_lower = name.lower()
        
        # Get selection metric (quality metric if available, else final loss)
        if quality_metrics_dict and name in quality_metrics_dict:
            val = quality_metrics_dict[name]
        else:
            val = losses_dict[name][-1] if losses_dict[name] else float('inf')
        
        if 'independent' in name_lower:
            if val < categories['independent']['best_val']:
                categories['independent']['best_name'] = name
                categories['independent']['best_val'] = val
        elif 'gaussian' in name_lower:
            if val < categories['gaussian']['best_val']:
                categories['gaussian']['best_name'] = name
                categories['gaussian']['best_val'] = val
        elif 'signature' in name_lower:
            if val < categories['signature']['best_val']:
                categories['signature']['best_name'] = name
                categories['signature']['best_val'] = val
        elif 'rbf' in name_lower:
            if val < categories['rbf']['best_val']:
                categories['rbf']['best_name'] = name
                categories['rbf']['best_val'] = val
        elif 'euclidean' in name_lower:
            if val < categories['euclidean']['best_val']:
                categories['euclidean']['best_name'] = name
                categories['euclidean']['best_val'] = val
    
    # Collect selected entries
    selected = []
    cat_order = ['independent', 'gaussian', 'euclidean', 'rbf', 'signature']
    for cat_key in cat_order:
        cat = categories[cat_key]
        if cat['best_name'] is not None:
            selected.append((cat['display'], cat['best_name'], losses_dict[cat['best_name']]))
    
    if not selected:
        return None
    
    # Plot loss curves
    fig, ax = plt.subplots(figsize=figsize)
    
    cat_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (display_name, orig_name, losses) in enumerate(selected):
        color = cat_colors[i % len(cat_colors)]
        ax.semilogy(losses, color=color, label=display_name, linewidth=2, alpha=0.9)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title('Training Loss Curves (Best per Kernel Type)', fontsize=13)
    ax.legend(loc='upper right', fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        base_path = str(save_path).replace('.pdf', '').replace('.png', '')
        plt.savefig(f"{base_path}_simplified.pdf", dpi=150, bbox_inches='tight')
        plt.savefig(f"{base_path}_simplified.png", dpi=150, bbox_inches='tight')
        print(f"Saved simplified convergence comparison to {base_path}_simplified.pdf")
    
    return fig


class GenerationQualityMetrics:
    """
    Container for generation quality metrics.
    
    Stores pointwise statistics MSEs and other quality measures.
    """
    
    def __init__(
        self,
        config_name: str = "",
        ot_kernel: str = "",
        ot_method: str = "",
        ot_coupling: str = "",
        use_ot: bool = False,
    ):
        self.config_name = config_name
        self.ot_kernel = ot_kernel
        self.ot_method = ot_method
        self.ot_coupling = ot_coupling
        self.use_ot = use_ot
        
        # Pointwise statistics MSEs
        self.mean_mse: Optional[float] = None
        self.variance_mse: Optional[float] = None
        self.skewness_mse: Optional[float] = None
        self.kurtosis_mse: Optional[float] = None
        self.autocorrelation_mse: Optional[float] = None
        
        # Additional metrics
        self.density_mse: Optional[float] = None
        self.final_train_loss: Optional[float] = None
        self.total_train_time: Optional[float] = None
        self.mean_path_length: Optional[float] = None
        self.mean_grad_variance: Optional[float] = None
        
        # Spectrum metrics
        self.spectrum_mse: Optional[float] = None
        self.spectrum_mse_log: Optional[float] = None
        
        # Seasonal pattern metrics
        self.seasonal_mse: Optional[float] = None
        self.seasonal_amplitude_error: Optional[float] = None
        self.seasonal_phase_correlation: Optional[float] = None
        
        # Convergence metrics
        self.convergence_rate: Optional[float] = None
        self.final_stability: Optional[float] = None
        self.epochs_to_90pct: Optional[int] = None
    
    def compute_from_samples(
        self, 
        real: torch.Tensor, 
        generated: torch.Tensor,
        compute_spectrum: bool = True,
        compute_seasonal: bool = True,
    ):
        """Compute all statistics from real and generated samples."""
        stats = compute_all_pointwise_statistics(real, generated)
        self.mean_mse = stats['mean_mse']
        self.variance_mse = stats['variance_mse']
        self.skewness_mse = stats['skewness_mse']
        self.kurtosis_mse = stats['kurtosis_mse']
        self.autocorrelation_mse = stats['autocorrelation_mse']
        
        # Compute density MSE
        try:
            self.density_mse = density_mse(real, generated)
        except Exception:
            self.density_mse = None
        
        # Compute spectrum metrics
        if compute_spectrum:
            try:
                self.spectrum_mse = spectrum_mse_1d(real, generated, log_scale=False)
                self.spectrum_mse_log = spectrum_mse_1d(real, generated, log_scale=True)
            except Exception:
                self.spectrum_mse = None
                self.spectrum_mse_log = None
        
        # Compute seasonal pattern metrics
        if compute_seasonal:
            try:
                seasonal_metrics = seasonal_pattern_mse(real, generated)
                self.seasonal_mse = seasonal_metrics['seasonal_mse']
                self.seasonal_amplitude_error = seasonal_metrics['amplitude_error']
                self.seasonal_phase_correlation = seasonal_metrics['phase_correlation']
            except Exception:
                self.seasonal_mse = None
                self.seasonal_amplitude_error = None
                self.seasonal_phase_correlation = None
    
    def set_convergence_metrics(self, losses: List[float]):
        """Compute and set convergence metrics from training losses."""
        conv_metrics = compute_convergence_metrics(losses)
        self.convergence_rate = conv_metrics['convergence_rate']
        self.final_stability = conv_metrics['final_stability']
        self.epochs_to_90pct = conv_metrics['epochs_to_90pct']
    
    @staticmethod
    def _to_python_float(val):
        """Convert numpy/torch floats to Python float for JSON serialization."""
        if val is None:
            return None
        if hasattr(val, 'item'):  # torch tensor or numpy scalar
            return float(val.item())
        if isinstance(val, (np.floating, np.integer)):
            return float(val)
        return float(val) if val is not None else None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'config_name': self.config_name,
            'ot_kernel': self.ot_kernel,
            'ot_method': self.ot_method,
            'ot_coupling': self.ot_coupling,
            'use_ot': self.use_ot,
            # Pointwise statistics
            'mean_mse': self._to_python_float(self.mean_mse),
            'variance_mse': self._to_python_float(self.variance_mse),
            'skewness_mse': self._to_python_float(self.skewness_mse),
            'kurtosis_mse': self._to_python_float(self.kurtosis_mse),
            'autocorrelation_mse': self._to_python_float(self.autocorrelation_mse),
            'density_mse': self._to_python_float(self.density_mse),
            # Training metrics
            'final_train_loss': self._to_python_float(self.final_train_loss),
            'total_train_time': self._to_python_float(self.total_train_time),
            'mean_path_length': self._to_python_float(self.mean_path_length),
            'mean_grad_variance': self._to_python_float(self.mean_grad_variance),
            # Spectrum metrics
            'spectrum_mse': self._to_python_float(self.spectrum_mse),
            'spectrum_mse_log': self._to_python_float(self.spectrum_mse_log),
            # Seasonal metrics
            'seasonal_mse': self._to_python_float(self.seasonal_mse),
            'seasonal_amplitude_error': self._to_python_float(self.seasonal_amplitude_error),
            'seasonal_phase_correlation': self._to_python_float(self.seasonal_phase_correlation),
            # Convergence metrics
            'convergence_rate': self._to_python_float(self.convergence_rate),
            'final_stability': self._to_python_float(self.final_stability),
            'epochs_to_90pct': self.epochs_to_90pct,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'GenerationQualityMetrics':
        """Load from dictionary."""
        obj = cls(
            config_name=d.get('config_name', ''),
            ot_kernel=d.get('ot_kernel', ''),
            ot_method=d.get('ot_method', ''),
            ot_coupling=d.get('ot_coupling', ''),
            use_ot=d.get('use_ot', False),
        )
        # Pointwise statistics
        obj.mean_mse = d.get('mean_mse')
        obj.variance_mse = d.get('variance_mse')
        obj.skewness_mse = d.get('skewness_mse')
        obj.kurtosis_mse = d.get('kurtosis_mse')
        obj.autocorrelation_mse = d.get('autocorrelation_mse')
        obj.density_mse = d.get('density_mse')
        # Training metrics
        obj.final_train_loss = d.get('final_train_loss')
        obj.total_train_time = d.get('total_train_time')
        obj.mean_path_length = d.get('mean_path_length')
        obj.mean_grad_variance = d.get('mean_grad_variance')
        # Spectrum metrics
        obj.spectrum_mse = d.get('spectrum_mse')
        obj.spectrum_mse_log = d.get('spectrum_mse_log')
        # Seasonal metrics
        obj.seasonal_mse = d.get('seasonal_mse')
        obj.seasonal_amplitude_error = d.get('seasonal_amplitude_error')
        obj.seasonal_phase_correlation = d.get('seasonal_phase_correlation')
        # Convergence metrics
        obj.convergence_rate = d.get('convergence_rate')
        obj.final_stability = d.get('final_stability')
        obj.epochs_to_90pct = d.get('epochs_to_90pct')
        return obj
    
    def save(self, path: Path):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'GenerationQualityMetrics':
        """Load from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


def _extract_common_prefix(config_names: List[str]) -> Tuple[str, List[str]]:
    """Extract common prefix from config names and return cleaned names.
    
    Handles two cases:
    1. All configs share a common prefix (e.g., "econ1_population_")
    2. Configs have dataset prefixes like "econ1_", "econ2_" that should be stripped
    """
    if not config_names:
        return "", []
    
    # Split each name and find common prefix
    split_names = [name.split('_') for name in config_names]
    
    # Method keywords - these indicate where the OT method starts
    method_keywords = ['independent', 'gaussian', 'euclidean', 'rbf', 'signature']
    
    # Find common prefix parts
    common_parts = []
    min_len = min(len(parts) for parts in split_names)
    
    for i in range(min_len):
        first_part = split_names[0][i]
        if all(parts[i] == first_part for parts in split_names):
            # Check if this is a method keyword - stop before it
            if first_part in method_keywords:
                break
            common_parts.append(first_part)
        else:
            break
    
    common_prefix = '_'.join(common_parts) if common_parts else ""
    
    # If no common prefix found, check if configs have dataset-like prefixes
    # (e.g., "econ1_", "econ2_", "econ3_" or "dataset1_", "dataset2_")
    if not common_prefix:
        # Check if first part looks like a dataset name (ends with digit or is short identifier)
        first_parts = set(parts[0] for parts in split_names)
        # Check if these look like dataset variants (same base, different numbers)
        if len(first_parts) > 1:
            # Try to find common base in first parts
            bases = set()
            for part in first_parts:
                # Remove trailing digits to find base
                base = part.rstrip('0123456789')
                if base:
                    bases.add(base)
            
            # If all first parts share the same base (e.g., econ1, econ2, econ3 -> econ)
            if len(bases) == 1:
                common_prefix = f"[{','.join(sorted(first_parts))}]"
                # Remove first part from each name
                cleaned_names = []
                for name in config_names:
                    parts = name.split('_', 1)
                    if len(parts) > 1:
                        cleaned_names.append(parts[1])
                    else:
                        cleaned_names.append(name)
                return common_prefix, cleaned_names
    
    # Remove common prefix from each name
    cleaned_names = []
    for name in config_names:
        if common_prefix and name.startswith(common_prefix + '_'):
            cleaned_names.append(name[len(common_prefix) + 1:])
        else:
            # Still try to extract just the method part
            parts = name.split('_')
            for i, part in enumerate(parts):
                if part in method_keywords:
                    cleaned_names.append('_'.join(parts[i:]))
                    break
            else:
                cleaned_names.append(name)
    
    return common_prefix, cleaned_names


def compare_generation_quality(
    metrics_list: List[GenerationQualityMetrics],
    save_path: Path = None,
    figsize: Tuple[int, int] = (14, 8),
    include_spectrum: bool = True,
    include_seasonal: bool = True,
    top_k: int = 5,
    show_all: bool = False,
) -> plt.Figure:
    """
    Create comparison visualization of generation quality across configurations.
    
    Uses a 2×3 grid layout with 6 core metrics:
    - Mean MSE, Variance MSE, Skewness MSE
    - Kurtosis MSE, Autocorrelation MSE, Spectrum MSE
    
    Parameters
    ----------
    metrics_list : List[GenerationQualityMetrics]
        List of metrics from different configurations
    save_path : Path, optional
        Where to save the figure
    include_spectrum : bool
        Include spectrum MSE in plot (replaces density in last position)
    include_seasonal : bool
        Not used in simplified layout, kept for API compatibility
    top_k : int
        Number of top configurations to show per metric (default: 5)
    show_all : bool
        If True, show all configurations instead of top-k (default: False)
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    if not metrics_list:
        return None
    
    # Extract common prefix for main title
    all_config_names = [m.config_name for m in metrics_list]
    common_prefix, _ = _extract_common_prefix(all_config_names)
    
    # Fixed 2×3 layout with 6 core metrics
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Define 6 core metrics (2 rows × 3 cols)
    # Format: (attr_name, title, ax, higher_is_better)
    metrics_to_plot = [
        ('mean_mse', 'Mean MSE', axes[0, 0], False),
        ('variance_mse', 'Variance MSE', axes[0, 1], False),
        ('skewness_mse', 'Skewness MSE', axes[0, 2], False),
        ('kurtosis_mse', 'Kurtosis MSE', axes[1, 0], False),
        ('autocorrelation_mse', 'Autocorrelation MSE', axes[1, 1], False),
        ('spectrum_mse', 'Spectrum MSE', axes[1, 2], False),
    ]
    
    # Color palette for top-k (green=best to red=worst)
    top_k_colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, top_k))
    
    for attr_name, title, ax, higher_is_better in metrics_to_plot:
        # Get (config_name, value, original_metric) tuples for valid entries
        valid_entries = []
        for m in metrics_list:
            val = getattr(m, attr_name, None)
            if val is not None and not np.isnan(val) and not np.isinf(val):
                # Extract short config name (remove common prefix)
                short_name = m.config_name
                if common_prefix and short_name.startswith(common_prefix + '_'):
                    short_name = short_name[len(common_prefix) + 1:]
                valid_entries.append((short_name, val, m))
        
        if not valid_entries:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=11)
            continue
        
        # Sort by value (ascending for MSE, descending for correlation)
        if higher_is_better:
            valid_entries.sort(key=lambda x: x[1], reverse=True)
        else:
            valid_entries.sort(key=lambda x: x[1])
        
        # Take top-k if not showing all
        if not show_all:
            display_entries = valid_entries[:top_k]
        else:
            display_entries = valid_entries
        
        # Create bar chart
        names = [e[0] for e in display_entries]
        values = [e[1] for e in display_entries]
        x = np.arange(len(names))
        
        # Use color gradient based on rank
        if len(display_entries) <= top_k:
            colors = top_k_colors[:len(display_entries)]
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, len(display_entries)))
        
        bars = ax.bar(x, values, color=colors)
        
        # Add rank labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            rank_label = f"#{i+1}"
            ax.annotate(rank_label, (bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Format title with "Top-k" indicator
        if not show_all and len(valid_entries) > top_k:
            ax.set_title(f'{title} (Top {top_k}/{len(valid_entries)})', fontsize=11)
        else:
            ax.set_title(title, fontsize=11)
        
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        
        # Use log scale for MSE metrics (but not for correlation)
        use_log = not higher_is_better and all(v > 0 for v in values)
        if use_log:
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Add main title with common prefix (dataset name)
    if common_prefix:
        title_text = f"Quality Metrics: {common_prefix.replace('_', ' ').title()}"
        fig.suptitle(title_text, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if str(save_path).endswith('.pdf'):
            plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        print(f"Saved quality comparison to {save_path}")
    
    return fig


def compare_generation_quality_simplified(
    metrics_list: List[GenerationQualityMetrics],
    save_path: Path = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Create simplified quality comparison showing only best from each kernel category.
    
    Shows 5 entries: Independent, Gaussian OT, Euclidean OT, RBF OT, Signature OT
    (selecting the best configuration from each category based on mean_mse)
    
    Parameters
    ----------
    metrics_list : List[GenerationQualityMetrics]
        List of metrics from all configurations
    save_path : Path, optional
        Where to save the figure
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    if not metrics_list:
        return None
    
    # Define kernel categories
    categories = {
        'independent': {'display': 'Independent', 'best_idx': None, 'best_val': float('inf')},
        'gaussian': {'display': 'Gaussian OT', 'best_idx': None, 'best_val': float('inf')},
        'euclidean': {'display': 'Euclidean OT', 'best_idx': None, 'best_val': float('inf')},
        'rbf': {'display': 'RBF OT', 'best_idx': None, 'best_val': float('inf')},
        'signature': {'display': 'Signature OT', 'best_idx': None, 'best_val': float('inf')},
    }
    
    # Find best from each category
    for i, m in enumerate(metrics_list):
        name_lower = m.config_name.lower()
        val = m.mean_mse if m.mean_mse is not None else float('inf')
        
        if 'independent' in name_lower:
            if val < categories['independent']['best_val']:
                categories['independent']['best_idx'] = i
                categories['independent']['best_val'] = val
        elif 'gaussian' in name_lower and 'ot' in name_lower:
            if val < categories['gaussian']['best_val']:
                categories['gaussian']['best_idx'] = i
                categories['gaussian']['best_val'] = val
        elif 'signature' in name_lower:
            if val < categories['signature']['best_val']:
                categories['signature']['best_idx'] = i
                categories['signature']['best_val'] = val
        elif 'rbf' in name_lower:
            if val < categories['rbf']['best_val']:
                categories['rbf']['best_idx'] = i
                categories['rbf']['best_val'] = val
        elif 'euclidean' in name_lower:
            if val < categories['euclidean']['best_val']:
                categories['euclidean']['best_idx'] = i
                categories['euclidean']['best_val'] = val
    
    # Collect selected metrics
    selected = []
    cat_order = ['independent', 'gaussian', 'euclidean', 'rbf', 'signature']
    for cat_key in cat_order:
        cat = categories[cat_key]
        if cat['best_idx'] is not None:
            selected.append((cat['display'], metrics_list[cat['best_idx']]))
    
    if not selected:
        return None
    
    # Create 2×3 subplot
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    metrics_to_plot = [
        ('mean_mse', 'Mean MSE', axes[0, 0]),
        ('variance_mse', 'Variance MSE', axes[0, 1]),
        ('skewness_mse', 'Skewness MSE', axes[0, 2]),
        ('kurtosis_mse', 'Kurtosis MSE', axes[1, 0]),
        ('autocorrelation_mse', 'Autocorr MSE', axes[1, 1]),
        ('spectrum_mse', 'Spectrum MSE', axes[1, 2]),
    ]
    
    # Distinct colors for 5 categories
    cat_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for attr_name, title, ax in metrics_to_plot:
        names = []
        values = []
        colors = []
        
        for i, (display_name, m) in enumerate(selected):
            val = getattr(m, attr_name, None)
            if val is not None and not np.isnan(val) and not np.isinf(val):
                names.append(display_name)
                values.append(val)
                colors.append(cat_colors[i % len(cat_colors)])
        
        if not values:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=11)
            continue
        
        x = np.arange(len(names))
        bars = ax.bar(x, values, color=colors)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.2e}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=8, rotation=0)
        
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('MSE' if 'mse' in attr_name.lower() else 'Value', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Generation Quality (Best per Kernel Type)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        base_path = str(save_path).replace('.pdf', '').replace('.png', '')
        plt.savefig(f"{base_path}_simplified.pdf", dpi=150, bbox_inches='tight')
        plt.savefig(f"{base_path}_simplified.png", dpi=150, bbox_inches='tight')
        print(f"Saved simplified quality comparison to {base_path}_simplified.pdf")
    
    return fig


def print_quality_table(metrics_list: List[GenerationQualityMetrics]):
    """Print a formatted table of generation quality metrics."""
    if not metrics_list:
        print("No metrics to display.")
        return
    
    print("\n" + "="*100)
    print("GENERATION QUALITY COMPARISON (Pointwise Statistics MSE)")
    print("="*100)
    
    # Header
    print(f"{'Config':<30} {'Mean':<12} {'Variance':<12} {'Skewness':<12} {'Kurtosis':<12} {'Autocorr':<12}")
    print("-"*100)
    
    for m in metrics_list:
        name = m.config_name[:29] if m.config_name else "Unknown"
        mean = f"{m.mean_mse:.2e}" if m.mean_mse is not None else "N/A"
        var = f"{m.variance_mse:.2e}" if m.variance_mse is not None else "N/A"
        skew_val = f"{m.skewness_mse:.2e}" if m.skewness_mse is not None else "N/A"
        kurt = f"{m.kurtosis_mse:.2e}" if m.kurtosis_mse is not None else "N/A"
        auto = f"{m.autocorrelation_mse:.2e}" if m.autocorrelation_mse is not None else "N/A"
        print(f"{name:<30} {mean:<12} {var:<12} {skew_val:<12} {kurt:<12} {auto:<12}")
    
    print("="*100 + "\n")


def print_spectrum_table(metrics_list: List[GenerationQualityMetrics]):
    """Print a formatted table of spectrum quality metrics."""
    if not metrics_list:
        print("No metrics to display.")
        return
    
    print("\n" + "="*80)
    print("SPECTRUM QUALITY COMPARISON")
    print("="*80)
    
    print(f"{'Config':<35} {'Spectrum MSE':<18} {'Spectrum MSE (log)':<18} {'Dom Freq MSE':<15}")
    print("-"*80)
    
    for m in metrics_list:
        name = m.config_name[:34] if m.config_name else "Unknown"
        spec = f"{m.spectrum_mse:.2e}" if m.spectrum_mse is not None else "N/A"
        spec_log = f"{m.spectrum_mse_log:.2e}" if m.spectrum_mse_log is not None else "N/A"
        
        print(f"{name:<35} {spec:<18} {spec_log:<18}")
    
    print("="*80 + "\n")


def print_seasonal_table(metrics_list: List[GenerationQualityMetrics]):
    """Print a formatted table of seasonal pattern metrics."""
    if not metrics_list:
        print("No metrics to display.")
        return
    
    print("\n" + "="*85)
    print("SEASONAL PATTERN QUALITY COMPARISON")
    print("="*85)
    
    print(f"{'Config':<35} {'Seasonal MSE':<18} {'Amplitude Error':<18} {'Phase Corr':<15}")
    print("-"*85)
    
    for m in metrics_list:
        name = m.config_name[:34] if m.config_name else "Unknown"
        seas = f"{m.seasonal_mse:.2e}" if m.seasonal_mse is not None else "N/A"
        amp = f"{m.seasonal_amplitude_error:.4f}" if m.seasonal_amplitude_error is not None else "N/A"
        phase = f"{m.seasonal_phase_correlation:.4f}" if m.seasonal_phase_correlation is not None else "N/A"
        
        print(f"{name:<35} {seas:<18} {amp:<18} {phase:<15}")
    
    print("="*85 + "\n")


def print_comprehensive_table(metrics_list: List[GenerationQualityMetrics]):
    """Print all quality metrics in a comprehensive format."""
    print_quality_table(metrics_list)
    print_spectrum_table(metrics_list)
    print_seasonal_table(metrics_list)


def _convert_to_python_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _convert_to_python_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_to_python_types(item) for item in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_all_metrics_summary(
    metrics_list: List[GenerationQualityMetrics],
    save_path: Path,
):
    """Save comprehensive metrics summary to JSON file."""
    summary = {
        'configs': [m.to_dict() for m in metrics_list],
        'comparison': {},
    }
    
    # Find best config for each metric (lowest MSE / highest correlation)
    metric_attrs = [
        'mean_mse', 'variance_mse', 'skewness_mse', 'kurtosis_mse',
        'autocorrelation_mse', 'density_mse', 'spectrum_mse', 'spectrum_mse_log',
        'seasonal_mse', 'seasonal_amplitude_error',
    ]
    
    for attr in metric_attrs:
        values = [(m.config_name, getattr(m, attr)) for m in metrics_list if getattr(m, attr) is not None]
        if values:
            best = min(values, key=lambda x: x[1])
            summary['comparison'][f'best_{attr}'] = {'config': best[0], 'value': best[1]}
    
    # For correlation, higher is better
    corr_values = [(m.config_name, m.seasonal_phase_correlation) for m in metrics_list 
                   if m.seasonal_phase_correlation is not None]
    if corr_values:
        best = max(corr_values, key=lambda x: x[1])
        summary['comparison']['best_seasonal_phase_correlation'] = {'config': best[0], 'value': best[1]}
    
    # Convert all numpy types to Python native types for JSON serialization
    summary = _convert_to_python_types(summary)
    
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved comprehensive metrics summary to {save_path}")


# =============================================================================
# 2D Field Evaluation (for Navier-Stokes, etc.)
# =============================================================================

def compute_pointwise_statistics_2d(samples: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute pointwise statistics for 2D field samples.
    
    Parameters
    ----------
    samples : Tensor, shape (n_samples, H, W) or (n_samples, C, H, W)
        2D field samples (e.g., vorticity fields from NS)
        
    Returns
    -------
    stats : dict
        Dictionary with:
        - mean: (H, W) or (C, H, W) mean field
        - variance: (H, W) or (C, H, W) variance field
        - std: (H, W) or (C, H, W) standard deviation field
    """
    return {
        'mean': samples.mean(dim=0),
        'variance': samples.var(dim=0),
        'std': samples.std(dim=0),
    }


def compute_all_pointwise_statistics_2d(
    real: torch.Tensor,
    generated: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute MSE between pointwise statistics of real and generated 2D samples.
    
    Analogous to Table 1 in FFM paper but for 2D fields.
    
    Parameters
    ----------
    real : Tensor, shape (n_samples, H, W) or (n_samples, C, H, W)
        Real 2D samples
    generated : Tensor
        Generated 2D samples, same shape structure as real
        
    Returns
    -------
    stats : dict
        Dictionary with MSE for each statistic:
        - mean_mse: MSE between mean fields
        - variance_mse: MSE between variance fields
        - std_mse: MSE between std fields
    """
    real = real.float()
    generated = generated.float()
    
    real_stats = compute_pointwise_statistics_2d(real)
    gen_stats = compute_pointwise_statistics_2d(generated)
    
    mean_mse = ((real_stats['mean'] - gen_stats['mean']) ** 2).mean().item()
    variance_mse = ((real_stats['variance'] - gen_stats['variance']) ** 2).mean().item()
    std_mse = ((real_stats['std'] - gen_stats['std']) ** 2).mean().item()
    
    return {
        'mean_mse': mean_mse,
        'variance_mse': variance_mse,
        'std_mse': std_mse,
    }


def compute_energy_spectrum_2d(
    samples: torch.Tensor,
    average: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D energy spectrum for field samples.
    
    Based on the spectrum function but with cleaner interface.
    
    Parameters
    ----------
    samples : Tensor, shape (n_samples, H, W) or (n_samples, C, H, W)
        2D field samples. If 4D, uses first channel.
    average : bool
        If True, return average spectrum across samples
        
    Returns
    -------
    wavenumbers : ndarray
        Wavenumber values
    spectrum : ndarray
        Energy spectrum (averaged if average=True)
    """
    if samples.ndim == 4:
        samples = samples[:, 0]  # Take first channel
    
    n_samples, s, _ = samples.shape
    assert samples.shape[1] == samples.shape[2], "Expected square fields"
    
    # FFT
    u_fft = torch.fft.fft2(samples)
    
    # Wavenumbers
    k_max = s // 2
    wavenumbers_1d = torch.cat([
        torch.arange(0, k_max),
        torch.arange(-k_max, 0)
    ])
    k_x = wavenumbers_1d.unsqueeze(0).repeat(s, 1).T
    k_y = wavenumbers_1d.unsqueeze(0).repeat(s, 1)
    
    # Sum of absolute wavenumbers
    sum_k = (torch.abs(k_x) + torch.abs(k_y)).numpy()
    
    # Remove symmetric components
    index = -1.0 * np.ones((s, s))
    index[0:k_max + 1, 0:k_max + 1] = sum_k[0:k_max + 1, 0:k_max + 1]
    
    # Bin by wavenumber
    spectrum_arr = np.zeros((n_samples, s))
    for j in range(1, s + 1):
        ind = np.where(index == j)
        if len(ind[0]) > 0:
            spectrum_arr[:, j - 1] = np.sqrt(
                np.abs(u_fft[:, ind[0], ind[1]].sum(axis=1).numpy()) ** 2
            )
    
    # Only keep up to Nyquist
    spectrum_arr = spectrum_arr[:, :s // 2]
    wavenumbers = np.arange(1, s // 2 + 1)
    
    if average:
        return wavenumbers, spectrum_arr.mean(axis=0)
    return wavenumbers, spectrum_arr


def spectrum_mse_2d(
    real: torch.Tensor,
    generated: torch.Tensor,
    log_scale: bool = True,
) -> float:
    """
    Compute MSE between 2D energy spectra of real and generated samples.
    
    Parameters
    ----------
    real : Tensor
        Real 2D samples
    generated : Tensor
        Generated 2D samples
    log_scale : bool
        If True, compute MSE in log scale
        
    Returns
    -------
    mse : float
        Spectrum MSE
    """
    _, spec_real = compute_energy_spectrum_2d(real)
    _, spec_gen = compute_energy_spectrum_2d(generated)
    
    # Ensure same length
    min_len = min(len(spec_real), len(spec_gen))
    spec_real = spec_real[:min_len]
    spec_gen = spec_gen[:min_len]
    
    if log_scale:
        eps = 1e-10
        spec_real = np.log10(spec_real + eps)
        spec_gen = np.log10(spec_gen + eps)
    
    return float(np.mean((spec_real - spec_gen) ** 2))


def plot_samples_2d(
    samples: torch.Tensor,
    n_samples: int = 4,
    title: str = "Samples",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 3),
    cmap: str = 'RdBu_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """
    Plot 2D field samples in a row.
    
    Parameters
    ----------
    samples : Tensor, shape (n_samples, H, W) or (n_samples, C, H, W)
        2D field samples
    n_samples : int
        Number of samples to show
    title : str
        Figure title
    save_path : Path, optional
        Where to save the figure
    cmap : str
        Colormap to use
    vmin, vmax : float, optional
        Color scale limits. If None, uses symmetric limits around 0.
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    if samples.ndim == 4:
        samples = samples[:, 0]  # Take first channel
    
    samples = samples[:n_samples].cpu().numpy()
    n_show = min(n_samples, len(samples))
    
    fig, axes = plt.subplots(1, n_show, figsize=figsize)
    if n_show == 1:
        axes = [axes]
    
    # Determine color limits
    if vmin is None or vmax is None:
        all_max = np.abs(samples).max()
        vmin = -all_max
        vmax = all_max
    
    for i, ax in enumerate(axes):
        im = ax.imshow(samples[i], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(f'Sample {i+1}', fontsize=10)
        ax.axis('off')
    
    fig.colorbar(im, ax=axes, shrink=0.8, aspect=20)
    fig.suptitle(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if str(save_path).endswith('.pdf'):
            plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    
    return fig


def plot_sample_comparison_2d(
    real: torch.Tensor,
    generated: torch.Tensor,
    n_samples: int = 3,
    save_path: Optional[Path] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = 'RdBu_r',
) -> plt.Figure:
    """
    Plot side-by-side comparison of real and generated 2D samples.
    
    Parameters
    ----------
    real : Tensor
        Real 2D samples
    generated : Tensor
        Generated 2D samples
    n_samples : int
        Number of samples to compare
    save_path : Path, optional
        Where to save the figure
    cmap : str
        Colormap to use
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    if real.ndim == 4:
        real = real[:, 0]
    if generated.ndim == 4:
        generated = generated[:, 0]
    
    real = real[:n_samples].cpu().numpy()
    generated = generated[:n_samples].cpu().numpy()
    n_show = min(n_samples, len(real), len(generated))
    
    if figsize is None:
        figsize = (4 * n_show, 8)
    
    fig, axes = plt.subplots(2, n_show, figsize=figsize)
    if n_show == 1:
        axes = axes.reshape(2, 1)
    
    # Determine color limits (symmetric around 0)
    all_max = max(np.abs(real).max(), np.abs(generated).max())
    vmin, vmax = -all_max, all_max
    
    for i in range(n_show):
        # Real
        im = axes[0, i].imshow(real[i], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        axes[0, i].set_title(f'Real {i+1}', fontsize=10)
        axes[0, i].axis('off')
        
        # Generated
        axes[1, i].imshow(generated[i], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        axes[1, i].set_title(f'Generated {i+1}', fontsize=10)
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Ground Truth', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Generated', fontsize=11, fontweight='bold')
    
    fig.colorbar(im, ax=axes, shrink=0.6, aspect=30)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if str(save_path).endswith('.pdf'):
            plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        print(f"Saved sample comparison to {save_path}")
    
    return fig


def plot_mean_variance_comparison_2d(
    real: torch.Tensor,
    generated: torch.Tensor,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 6),
    cmap_mean: str = 'RdBu_r',
    cmap_var: str = 'viridis',
) -> plt.Figure:
    """
    Compare mean and variance fields between real and generated samples.
    
    Parameters
    ----------
    real : Tensor
        Real 2D samples
    generated : Tensor
        Generated 2D samples
    save_path : Path, optional
        Where to save the figure
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    if real.ndim == 4:
        real = real[:, 0]
    if generated.ndim == 4:
        generated = generated[:, 0]
    
    real_stats = compute_pointwise_statistics_2d(real)
    gen_stats = compute_pointwise_statistics_2d(generated)
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Mean fields
    mean_max = max(real_stats['mean'].abs().max(), gen_stats['mean'].abs().max())
    
    im1 = axes[0, 0].imshow(real_stats['mean'].cpu().numpy(), cmap=cmap_mean, 
                            vmin=-mean_max, vmax=mean_max, origin='lower')
    axes[0, 0].set_title('Real Mean', fontsize=11)
    axes[0, 0].axis('off')
    
    im2 = axes[0, 1].imshow(gen_stats['mean'].cpu().numpy(), cmap=cmap_mean,
                            vmin=-mean_max, vmax=mean_max, origin='lower')
    axes[0, 1].set_title('Generated Mean', fontsize=11)
    axes[0, 1].axis('off')
    
    # Mean difference
    mean_diff = (real_stats['mean'] - gen_stats['mean']).cpu().numpy()
    diff_max = np.abs(mean_diff).max()
    im3 = axes[0, 2].imshow(mean_diff, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max, origin='lower')
    axes[0, 2].set_title(f'Mean Diff (MSE={np.mean(mean_diff**2):.2e})', fontsize=11)
    axes[0, 2].axis('off')
    
    fig.colorbar(im1, ax=axes[0, :2], shrink=0.7, aspect=20)
    fig.colorbar(im3, ax=axes[0, 2], shrink=0.7, aspect=20)
    
    # Variance fields
    var_max = max(real_stats['variance'].max(), gen_stats['variance'].max())
    
    im4 = axes[1, 0].imshow(real_stats['variance'].cpu().numpy(), cmap=cmap_var,
                            vmin=0, vmax=var_max, origin='lower')
    axes[1, 0].set_title('Real Variance', fontsize=11)
    axes[1, 0].axis('off')
    
    im5 = axes[1, 1].imshow(gen_stats['variance'].cpu().numpy(), cmap=cmap_var,
                            vmin=0, vmax=var_max, origin='lower')
    axes[1, 1].set_title('Generated Variance', fontsize=11)
    axes[1, 1].axis('off')
    
    # Variance difference
    var_diff = (real_stats['variance'] - gen_stats['variance']).cpu().numpy()
    var_diff_max = np.abs(var_diff).max()
    im6 = axes[1, 2].imshow(var_diff, cmap='RdBu_r', vmin=-var_diff_max, vmax=var_diff_max, origin='lower')
    axes[1, 2].set_title(f'Var Diff (MSE={np.mean(var_diff**2):.2e})', fontsize=11)
    axes[1, 2].axis('off')
    
    fig.colorbar(im4, ax=axes[1, :2], shrink=0.7, aspect=20)
    fig.colorbar(im6, ax=axes[1, 2], shrink=0.7, aspect=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if str(save_path).endswith('.pdf'):
            plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        print(f"Saved mean/variance comparison to {save_path}")
    
    return fig


def compare_spectra_2d(
    real: torch.Tensor,
    generated: Union[torch.Tensor, List[torch.Tensor]],
    config_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
    top_k: int = 5,
    save_top_k: bool = True,
) -> plt.Figure:
    """
    Plot 2D energy spectrum comparison between real and generated samples.
    
    Parameters
    ----------
    real : Tensor
        Real 2D samples
    generated : Tensor or List[Tensor]
        Generated samples (single or multiple configurations)
    config_names : List[str], optional
        Names for each generated configuration
    save_path : Path, optional
        Where to save the figure
    top_k : int
        Number of top configurations for additional top-k plot
    save_top_k : bool
        Whether to generate additional top-k plot
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    # Compute real spectrum
    k_real, spec_real = compute_energy_spectrum_2d(real)
    
    # Handle single or multiple generated samples
    if isinstance(generated, torch.Tensor):
        generated = [generated]
        config_names = config_names or ['Generated']
    elif config_names is None:
        config_names = [f'Config {i+1}' for i in range(len(generated))]
    
    # Compute spectrum for each configuration
    all_specs = []
    spectrum_mses = []
    for gen in generated:
        k_gen, spec_gen = compute_energy_spectrum_2d(gen)
        all_specs.append((k_gen, spec_gen))
        # Compute MSE (log scale)
        mse = spectrum_mse_2d(real, gen, log_scale=True)
        spectrum_mses.append(mse)
    
    # =========================================================================
    # Plot 1: All configurations
    # =========================================================================
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.semilogy(k_real, spec_real, 'k-', linewidth=2.5, label='Ground Truth', alpha=0.9)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(generated)))
    
    for i, (name, color) in enumerate(zip(config_names, colors)):
        k_gen, spec_gen = all_specs[i]
        ax.semilogy(k_gen, spec_gen, '--', color=color, linewidth=1.5, label=name, alpha=0.7)
    
    ax.set_xlabel('Wavenumber k', fontsize=11)
    ax.set_ylabel('Energy E(k)', fontsize=11)
    ax.set_title('Energy Spectrum Comparison (All Configurations)', fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if str(save_path).endswith('.pdf'):
            plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        print(f"Saved 2D spectrum comparison to {save_path}")
    
    # =========================================================================
    # Plot 2: Top-K configurations (by spectrum MSE)
    # =========================================================================
    if save_top_k and save_path and len(generated) > 1:
        sorted_indices = np.argsort(spectrum_mses)[:top_k]
        
        fig2, ax2 = plt.subplots(figsize=figsize)
        
        ax2.semilogy(k_real, spec_real, 'k-', linewidth=2.5, label='Ground Truth', alpha=0.9)
        
        top_colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_indices)))
        
        for rank, (idx, color) in enumerate(zip(sorted_indices, top_colors)):
            k_gen, spec_gen = all_specs[idx]
            mse = spectrum_mses[idx]
            label = f"#{rank+1} {config_names[idx]} (MSE={mse:.2e})"
            ax2.semilogy(k_gen, spec_gen, '-', color=color, linewidth=2, label=label, alpha=0.85)
        
        ax2.set_xlabel('Wavenumber k', fontsize=11)
        ax2.set_ylabel('Energy E(k)', fontsize=11)
        ax2.set_title(f'Energy Spectrum Comparison (Top {min(top_k, len(generated))} by Spectrum MSE)', fontsize=12)
        ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=True)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        base_path = str(save_path).replace('.pdf', '').replace('.png', '')
        plt.savefig(f"{base_path}_top{top_k}.pdf", dpi=150, bbox_inches='tight')
        plt.savefig(f"{base_path}_top{top_k}.png", dpi=150, bbox_inches='tight')
        print(f"Saved top-{top_k} 2D spectrum comparison to {base_path}_top{top_k}.pdf")
        plt.close(fig2)
    
    return fig


class GenerationQualityMetrics2D:
    """
    Container for 2D generation quality metrics (for Navier-Stokes, etc.).
    """
    
    def __init__(
        self,
        config_name: str = "",
        ot_kernel: str = "",
        ot_method: str = "",
        ot_coupling: str = "",
        use_ot: bool = False,
    ):
        self.config_name = config_name
        self.ot_kernel = ot_kernel
        self.ot_method = ot_method
        self.ot_coupling = ot_coupling
        self.use_ot = use_ot
        
        # Pointwise statistics MSEs
        self.mean_mse: Optional[float] = None
        self.variance_mse: Optional[float] = None
        self.std_mse: Optional[float] = None
        
        # Spectrum metrics
        self.spectrum_mse: Optional[float] = None
        self.spectrum_mse_log: Optional[float] = None
        
        # Density metric
        self.density_mse: Optional[float] = None
        
        # Training metrics
        self.final_train_loss: Optional[float] = None
        self.total_train_time: Optional[float] = None
        
        # Convergence metrics
        self.convergence_rate: Optional[float] = None
        self.final_stability: Optional[float] = None
        self.epochs_to_90pct: Optional[int] = None
    
    def compute_from_samples(
        self,
        real: torch.Tensor,
        generated: torch.Tensor,
    ):
        """Compute all 2D statistics from real and generated samples."""
        stats = compute_all_pointwise_statistics_2d(real, generated)
        self.mean_mse = stats['mean_mse']
        self.variance_mse = stats['variance_mse']
        self.std_mse = stats['std_mse']
        
        # Spectrum metrics
        try:
            self.spectrum_mse = spectrum_mse_2d(real, generated, log_scale=False)
            self.spectrum_mse_log = spectrum_mse_2d(real, generated, log_scale=True)
        except Exception:
            self.spectrum_mse = None
            self.spectrum_mse_log = None
        
        # Density metric
        try:
            self.density_mse = density_mse(real, generated)
        except Exception:
            self.density_mse = None
    
    def set_convergence_metrics(self, losses: List[float]):
        """Compute and set convergence metrics from training losses."""
        conv_metrics = compute_convergence_metrics(losses)
        self.convergence_rate = conv_metrics['convergence_rate']
        self.final_stability = conv_metrics['final_stability']
        self.epochs_to_90pct = conv_metrics['epochs_to_90pct']
    
    @staticmethod
    def _to_python_float(val):
        """Convert numpy/torch floats to Python float for JSON serialization."""
        if val is None:
            return None
        if hasattr(val, 'item'):
            return float(val.item())
        if isinstance(val, (np.floating, np.integer)):
            return float(val)
        return float(val) if val is not None else None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'config_name': self.config_name,
            'ot_kernel': self.ot_kernel,
            'ot_method': self.ot_method,
            'ot_coupling': self.ot_coupling,
            'use_ot': self.use_ot,
            'mean_mse': self._to_python_float(self.mean_mse),
            'variance_mse': self._to_python_float(self.variance_mse),
            'std_mse': self._to_python_float(self.std_mse),
            'spectrum_mse': self._to_python_float(self.spectrum_mse),
            'spectrum_mse_log': self._to_python_float(self.spectrum_mse_log),
            'density_mse': self._to_python_float(self.density_mse),
            'final_train_loss': self._to_python_float(self.final_train_loss),
            'total_train_time': self._to_python_float(self.total_train_time),
            'convergence_rate': self._to_python_float(self.convergence_rate),
            'final_stability': self._to_python_float(self.final_stability),
            'epochs_to_90pct': self.epochs_to_90pct,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'GenerationQualityMetrics2D':
        """Load from dictionary."""
        obj = cls(
            config_name=d.get('config_name', ''),
            ot_kernel=d.get('ot_kernel', ''),
            ot_method=d.get('ot_method', ''),
            ot_coupling=d.get('ot_coupling', ''),
            use_ot=d.get('use_ot', False),
        )
        obj.mean_mse = d.get('mean_mse')
        obj.variance_mse = d.get('variance_mse')
        obj.std_mse = d.get('std_mse')
        obj.spectrum_mse = d.get('spectrum_mse')
        obj.spectrum_mse_log = d.get('spectrum_mse_log')
        obj.density_mse = d.get('density_mse')
        obj.final_train_loss = d.get('final_train_loss')
        obj.total_train_time = d.get('total_train_time')
        obj.convergence_rate = d.get('convergence_rate')
        obj.final_stability = d.get('final_stability')
        obj.epochs_to_90pct = d.get('epochs_to_90pct')
        return obj
    
    def save(self, path: Path):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'GenerationQualityMetrics2D':
        """Load from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


def compare_generation_quality_2d(
    metrics_list: List[GenerationQualityMetrics2D],
    save_path: Path = None,
    figsize: Tuple[int, int] = (14, 5),
    top_k: int = 5,
    show_all: bool = False,
) -> plt.Figure:
    """
    Create comparison visualization of 2D generation quality across configurations.
    
    Uses a 1×4 grid layout with 4 core metrics:
    - Mean MSE, Variance MSE, Spectrum MSE, Density MSE
    
    Parameters
    ----------
    metrics_list : List[GenerationQualityMetrics2D]
        List of metrics from different configurations
    save_path : Path, optional
        Where to save the figure
    top_k : int
        Number of top configurations to show per metric
    show_all : bool
        If True, show all configurations instead of top-k
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    if not metrics_list:
        return None
    
    # Extract common prefix
    all_config_names = [m.config_name for m in metrics_list]
    common_prefix, _ = _extract_common_prefix(all_config_names)
    
    # 1×4 layout
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    metrics_to_plot = [
        ('mean_mse', 'Mean MSE', axes[0]),
        ('variance_mse', 'Variance MSE', axes[1]),
        ('spectrum_mse_log', 'Spectrum MSE (log)', axes[2]),
        ('density_mse', 'Density MSE', axes[3]),
    ]
    
    top_k_colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, top_k))
    
    for attr_name, title, ax in metrics_to_plot:
        valid_entries = []
        for m in metrics_list:
            val = getattr(m, attr_name, None)
            if val is not None and not np.isnan(val) and not np.isinf(val):
                short_name = m.config_name
                if common_prefix and short_name.startswith(common_prefix + '_'):
                    short_name = short_name[len(common_prefix) + 1:]
                valid_entries.append((short_name, val, m))
        
        if not valid_entries:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=11)
            continue
        
        # Sort ascending (lower is better for MSE)
        valid_entries.sort(key=lambda x: x[1])
        
        if not show_all:
            display_entries = valid_entries[:top_k]
        else:
            display_entries = valid_entries
        
        names = [e[0] for e in display_entries]
        values = [e[1] for e in display_entries]
        x = np.arange(len(names))
        
        if len(display_entries) <= top_k:
            colors = top_k_colors[:len(display_entries)]
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, len(display_entries)))
        
        bars = ax.bar(x, values, color=colors)
        
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.annotate(f"#{i+1}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        if not show_all and len(valid_entries) > top_k:
            ax.set_title(f'{title} (Top {top_k}/{len(valid_entries)})', fontsize=11)
        else:
            ax.set_title(title, fontsize=11)
        
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        
        if all(v > 0 for v in values):
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
    
    if common_prefix:
        title_text = f"2D Quality Metrics: {common_prefix.replace('_', ' ').title()}"
        fig.suptitle(title_text, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if str(save_path).endswith('.pdf'):
            plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        print(f"Saved 2D quality comparison to {save_path}")
    
    return fig


def print_quality_table_2d(metrics_list: List[GenerationQualityMetrics2D]):
    """Print a formatted table of 2D generation quality metrics."""
    if not metrics_list:
        print("No metrics to display.")
        return
    
    print("\n" + "="*100)
    print("2D GENERATION QUALITY COMPARISON")
    print("="*100)
    
    print(f"{'Config':<35} {'Mean MSE':<14} {'Var MSE':<14} {'Spectrum MSE':<14} {'Density MSE':<14}")
    print("-"*100)
    
    for m in metrics_list:
        name = m.config_name[:34] if m.config_name else "Unknown"
        mean = f"{m.mean_mse:.2e}" if m.mean_mse is not None else "N/A"
        var = f"{m.variance_mse:.2e}" if m.variance_mse is not None else "N/A"
        spec = f"{m.spectrum_mse_log:.2e}" if m.spectrum_mse_log is not None else "N/A"
        dens = f"{m.density_mse:.2e}" if m.density_mse is not None else "N/A"
        
        print(f"{name:<35} {mean:<14} {var:<14} {spec:<14} {dens:<14}")
    
    print("="*100 + "\n")


def save_all_metrics_summary_2d(
    metrics_list: List[GenerationQualityMetrics2D],
    save_path: Path,
):
    """Save comprehensive 2D metrics summary to JSON file."""
    summary = {
        'configs': [m.to_dict() for m in metrics_list],
        'comparison': {},
    }
    
    metric_attrs = ['mean_mse', 'variance_mse', 'std_mse', 'spectrum_mse', 
                    'spectrum_mse_log', 'density_mse']
    
    for attr in metric_attrs:
        values = [(m.config_name, getattr(m, attr)) for m in metrics_list 
                  if getattr(m, attr) is not None]
        if values:
            best = min(values, key=lambda x: x[1])
            summary['comparison'][f'best_{attr}'] = {'config': best[0], 'value': best[1]}
    
    summary = _convert_to_python_types(summary)
    
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved 2D metrics summary to {save_path}")


# =============================================================================
# Original functions (kept for backward compatibility)
# =============================================================================

def distribution_kde(real, gen, n=1000, bw=0.5, save_path=None, gridsize=200, cutoff=3):
    # Compares the distribution of pointwise values between real and generated
    # u is data, n is how large of a subset to look at (useful for very large # of samples)
    
    real = real.flatten()
    idx = torch.randperm(real.numel())
    real = real[idx[:n]]
    
    gen = gen.flatten()
    idx = torch.randperm(gen.numel())
    gen = gen[idx[:n]]

    kernel_real = gaussian_kde(real, bw_method=bw)
    kernel_gen = gaussian_kde(gen, bw_method=bw)

    min_val = torch.min(torch.min(real), torch.min(gen))
    max_val = torch.max(torch.max(real), torch.max(gen))

    min_cutoff = min_val - cutoff * bw * np.abs(min_val)
    max_cutoff = max_val + cutoff * bw * np.abs(max_val)

    grid = torch.linspace(min_cutoff, max_cutoff, gridsize)

    y_real = kernel_real(grid)
    y_gen = kernel_gen(grid)

    mse = np.mean((y_real - y_gen)**2)

    fig, ax = plt.subplots()
    ax.plot(grid, y_real, label='Ground Truth')
    ax.plot(grid, y_gen, label='Generated', linestyle='--')

    ax.legend()

    ax.set_title(f'MSE: {mse:.4e}')
    
    plt.legend()
    if save_path:
        plt.savefig(save_path)

def spectrum(u, s):
    # https://github.com/neuraloperator/markov_neural_operator/blob/main/visualize_navier_stokes2d.ipynb
    # s is the resolution of u, i.e. u is shape (batch_size, s, s)

    T = u.shape[0]
    u = u.reshape(T, s, s)
    u = torch.fft.fft2(u)

    # 2d wavenumbers following Pytorch fft convention
    k_max = s // 2
    wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1), \
                            torch.arange(start=-k_max, end=0, step=1)), 0).repeat(s, 1)
    k_x = wavenumers.transpose(0, 1)
    k_y = wavenumers
    
    # wavenumers = [0, 1, 2, n/2-1, -n/2, -n/2 + 1, ..., -3, -2, -1]
    
    # Sum wavenumbers
    sum_k = torch.abs(k_x) + torch.abs(k_y)
    sum_k = sum_k.numpy()
    
    # Remove symmetric components from wavenumbers
    index = -1.0 * np.ones((s, s))
    index[0:k_max + 1, 0:k_max + 1] = sum_k[0:k_max + 1, 0:k_max + 1]
    
    spectrum = np.zeros((T, s))
    for j in range(1, s + 1):
        ind = np.where(index == j)
        spectrum[:, j - 1] = np.sqrt( (u[:, ind[0], ind[1]].sum(axis=1)).abs() ** 2)
        
    spectrum = spectrum.mean(axis=0)
    spectrum = spectrum[:s//2]

    return spectrum


def compare_spectra(real, gen, save_path=None):
    s = real.shape[-1]
    spec_true = spectrum(real, s)
    spec_gen = spectrum(gen, s)

    mse = np.mean( (spec_true-spec_gen)**2 )

    fig, ax = plt.subplots()

    ax.semilogy(spec_true, label='Ground Truth')
    ax.semilogy(spec_gen, label='Generated')
    ax.legend()

    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Energy')

    ax.set_title(f'MSE: {mse:4e}')

    if save_path:
        plt.savefig(save_path)    


def spectra_mse(real, gen):
    s = real.shape[-1]
    spec_true = spectrum(real, s)
    spec_gen = spectrum(gen, s)

    mse = np.mean( (spec_true-spec_gen)**2 )
    return mse

def density_mse(real, gen, bw=0.5, gridsize=200, cutoff=3):
    # Compares the distribution of pointwise values between real and generated
    # u is data, n is how large of a subset to look at (useful for very large # of samples)

    real = real.flatten()
    gen = gen.flatten()

    kernel_real = gaussian_kde(real, bw_method=bw)
    kernel_gen = gaussian_kde(gen, bw_method=bw)

    min_val = torch.min(torch.min(real), torch.min(gen))
    max_val = torch.max(torch.max(real), torch.max(gen))

    min_cutoff = min_val - cutoff * bw * np.abs(min_val)
    max_cutoff = max_val + cutoff * bw * np.abs(max_val)

    grid = torch.linspace(min_cutoff, max_cutoff, gridsize)

    y_real = kernel_real(grid)
    y_gen = kernel_gen(grid)

    mse = np.mean((y_real - y_gen)**2)

    return mse