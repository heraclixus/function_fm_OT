"""
OT Monitoring Utilities for Functional Flow Matching.

This module provides utilities to monitor and compare OT vs non-OT training:
1. Convergence speed tracking
2. Interpolation path length computation
3. Gradient variance estimation
4. Visualization utilities for comparison

Usage:
    from util.ot_monitoring import TrainingMonitor, compare_training_runs
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json


@dataclass
class TrainingMetrics:
    """Container for training metrics collected during a run."""
    
    # Basic training metrics
    train_losses: List[float] = field(default_factory=list)
    test_losses: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    
    # OT-specific metrics
    path_lengths: List[float] = field(default_factory=list)  # Mean interpolation path length per epoch
    gradient_variances: List[float] = field(default_factory=list)  # Gradient variance per epoch
    gradient_norms: List[float] = field(default_factory=list)  # Mean gradient norm per epoch
    
    # Per-batch metrics (for finer analysis)
    batch_losses: List[float] = field(default_factory=list)
    batch_path_lengths: List[float] = field(default_factory=list)
    batch_grad_norms: List[float] = field(default_factory=list)
    
    # Metadata
    method_name: str = ""
    use_ot: bool = False
    ot_kernel: str = ""
    
    @staticmethod
    def _to_python_list(lst):
        """Convert list of numpy/torch floats to Python floats for JSON."""
        if not lst:
            return lst
        result = []
        for val in lst:
            if hasattr(val, 'item'):
                result.append(float(val.item()))
            elif isinstance(val, (np.floating, np.integer)):
                result.append(float(val))
            else:
                result.append(float(val) if val is not None else None)
        return result
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'train_losses': self._to_python_list(self.train_losses),
            'test_losses': self._to_python_list(self.test_losses),
            'epoch_times': self._to_python_list(self.epoch_times),
            'path_lengths': self._to_python_list(self.path_lengths),
            'gradient_variances': self._to_python_list(self.gradient_variances),
            'gradient_norms': self._to_python_list(self.gradient_norms),
            'method_name': self.method_name,
            'use_ot': self.use_ot,
            'ot_kernel': self.ot_kernel,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'TrainingMetrics':
        """Load from dictionary."""
        return cls(**d)
    
    def save(self, path: Path):
        """Save metrics to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'TrainingMetrics':
        """Load metrics from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


class TrainingMonitor:
    """
    Monitor for tracking OT vs non-OT training metrics.
    
    Tracks:
    - Convergence speed via loss curves
    - Interpolation path lengths (should be shorter with OT)
    - Gradient variance (should be lower with OT)
    
    Usage:
        monitor = TrainingMonitor(method_name="ffm_eucl_ot", use_ot=True)
        
        for epoch in range(epochs):
            for batch in loader:
                # ... training code ...
                monitor.log_batch(
                    loss=loss.item(),
                    x_data=x_data,
                    z_noise=z_paired,
                    gradients=get_gradients(model)
                )
            monitor.end_epoch(epoch_time=t1-t0)
        
        metrics = monitor.get_metrics()
        monitor.save(save_path / 'metrics.json')
    """
    
    def __init__(
        self,
        method_name: str = "",
        use_ot: bool = False,
        ot_kernel: str = "",
        track_batch_metrics: bool = False,
    ):
        """
        Initialize the training monitor.
        
        Parameters
        ----------
        method_name : str
            Name of the method (e.g., "ffm_indep", "ffm_eucl_ot")
        use_ot : bool
            Whether OT pairing is used
        ot_kernel : str
            Type of OT kernel (e.g., "euclidean", "rbf", "signature")
        track_batch_metrics : bool
            Whether to track per-batch metrics (more memory intensive)
        """
        self.metrics = TrainingMetrics(
            method_name=method_name,
            use_ot=use_ot,
            ot_kernel=ot_kernel,
        )
        self.track_batch_metrics = track_batch_metrics
        
        # Accumulators for current epoch
        self._epoch_losses = []
        self._epoch_path_lengths = []
        self._epoch_grad_norms = []
    
    def compute_path_length(
        self,
        x_data: torch.Tensor,
        z_noise: torch.Tensor,
        sigma_min: float = 1e-4,
    ) -> float:
        """
        Compute the mean interpolation path length for the batch.
        
        For OT path: x_t = t * x_data + (1 - (1-σ_min)*t) * z_noise
        The path length is approximately ||x_data - z_noise|| (for small σ_min).
        
        Parameters
        ----------
        x_data : Tensor, shape (B, C, *dims)
            Data samples (endpoint at t=1)
        z_noise : Tensor, shape (B, C, *dims)
            Base samples (endpoint at t=0)
        sigma_min : float
            Minimum noise level
            
        Returns
        -------
        mean_path_length : float
            Mean L2 path length across the batch
        """
        # Flatten to (B, -1) for distance computation
        x_flat = x_data.reshape(x_data.shape[0], -1)
        z_flat = z_noise.reshape(z_noise.shape[0], -1)
        
        # L2 distance between paired endpoints
        # This approximates the path length for the linear interpolation
        path_lengths = torch.norm(x_flat - z_flat, dim=1)
        
        return path_lengths.mean().item()
    
    def compute_gradient_stats(
        self,
        model: torch.nn.Module,
    ) -> Tuple[float, float]:
        """
        Compute gradient norm and prepare for variance estimation.
        
        Parameters
        ----------
        model : nn.Module
            The model after backward() has been called
            
        Returns
        -------
        grad_norm : float
            Total gradient norm
        grad_vector : Tensor or None
            Flattened gradient vector (for variance computation)
        """
        total_norm = 0.0
        grad_list = []
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                grad_list.append(p.grad.data.flatten())
        
        total_norm = np.sqrt(total_norm)
        
        return total_norm
    
    def log_batch(
        self,
        loss: float,
        x_data: torch.Tensor = None,
        z_noise: torch.Tensor = None,
        model: torch.nn.Module = None,
        sigma_min: float = 1e-4,
    ):
        """
        Log metrics for a single batch.
        
        Parameters
        ----------
        loss : float
            Batch loss value
        x_data : Tensor, optional
            Data samples for path length computation
        z_noise : Tensor, optional
            Noise samples for path length computation
        model : nn.Module, optional
            Model for gradient statistics (after backward)
        sigma_min : float
            Minimum noise level for path computation
        """
        self._epoch_losses.append(loss)
        
        if self.track_batch_metrics:
            self.metrics.batch_losses.append(loss)
        
        # Compute path length if samples provided
        if x_data is not None and z_noise is not None:
            path_length = self.compute_path_length(x_data, z_noise, sigma_min)
            self._epoch_path_lengths.append(path_length)
            
            if self.track_batch_metrics:
                self.metrics.batch_path_lengths.append(path_length)
        
        # Compute gradient stats if model provided
        if model is not None:
            grad_norm = self.compute_gradient_stats(model)
            self._epoch_grad_norms.append(grad_norm)
            
            if self.track_batch_metrics:
                self.metrics.batch_grad_norms.append(grad_norm)
    
    def end_epoch(self, epoch_time: float = 0.0, test_loss: float = None):
        """
        Finalize metrics for the current epoch.
        
        Parameters
        ----------
        epoch_time : float
            Time taken for this epoch
        test_loss : float, optional
            Test loss if available
        """
        # Aggregate epoch losses
        if self._epoch_losses:
            epoch_loss = np.mean(self._epoch_losses)
            self.metrics.train_losses.append(epoch_loss)
        
        # Aggregate path lengths
        if self._epoch_path_lengths:
            mean_path_length = np.mean(self._epoch_path_lengths)
            self.metrics.path_lengths.append(mean_path_length)
        
        # Aggregate gradient statistics
        if self._epoch_grad_norms:
            mean_grad_norm = np.mean(self._epoch_grad_norms)
            grad_variance = np.var(self._epoch_grad_norms)
            self.metrics.gradient_norms.append(mean_grad_norm)
            self.metrics.gradient_variances.append(grad_variance)
        
        # Record epoch time
        self.metrics.epoch_times.append(epoch_time)
        
        # Record test loss if provided
        if test_loss is not None:
            self.metrics.test_losses.append(test_loss)
        
        # Reset accumulators
        self._epoch_losses = []
        self._epoch_path_lengths = []
        self._epoch_grad_norms = []
    
    def get_metrics(self) -> TrainingMetrics:
        """Return collected metrics."""
        return self.metrics
    
    def save(self, path: Path):
        """Save metrics to file."""
        self.metrics.save(path)


def _extract_short_method_name(method_name: str) -> str:
    """Extract short method name by removing common dataset prefixes."""
    if not method_name:
        return method_name
    
    parts = method_name.split('_')
    # Find where the OT method starts
    for i, part in enumerate(parts):
        if part in ['independent', 'gaussian', 'euclidean', 'rbf', 'signature']:
            return '_'.join(parts[i:])
    return method_name


def select_best_per_kernel(
    training_metrics_list: List['TrainingMetrics'],
    quality_metrics_list: List = None,
    metric_key: str = 'mean_mse',
) -> Tuple[List['TrainingMetrics'], List[str]]:
    """
    Select the best configuration from each kernel type.
    
    Groups configurations by kernel type (independent, gaussian, euclidean, rbf, signature)
    and selects the one with the best quality metric.
    
    Parameters
    ----------
    training_metrics_list : List[TrainingMetrics]
        List of training metrics from different configurations
    quality_metrics_list : List, optional
        List of quality metrics (must have same order as training_metrics_list).
        If None, uses final training loss as the selection criterion.
    metric_key : str
        Which quality metric to use for selection (e.g., 'mean_mse', 'spectrum_mse')
        
    Returns
    -------
    selected_metrics : List[TrainingMetrics]
        Filtered list with one config per kernel type
    display_names : List[str]
        Simplified display names for plotting
    """
    # Define kernel categories and their display names
    kernel_categories = {
        'independent': 'Independent',
        'gaussian': 'Gaussian OT',
        'euclidean': 'Euclidean OT',
        'rbf': 'RBF OT',
        'signature': 'Signature OT',
    }
    
    # Group by kernel type
    groups = {cat: [] for cat in kernel_categories}
    
    for i, tm in enumerate(training_metrics_list):
        name = tm.method_name.lower()
        
        if 'independent' in name:
            groups['independent'].append((i, tm))
        elif 'gaussian' in name:
            groups['gaussian'].append((i, tm))
        elif 'signature' in name:
            groups['signature'].append((i, tm))
        elif 'rbf' in name:
            groups['rbf'].append((i, tm))
        elif 'euclidean' in name:
            groups['euclidean'].append((i, tm))
    
    selected_metrics = []
    display_names = []
    
    for cat, cat_name in kernel_categories.items():
        if not groups[cat]:
            continue
        
        if len(groups[cat]) == 1:
            # Only one config in this category
            idx, tm = groups[cat][0]
            selected_metrics.append(tm)
            display_names.append(cat_name)
        else:
            # Multiple configs - select best by quality metric or final loss
            if quality_metrics_list is not None:
                # Use quality metric
                best_idx = None
                best_val = float('inf')
                best_tm = None
                
                for idx, tm in groups[cat]:
                    qm = quality_metrics_list[idx]
                    val = getattr(qm, metric_key, None)
                    if val is not None and val < best_val:
                        best_val = val
                        best_idx = idx
                        best_tm = tm
                
                if best_tm is not None:
                    selected_metrics.append(best_tm)
                    display_names.append(cat_name)
            else:
                # Use final training loss
                best_tm = min(groups[cat], key=lambda x: x[1].train_losses[-1] if x[1].train_losses else float('inf'))[1]
                selected_metrics.append(best_tm)
                display_names.append(cat_name)
    
    return selected_metrics, display_names


def compare_training_runs_simplified(
    training_metrics_list: List['TrainingMetrics'],
    quality_metrics_list: List = None,
    metric_key: str = 'mean_mse',
    save_path: Path = None,
    figsize: Tuple[int, int] = (10, 6),
) -> List[plt.Figure]:
    """
    Compare training runs with simplified view (best of each kernel type only).
    
    Shows only: Independent, Gaussian OT, Euclidean OT, RBF OT, Signature OT
    (selecting the best configuration from each category based on quality metrics)
    
    Parameters
    ----------
    training_metrics_list : List[TrainingMetrics]
        List of all training metrics
    quality_metrics_list : List, optional
        List of quality metrics for selecting best configs
    metric_key : str
        Quality metric to use for selection
    save_path : Path, optional
        Base path for saving figures
    figsize : tuple
        Figure size
        
    Returns
    -------
    figs : List[matplotlib.Figure]
        List of figures
    """
    # Select best from each kernel type
    selected_metrics, display_names = select_best_per_kernel(
        training_metrics_list, quality_metrics_list, metric_key
    )
    
    # Override method names with simplified display names
    for tm, name in zip(selected_metrics, display_names):
        tm._display_name = name
    
    # Use distinct colors for 5 categories
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Blue, Orange, Green, Red, Purple
    
    figs = []
    base_path = None
    if save_path:
        base_path = str(save_path).replace('.pdf', '').replace('.png', '')
    
    # =========================================================================
    # Plot 1: Loss curves
    # =========================================================================
    fig1, ax = plt.subplots(figsize=figsize)
    for metrics, color, name in zip(selected_metrics, colors[:len(selected_metrics)], display_names):
        if metrics.train_losses:
            ax.semilogy(metrics.train_losses, label=name, color=color, linewidth=2, alpha=0.9)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title('Training Loss Curves', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig1.tight_layout()
    figs.append(fig1)
    
    if base_path:
        fig1.savefig(f"{base_path}_loss.pdf", dpi=150, bbox_inches='tight')
        fig1.savefig(f"{base_path}_loss.png", dpi=150, bbox_inches='tight')
    
    # =========================================================================
    # Plot 2: Path lengths
    # =========================================================================
    fig2, ax = plt.subplots(figsize=figsize)
    for metrics, color, name in zip(selected_metrics, colors[:len(selected_metrics)], display_names):
        if metrics.path_lengths:
            ax.plot(metrics.path_lengths, label=name, color=color, linewidth=2, alpha=0.9)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean Path Length', fontsize=12)
    ax.set_title('Interpolation Path Lengths (shorter = straighter paths)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    figs.append(fig2)
    
    if base_path:
        fig2.savefig(f"{base_path}_path_length.pdf", dpi=150, bbox_inches='tight')
        fig2.savefig(f"{base_path}_path_length.png", dpi=150, bbox_inches='tight')
    
    # =========================================================================
    # Plot 3: Gradient variance
    # =========================================================================
    has_grad_var = any(m.gradient_variances for m in selected_metrics)
    if has_grad_var:
        fig3, ax = plt.subplots(figsize=figsize)
        for metrics, color, name in zip(selected_metrics, colors[:len(selected_metrics)], display_names):
            if metrics.gradient_variances:
                ax.semilogy(metrics.gradient_variances, label=name, color=color, linewidth=2, alpha=0.9)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Gradient Variance (log scale)', fontsize=12)
        ax.set_title('Gradient Variance (lower = more stable)', fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        fig3.tight_layout()
        figs.append(fig3)
        
        if base_path:
            fig3.savefig(f"{base_path}_grad_variance.pdf", dpi=150, bbox_inches='tight')
            fig3.savefig(f"{base_path}_grad_variance.png", dpi=150, bbox_inches='tight')
    
    return figs


def compare_training_runs(
    metrics_list: List[TrainingMetrics],
    save_path: Path = None,
    figsize: Tuple[int, int] = (10, 6),
) -> List[plt.Figure]:
    """
    Compare multiple training runs with different OT configurations.
    
    Creates 4 separate figures:
    - Loss curves (convergence speed)
    - Path lengths (shorter = better OT matching)
    - Gradient variance (lower = more stable training)
    - Gradient norms
    
    Parameters
    ----------
    metrics_list : List[TrainingMetrics]
        List of metrics from different training runs
    save_path : Path, optional
        Base path for saving figures. Will create:
        - {save_path}_loss.pdf
        - {save_path}_path_length.pdf
        - {save_path}_grad_variance.pdf
        - {save_path}_grad_norm.pdf
    figsize : tuple
        Figure size for each plot
        
    Returns
    -------
    figs : List[matplotlib.Figure]
        List of 4 figures
    """
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_list)))
    figs = []
    
    # Get base path for saving
    base_path = None
    if save_path:
        base_path = str(save_path).replace('.pdf', '').replace('.png', '')
    
    # =========================================================================
    # Plot 1: Loss curves (convergence speed)
    # =========================================================================
    fig1, ax = plt.subplots(figsize=figsize)
    for metrics, color in zip(metrics_list, colors):
        raw_label = metrics.method_name or f"{'OT' if metrics.use_ot else 'Indep'}"
        label = _extract_short_method_name(raw_label)
        if metrics.train_losses:
            ax.semilogy(metrics.train_losses, label=label, color=color, linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Training Loss', fontsize=11)
    ax.set_title('Convergence Speed', fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    figs.append(fig1)
    
    if base_path:
        fig1.savefig(f"{base_path}_loss.pdf", dpi=150, bbox_inches='tight')
        fig1.savefig(f"{base_path}_loss.png", dpi=150, bbox_inches='tight')
        print(f"Saved loss curves to {base_path}_loss.pdf")
    
    # =========================================================================
    # Plot 2: Path lengths
    # =========================================================================
    fig2, ax = plt.subplots(figsize=figsize)
    for metrics, color in zip(metrics_list, colors):
        raw_label = metrics.method_name or f"{'OT' if metrics.use_ot else 'Indep'}"
        label = _extract_short_method_name(raw_label)
        if metrics.path_lengths:
            ax.plot(metrics.path_lengths, label=label, color=color, linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Mean Path Length', fontsize=11)
    ax.set_title('Interpolation Path Lengths (lower = better matching)', fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    figs.append(fig2)
    
    if base_path:
        fig2.savefig(f"{base_path}_path_length.pdf", dpi=150, bbox_inches='tight')
        fig2.savefig(f"{base_path}_path_length.png", dpi=150, bbox_inches='tight')
        print(f"Saved path lengths to {base_path}_path_length.pdf")
    
    # =========================================================================
    # Plot 3: Gradient variance
    # =========================================================================
    fig3, ax = plt.subplots(figsize=figsize)
    for metrics, color in zip(metrics_list, colors):
        raw_label = metrics.method_name or f"{'OT' if metrics.use_ot else 'Indep'}"
        label = _extract_short_method_name(raw_label)
        if metrics.gradient_variances:
            ax.semilogy(metrics.gradient_variances, label=label, color=color, linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Gradient Variance', fontsize=11)
    ax.set_title('Gradient Variance (lower = more stable)', fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    figs.append(fig3)
    
    if base_path:
        fig3.savefig(f"{base_path}_grad_variance.pdf", dpi=150, bbox_inches='tight')
        fig3.savefig(f"{base_path}_grad_variance.png", dpi=150, bbox_inches='tight')
        print(f"Saved gradient variance to {base_path}_grad_variance.pdf")
    
    # =========================================================================
    # Plot 4: Gradient norms
    # =========================================================================
    fig4, ax = plt.subplots(figsize=figsize)
    for metrics, color in zip(metrics_list, colors):
        raw_label = metrics.method_name or f"{'OT' if metrics.use_ot else 'Indep'}"
        label = _extract_short_method_name(raw_label)
        if metrics.gradient_norms:
            ax.plot(metrics.gradient_norms, label=label, color=color, linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Mean Gradient Norm', fontsize=11)
    ax.set_title('Gradient Norms', fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    figs.append(fig4)
    
    if base_path:
        fig4.savefig(f"{base_path}_grad_norm.pdf", dpi=150, bbox_inches='tight')
        fig4.savefig(f"{base_path}_grad_norm.png", dpi=150, bbox_inches='tight')
        print(f"Saved gradient norms to {base_path}_grad_norm.pdf")
    
    return figs


def plot_convergence_comparison(
    metrics_list: List[TrainingMetrics],
    save_path: Path = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Plot convergence comparison focused on loss curves.
    
    Parameters
    ----------
    metrics_list : List[TrainingMetrics]
        List of metrics from different training runs
    save_path : Path, optional
        Where to save the figure
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_list)))
    
    for metrics, color in zip(metrics_list, colors):
        label = metrics.method_name or f"{'OT' if metrics.use_ot else 'Indep'}"
        if metrics.train_losses:
            ax.semilogy(metrics.train_losses, label=label, color=color, linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Convergence Speed Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def compute_convergence_metrics(
    metrics: TrainingMetrics,
    target_loss: float = None,
    relative_improvement: float = 0.9,
) -> dict:
    """
    Compute summary convergence metrics.
    
    Parameters
    ----------
    metrics : TrainingMetrics
        Training metrics
    target_loss : float, optional
        Target loss to reach (for epochs-to-target computation)
    relative_improvement : float
        Fraction of improvement for convergence (default 0.9 = 90% of final improvement)
        
    Returns
    -------
    summary : dict
        Dictionary with convergence statistics
    """
    losses = np.array(metrics.train_losses)
    
    if len(losses) == 0:
        return {}
    
    summary = {
        'method': metrics.method_name,
        'use_ot': metrics.use_ot,
        'ot_kernel': metrics.ot_kernel,
        'final_loss': losses[-1],
        'initial_loss': losses[0],
        'total_improvement': losses[0] - losses[-1],
        'relative_improvement': (losses[0] - losses[-1]) / losses[0] if losses[0] > 0 else 0,
    }
    
    # Epochs to reach target loss
    if target_loss is not None:
        epochs_to_target = np.where(losses <= target_loss)[0]
        summary['epochs_to_target'] = epochs_to_target[0] + 1 if len(epochs_to_target) > 0 else None
        summary['target_loss'] = target_loss
    
    # Epochs to reach X% of final improvement
    if len(losses) > 1:
        total_drop = losses[0] - losses[-1]
        threshold = losses[0] - relative_improvement * total_drop
        epochs_to_threshold = np.where(losses <= threshold)[0]
        summary[f'epochs_to_{int(relative_improvement*100)}pct'] = (
            epochs_to_threshold[0] + 1 if len(epochs_to_threshold) > 0 else None
        )
    
    # Path length statistics
    if metrics.path_lengths:
        summary['mean_path_length'] = np.mean(metrics.path_lengths)
        summary['final_path_length'] = metrics.path_lengths[-1]
    
    # Gradient variance statistics
    if metrics.gradient_variances:
        summary['mean_grad_variance'] = np.mean(metrics.gradient_variances)
        summary['final_grad_variance'] = metrics.gradient_variances[-1]
    
    # Total training time
    if metrics.epoch_times:
        summary['total_time'] = sum(metrics.epoch_times)
        summary['mean_epoch_time'] = np.mean(metrics.epoch_times)
    
    return summary


def print_comparison_table(metrics_list: List[TrainingMetrics], target_loss: float = None):
    """
    Print a formatted comparison table of convergence metrics.
    
    Parameters
    ----------
    metrics_list : List[TrainingMetrics]
        List of metrics from different training runs
    target_loss : float, optional
        Target loss for epochs-to-target computation
    """
    summaries = [compute_convergence_metrics(m, target_loss) for m in metrics_list]
    
    print("\n" + "="*80)
    print("TRAINING COMPARISON SUMMARY")
    print("="*80)
    
    # Header
    print(f"{'Method':<25} {'Final Loss':<12} {'Path Len':<12} {'Grad Var':<12} {'Time (s)':<10}")
    print("-"*80)
    
    for s in summaries:
        method = s.get('method', 'Unknown')[:24]
        final_loss = f"{s.get('final_loss', 0):.6f}"
        path_len = f"{s.get('mean_path_length', 0):.4f}" if s.get('mean_path_length') else "N/A"
        grad_var = f"{s.get('mean_grad_variance', 0):.2e}" if s.get('mean_grad_variance') else "N/A"
        time = f"{s.get('total_time', 0):.1f}" if s.get('total_time') else "N/A"
        
        print(f"{method:<25} {final_loss:<12} {path_len:<12} {grad_var:<12} {time:<10}")
    
    print("="*80 + "\n")
    
    return summaries

