"""
Functional Flow Matching with Optimal Transport Pairing.

This module extends FFMModel to support minibatch OT coupling between
data functions and GP base samples, using kernel-based ground costs.

Key extension: Instead of independent sampling (x_data, z_noise), we use
OT pairing: (x_data, z_noise) ~ π* where π* minimizes transport cost
under a kernel-induced RKHS metric.

References:
- Functional Flow Matching: Kerrigan et al.
- OT-CFM: Tong et al. "Improving and Generalizing Flow-Based Generative Models 
  with minibatch optimal transport"
"""

import numpy as np
import torch
from torchdiffeq import odeint
from typing import Optional, Literal, Callable

from util.gaussian_process import GPPrior
from util.util import make_grid, reshape_for_batchwise, plot_loss_curve, plot_samples
from util.ot_monitoring import TrainingMonitor, TrainingMetrics
from optimal_transport import KernelOTPlanSampler, GaussianOTPlanSampler, create_kernel

import time


class FFMModelOT:
    """Functional Flow Matching with Optimal Transport pairing.
    
    Extends FFMModel to optionally use OT coupling between data functions
    and GP noise samples. The OT cost can be computed using:
    - Euclidean distance (standard)
    - RBF kernel-induced RKHS distance
    - Signature kernel-induced RKHS distance (for time series)
    - Closed-form Gaussian OT (Bures-Wasserstein) when Gaussian assumption holds
    
    The OT coupling is computed without gradients (detached) for stability.
    """
    
    def __init__(
        self,
        model,
        # GP prior params
        kernel_length: float = 0.001,
        kernel_variance: float = 1.0,
        # FFM path params
        sigma_min: float = 1e-4,
        vp: bool = False,
        # OT params
        use_ot: bool = True,
        ot_method: Literal["exact", "sinkhorn", "unbalanced", "partial", "gaussian"] = "sinkhorn",
        ot_reg: float = 0.05,
        ot_reg_m: float = 1.0,  # Marginal regularization for unbalanced OT
        ot_kernel: Optional[str] = "rbf",
        ot_kernel_params: Optional[dict] = None,
        ot_coupling: Literal["sample", "barycentric"] = "sample",
        # Device
        device: str = 'cpu',
        dtype: torch.dtype = torch.double,
    ):
        """
        Parameters
        ----------
        model : nn.Module
            Neural network for the vector field v_θ(t, x)
        kernel_length : float
            GP prior kernel lengthscale
        kernel_variance : float
            GP prior kernel variance
        sigma_min : float
            Minimum noise level in FFM probability path
        vp : bool
            Use variance-preserving path (trigonometric interpolant)
        use_ot : bool
            Whether to use OT pairing (if False, uses independent sampling)
        ot_method : str
            OT solver: 
            - "exact": exact OT solver (EMD)
            - "sinkhorn": entropic regularization (default)
            - "unbalanced": unbalanced OT (allows mass variation)
            - "partial": partial OT (allows partial matching)
            - "gaussian": closed-form Gaussian OT (Bures-Wasserstein)
              Fast & stable, but assumes data is well-approximated by Gaussian.
              Ignores ot_kernel parameter since it uses closed-form solution.
        ot_reg : float
            Sinkhorn regularization parameter ε (for method="sinkhorn")
            Or covariance regularization (for method="gaussian")
        ot_kernel : str or None
            Kernel for OT cost: "euclidean", "rbf", "signature", or None
            (ignored when ot_method="gaussian")
        ot_kernel_params : dict
            Parameters for the OT kernel (e.g., sigma for RBF, time_aug for signature)
        ot_coupling : str
            How to extract pairs from OT plan: "sample" or "barycentric"
            (ignored when ot_method="gaussian", which uses deterministic map)
        device : str
        dtype : torch.dtype
        """
        self.model = model
        self.device = device
        self.dtype = dtype
        
        # GP prior for base distribution
        self.gp = GPPrior(lengthscale=kernel_length, var=kernel_variance, device=device)
        self.kernel_length = kernel_length
        self.kernel_variance = kernel_variance
        
        # FFM path parameters
        self.sigma_min = sigma_min
        self.vp = vp
        if self.vp:
            self.alpha, self.dalpha = self.construct_alpha()
        
        # OT configuration
        self.use_ot = use_ot
        self.ot_method = ot_method
        self.ot_coupling = ot_coupling
        
        if use_ot:
            if ot_method == "gaussian":
                # Closed-form Gaussian OT (Bures-Wasserstein)
                # Much faster and more stable when Gaussian assumption holds
                self.ot_sampler = GaussianOTPlanSampler(
                    reg=ot_reg,
                    # We could pass GP prior covariance here, but it's batch-dependent
                    # so we estimate from samples instead
                    source_mean=None,
                    source_cov=None,
                    estimate_target_params=True,
                )
                self.ot_kernel_fn = None  # Not used for Gaussian OT
            else:
                # Kernel-based OT (Sinkhorn, exact, unbalanced, or partial)
                self.ot_sampler = KernelOTPlanSampler(
                    method=ot_method,
                    reg=ot_reg,
                    reg_m=ot_reg_m,  # For unbalanced OT
                )
                ot_kernel_params = ot_kernel_params or {}
                self.ot_kernel_fn = create_kernel(ot_kernel, **ot_kernel_params)
        else:
            self.ot_sampler = None
            self.ot_kernel_fn = None
    
    def construct_alpha(self):
        """Construct variance-preserving path interpolants."""
        def alpha(t):
            return torch.cos((t + 0.08)/2.16 * np.pi).to(self.device)
        def dalpha(t):
            return -np.pi/2.16 * torch.sin((t + 0.08)/2.16 * np.pi).to(self.device)
        return alpha, dalpha
    
    def sample_base(self, batch_size: int, n_channels: int, dims: tuple) -> torch.Tensor:
        """Sample from the GP base distribution.
        
        Parameters
        ----------
        batch_size : int
        n_channels : int
        dims : tuple
            Spatial dimensions, e.g., (64,) for 1D or (64, 64) for 2D
            
        Returns
        -------
        z : Tensor, shape (batch_size, n_channels, *dims)
        """
        query_points = make_grid(dims)
        z = self.gp.sample(query_points, dims, n_samples=batch_size, n_channels=n_channels)
        return z.to(self.device)
    
    def pair_samples(
        self, 
        x_data: torch.Tensor, 
        z_noise: torch.Tensor
    ) -> tuple:
        """Pair data samples with noise samples using OT or independent sampling.
        
        Parameters
        ----------
        x_data : Tensor, shape (B, C, *dims)
            Data function batch
        z_noise : Tensor, shape (B, C, *dims)
            GP noise batch
            
        Returns
        -------
        x_paired : Tensor
            Paired data samples
        z_paired : Tensor
            Paired noise samples
        """
        if not self.use_ot:
            # Independent sampling (original FFM behavior)
            return x_data, z_noise
        
        # Gaussian OT (closed-form Bures-Wasserstein)
        if self.ot_method == "gaussian":
            # Gaussian OT applies deterministic map T(z) to transport z -> x
            # Returns (x_data, T(z_noise)) where T is the optimal Gaussian map
            x_paired, z_paired = self.ot_sampler.sample_plan(x_data, z_noise)
            return x_paired, z_paired
        
        # Kernel-based OT (Sinkhorn or exact)
        if self.ot_coupling == "sample":
            x_paired, z_paired = self.ot_sampler.sample_plan(
                x_data, z_noise, 
                kernel_fn=self.ot_kernel_fn,
                replace=True
            )
        elif self.ot_coupling == "barycentric":
            x_paired, z_paired = self.ot_sampler.barycentric_map(
                x_data, z_noise,
                kernel_fn=self.ot_kernel_fn
            )
        else:
            raise ValueError(f"Unknown coupling method: {self.ot_coupling}")
        
        return x_paired, z_paired
    
    def simulate(self, t: torch.Tensor, x_data: torch.Tensor, z_noise: torch.Tensor) -> torch.Tensor:
        """Sample from the probability path p_t(x | x_data, z_noise).
        
        Unlike the original FFMModel.simulate which samples z internally,
        this version takes both endpoints as input (after OT pairing).
        
        Parameters
        ----------
        t : Tensor, shape (batch_size,)
            Time values in [0, 1]
        x_data : Tensor, shape (batch_size, n_channels, *dims)
            Target data functions (endpoint at t=1)
        z_noise : Tensor, shape (batch_size, n_channels, *dims)
            Base GP samples (endpoint at t=0)
            
        Returns
        -------
        x_t : Tensor, shape (batch_size, n_channels, *dims)
            Samples from the probability path at time t
        """
        n_dims = len(x_data.shape) - 2
        t = reshape_for_batchwise(t, 1 + n_dims)
        
        if self.vp:
            # Variance-preserving (trigonometric) interpolant
            mu = self.alpha(1-t) * x_data
            sigma = torch.sqrt(1 - self.alpha(1-t)**2)
            x_t = mu + sigma * z_noise
        else:
            # OT (linear) interpolant: x_t = t*x_data + (1-t)*sigma*z_noise  
            # Actually the original uses: mu = t * x_data, sigma = 1 - (1-sigma_min)*t
            mu = t * x_data
            sigma = 1. - (1. - self.sigma_min) * t
            x_t = mu + sigma * z_noise
        
        return x_t
    
    def get_conditional_fields(
        self, 
        t: torch.Tensor, 
        x_data: torch.Tensor, 
        x_noisy: torch.Tensor,
        z_noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute the conditional vector field u_t(x_noisy | x_data, z_noise).
        
        Parameters
        ----------
        t : Tensor, shape (batch_size,)
        x_data : Tensor, shape (batch_size, n_channels, *dims)
            Target data
        x_noisy : Tensor, shape (batch_size, n_channels, *dims)
            Current sample on the path
        z_noise : Tensor, optional
            Base noise (needed for some formulations)
            
        Returns
        -------
        u_t : Tensor, shape (batch_size, n_channels, *dims)
            Conditional velocity field
        """
        n_dims = len(x_data.shape) - 2
        t = reshape_for_batchwise(t, 1 + n_dims)
        
        if self.vp:
            conditional_fields = (
                self.dalpha(1-t) / (1 - self.alpha(1-t)**2)
            ) * (self.alpha(1-t) * x_noisy - x_data)
        else:
            c = 1. - (1. - self.sigma_min) * t
            conditional_fields = (x_data - (1. - self.sigma_min) * x_noisy) / c
        
        return conditional_fields
    
    def train(
        self,
        train_loader,
        optimizer,
        epochs: int,
        scheduler=None,
        test_loader=None,
        eval_int: int = 0,
        save_int: int = 0,
        generate: bool = False,
        save_path=None,
        monitor: Optional[TrainingMonitor] = None,
    ):
        """Train the FFM model with OT pairing.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader yielding batches of shape (B, C, *dims)
        optimizer : torch.optim.Optimizer
        epochs : int
        scheduler : lr_scheduler, optional
        test_loader : DataLoader, optional
        eval_int : int
            Evaluate every eval_int epochs (0 = no eval)
        save_int : int
            Save checkpoint every save_int epochs
        generate : bool
            Generate samples during evaluation
        save_path : Path
            Where to save checkpoints and plots
        monitor : TrainingMonitor, optional
            Monitor for tracking convergence, path lengths, and gradient variance
        """
        tr_losses = []
        te_losses = []
        eval_eps = []
        evaluate = (eval_int > 0) and (test_loader is not None)
        
        model = self.model
        device = self.device
        dtype = self.dtype
        
        first = True
        for ep in range(1, epochs + 1):
            ##### TRAINING LOOP
            t0 = time.time()
            model.train()
            tr_loss = 0.0
            
            for batch in train_loader:
                batch = batch.to(device).to(dtype)
                batch_size = batch.shape[0]
                
                if first:
                    self.n_channels = batch.shape[1]
                    self.train_dims = batch.shape[2:]
                    first = False
                
                # Sample base GP noise
                z_noise = self.sample_base(batch_size, self.n_channels, self.train_dims)
                z_noise = z_noise.to(dtype)
                
                # OT pairing (or independent if use_ot=False)
                x_data, z_paired = self.pair_samples(batch, z_noise)
                
                # Sample time t ~ Unif[0, 1)
                t = torch.rand(batch_size, device=device)
                
                # Sample from probability path p_t(x | x_data, z_paired)
                x_noisy = self.simulate(t, x_data, z_paired)
                
                # Get conditional vector field target
                target = self.get_conditional_fields(t, x_data, x_noisy, z_paired)
                
                x_noisy = x_noisy.to(device)
                target = target.to(device)
                
                # Forward pass and loss
                model_out = model(t, x_noisy)
                
                optimizer.zero_grad()
                loss = torch.mean((model_out - target) ** 2)
                loss.backward()
                
                # Log batch metrics if monitor is provided
                if monitor is not None:
                    monitor.log_batch(
                        loss=loss.item(),
                        x_data=x_data.detach(),
                        z_noise=z_paired.detach(),
                        model=model,
                        sigma_min=self.sigma_min,
                    )
                
                optimizer.step()
                
                tr_loss += loss.item()
            
            tr_loss /= len(train_loader)
            tr_losses.append(tr_loss)
            if scheduler:
                scheduler.step()
            
            t1 = time.time()
            epoch_time = t1 - t0
            ot_tag = "OT" if self.use_ot else "indep"
            print(f'tr @ epoch {ep}/{epochs} [{ot_tag}] | Loss {tr_loss:.6f} | {epoch_time:.2f} (s)')
            
            ##### EVAL LOOP
            te_loss_epoch = None
            if eval_int > 0 and (ep % eval_int == 0):
                t0_eval = time.time()
                eval_eps.append(ep)
                
                with torch.no_grad():
                    model.eval()
                    
                    if evaluate:
                        te_loss = 0.0
                        for batch in test_loader:
                            batch = batch.to(device).to(dtype)
                            batch_size = batch.shape[0]
                            
                            z_noise = self.sample_base(batch_size, self.n_channels, self.train_dims)
                            z_noise = z_noise.to(dtype)
                            
                            x_data, z_paired = self.pair_samples(batch, z_noise)
                            
                            t = torch.rand(batch_size, device=device)
                            x_noisy = self.simulate(t, x_data, z_paired)
                            target = self.get_conditional_fields(t, x_data, x_noisy, z_paired)
                            
                            x_noisy = x_noisy.to(device)
                            target = target.to(device)
                            model_out = model(t, x_noisy)
                            
                            loss = torch.mean((model_out - target) ** 2)
                            te_loss += loss.item()
                        
                        te_loss /= len(test_loader)
                        te_losses.append(te_loss)
                        te_loss_epoch = te_loss
                        
                        t1_eval = time.time()
                        print(f'te @ epoch {ep}/{epochs} | Loss {te_loss:.6f} | {t1_eval-t0_eval:.2f} (s)')
                    
                    if generate:
                        samples = self.sample(self.train_dims, n_channels=self.n_channels, n_samples=16)
                        plot_samples(samples, save_path / f'samples_epoch{ep}.pdf')
            
            # End epoch in monitor
            if monitor is not None:
                monitor.end_epoch(epoch_time=epoch_time, test_loss=te_loss_epoch)
            
            ##### BOOKKEEPING
            if save_int > 0 and (ep % save_int == 0) and save_path is not None:
                torch.save(model.state_dict(), save_path / f'epoch_{ep}.pt')
            
            if save_path is not None:
                if evaluate:
                    plot_loss_curve(tr_losses, save_path / 'loss.pdf', te_loss=te_losses, te_epochs=eval_eps)
                else:
                    plot_loss_curve(tr_losses, save_path / 'loss.pdf')
        
        # Return metrics if monitor provided
        if monitor is not None:
            return monitor.get_metrics()
        return None
    
    @torch.no_grad()
    def sample(
        self,
        dims: tuple,
        n_channels: int = 1,
        n_samples: int = 1,
        n_eval: int = 2,
        return_path: bool = False,
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ) -> torch.Tensor:
        """Sample from the learned flow.
        
        Parameters
        ----------
        dims : tuple
            Spatial dimensions, e.g., (64,) for 1D
        n_channels : int
        n_samples : int
        n_eval : int
            Number of ODE evaluation points in [0, 1]
        return_path : bool
            Return full ODE trajectory
        rtol, atol : float
            ODE solver tolerances
            
        Returns
        -------
        samples : Tensor
            Generated samples at t=1, or full path if return_path=True
        """
        t = torch.linspace(0, 1, n_eval, device=self.device)
        grid = make_grid(dims)
        x0 = self.gp.sample(grid, dims, n_samples=n_samples, n_channels=n_channels)
        x0 = x0.to(self.device)
        
        method = 'dopri5'
        out = odeint(self.model, x0, t, method=method, rtol=rtol, atol=atol)
        
        if return_path:
            return out
        else:
            return out[-1]


# =============================================================================
# Convenience function for backward compatibility
# =============================================================================

def FFMModel(
    model,
    kernel_length: float = 0.001,
    kernel_variance: float = 1.0,
    sigma_min: float = 1e-4,
    device: str = 'cpu',
    dtype: torch.dtype = torch.double,
    vp: bool = False,
    # New OT parameters (defaults to no OT for backward compat)
    use_ot: bool = False,
    **ot_kwargs,
):
    """Factory function for FFMModel with optional OT support.
    
    For backward compatibility, use_ot defaults to False.
    """
    return FFMModelOT(
        model=model,
        kernel_length=kernel_length,
        kernel_variance=kernel_variance,
        sigma_min=sigma_min,
        vp=vp,
        use_ot=use_ot,
        device=device,
        dtype=dtype,
        **ot_kwargs,
    )

