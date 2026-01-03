"""
Optimal Transport utilities for Functional Flow Matching.

Implements OT plan sampling with kernel-based cost matrices (RBF, signature kernels, etc.)
for pairing data functions with base GP samples.

Based on the CFM OT implementation but extended for:
- Kernel-based ground costs (RKHS distances)
- Signature kernels for time series (via pySigLib)
- Functional data representations
- Closed-form Gaussian OT (Bures-Wasserstein) for Gaussian measures
"""

import warnings
from functools import partial
from typing import Optional, Union, Callable, Literal, Tuple

import numpy as np
import ot as pot
import ot.gaussian
import torch


class KernelOTPlanSampler:
    """OT Plan Sampler with kernel-based cost matrices.
    
    Unlike standard OT which uses squared Euclidean distance, this sampler
    allows using kernel-induced RKHS distances as the ground cost:
    
        c(x, z) = ||φ(x) - φ(z)||²_H = k(x,x) + k(z,z) - 2k(x,z)
    
    where k is a kernel function and φ is the induced feature map.
    """
    
    def __init__(
        self,
        method: Literal["exact", "sinkhorn", "unbalanced", "partial"] = "sinkhorn",
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost: bool = True,  # Changed default to True for stability
        num_threads: Union[int, str] = 1,
        warn: bool = True,
        sinkhorn_max_iter: int = 1500,
    ) -> None:
        """Initialize the KernelOTPlanSampler.
        
        Parameters
        ----------
        method : str
            OT solver method: "exact", "sinkhorn", "unbalanced", "partial"
        reg : float
            Entropic regularization for Sinkhorn-based solvers.
            Larger values = more regularization = more stable but less sharp OT plan.
            Recommended: 0.1-1.0 for functional data.
        reg_m : float
            Marginal regularization for unbalanced OT
        normalize_cost : bool
            Whether to normalize cost matrix by its median. 
            STRONGLY RECOMMENDED for functional data where costs can be large.
        num_threads : int or str
            Number of threads for exact OT solver
        warn : bool
            Whether to warn on numerical issues
        sinkhorn_max_iter : int
            Maximum Sinkhorn iterations
        """
        if method == "exact":
            self.ot_fn = partial(pot.emd, numThreads=num_threads)
        elif method == "sinkhorn":
            # Use log-stabilized Sinkhorn for better numerical stability
            # This computes in log-domain, avoiding underflow/overflow issues
            # that occur with the standard Sinkhorn when reg is small or costs are large
            self.ot_fn = partial(
                pot.sinkhorn, 
                reg=reg, 
                numItermax=sinkhorn_max_iter, 
                stopThr=1e-9, 
                warn=warn,
                method='sinkhorn_log',  # Log-domain computation for stability
            )
        elif method == "unbalanced":
            self.ot_fn = partial(pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m)
        elif method == "partial":
            self.ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.method = method
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.warn = warn
        self.sinkhorn_max_iter = sinkhorn_max_iter
    
    def compute_euclidean_cost(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Compute squared Euclidean cost matrix.
        
        Parameters
        ----------
        x0 : Tensor, shape (B, *)
            Source batch
        x1 : Tensor, shape (B, *)
            Target batch
            
        Returns
        -------
        C : Tensor, shape (B, B)
            Cost matrix C[i,j] = ||x0[i] - x1[j]||²
        """
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        return torch.cdist(x0, x1) ** 2
    
    def compute_kernel_cost(
        self, 
        x0: torch.Tensor, 
        x1: torch.Tensor,
        kernel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Compute RKHS distance cost matrix from a kernel function.
        
        The RKHS distance induced by kernel k is:
            c(x, z) = ||φ(x) - φ(z)||²_H = k(x,x) + k(z,z) - 2k(x,z)
        
        Parameters
        ----------
        x0 : Tensor, shape (B, C, *dims)
            Source batch (data functions)
        x1 : Tensor, shape (B, C, *dims)
            Target batch (GP samples)
        kernel_fn : Callable
            Kernel function k(X, Y) -> Gram matrix of shape (B_x, B_y)
            
        Returns
        -------
        C : Tensor, shape (B, B)
            Cost matrix where C[i,j] = RKHS distance between x0[i] and x1[j]
        """
        # Compute Gram matrices
        K_00 = kernel_fn(x0, x0)  # (B, B)
        K_11 = kernel_fn(x1, x1)  # (B, B)
        K_01 = kernel_fn(x0, x1)  # (B, B)
        
        # RKHS distance: c(x,z) = k(x,x) + k(z,z) - 2k(x,z)
        diag_K00 = torch.diag(K_00).unsqueeze(1)  # (B, 1)
        diag_K11 = torch.diag(K_11).unsqueeze(0)  # (1, B)
        
        C = diag_K00 + diag_K11 - 2 * K_01  # (B, B)
        
        # Ensure non-negative (numerical precision)
        C = torch.clamp(C, min=0.0)
        
        return C
    
    def get_map(
        self, 
        x0: torch.Tensor, 
        x1: torch.Tensor,
        kernel_fn: Optional[Callable] = None,
    ) -> np.ndarray:
        """Compute the OT plan between source and target minibatch.
        
        Parameters
        ----------
        x0 : Tensor, shape (B, *)
            Source minibatch
        x1 : Tensor, shape (B, *)
            Target minibatch
        kernel_fn : Callable, optional
            Kernel function for RKHS cost. If None, uses Euclidean cost.
            
        Returns
        -------
        pi : ndarray, shape (B, B)
            OT plan (coupling)
        """
        # Compute cost matrix
        if kernel_fn is not None:
            M = self.compute_kernel_cost(x0, x1, kernel_fn)
        else:
            M = self.compute_euclidean_cost(x0, x1)
        
        # Normalize cost matrix for numerical stability
        # This is CRITICAL for functional data where costs can be very large
        if self.normalize_cost:
            # Use median for robustness (less sensitive to outliers than max)
            cost_scale = M.median()
            if cost_scale > 1e-8:
                M = M / cost_scale
            else:
                # If median is tiny, use max instead
                cost_scale = M.max()
                if cost_scale > 1e-8:
                    M = M / cost_scale
        
        # Uniform marginals
        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        
        # Convert to numpy and solve OT (detach for gradient stability)
        M_np = M.detach().cpu().numpy()
        
        # Check for any issues with cost matrix
        if not np.all(np.isfinite(M_np)):
            if self.warn:
                warnings.warn("Cost matrix contains non-finite values, using uniform plan")
            return np.ones((x0.shape[0], x1.shape[0])) / (x0.shape[0] * x1.shape[0])
        
        try:
            pi = self.ot_fn(a, b, M_np)
        except Exception as e:
            if self.warn:
                warnings.warn(f"OT solver failed ({e}), using uniform plan")
            return np.ones((x0.shape[0], x1.shape[0])) / (x0.shape[0] * x1.shape[0])
        
        # Handle numerical issues
        if not np.all(np.isfinite(pi)):
            if self.warn:
                warnings.warn("OT plan contains non-finite values, reverting to uniform")
            pi = np.ones_like(pi) / pi.size
        
        if np.abs(pi.sum()) < 1e-8:
            if self.warn:
                warnings.warn("OT plan sum is near zero, reverting to uniform")
            pi = np.ones_like(pi) / pi.size
        
        return pi
    
    def sample_map(
        self, 
        pi: np.ndarray, 
        batch_size: int, 
        replace: bool = True
    ) -> tuple:
        """Sample indices from the OT plan.
        
        Parameters
        ----------
        pi : ndarray, shape (B, B)
            OT plan
        batch_size : int
            Number of samples to draw
        replace : bool
            Whether to sample with replacement
            
        Returns
        -------
        (i, j) : tuple of ndarrays
            Indices into source and target batches
        """
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(
            pi.shape[0] * pi.shape[1], 
            p=p, 
            size=batch_size, 
            replace=replace
        )
        return np.divmod(choices, pi.shape[1])
    
    def sample_plan(
        self, 
        x0: torch.Tensor, 
        x1: torch.Tensor,
        kernel_fn: Optional[Callable] = None,
        replace: bool = True,
    ) -> tuple:
        """Compute OT plan and sample paired endpoints.
        
        Parameters
        ----------
        x0 : Tensor, shape (B, *)
            Source minibatch (data functions)
        x1 : Tensor, shape (B, *)
            Target minibatch (GP samples)
        kernel_fn : Callable, optional
            Kernel function for RKHS cost
        replace : bool
            Sample with replacement
            
        Returns
        -------
        x0_paired : Tensor
            Paired source samples
        x1_paired : Tensor
            Paired target samples
        """
        pi = self.get_map(x0, x1, kernel_fn)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        return x0[i], x1[j]
    
    def barycentric_map(
        self, 
        x0: torch.Tensor, 
        x1: torch.Tensor,
        kernel_fn: Optional[Callable] = None,
    ) -> tuple:
        """Compute barycentric mapping (soft assignment).
        
        Instead of sampling, compute weighted averages:
            x1_bary[i] = Σ_j π[i,j] * x1[j]
        
        This can reduce variance when convex combinations are meaningful.
        
        Parameters
        ----------
        x0 : Tensor, shape (B, *)
            Source minibatch
        x1 : Tensor, shape (B, *)
            Target minibatch
        kernel_fn : Callable, optional
            Kernel function for RKHS cost
            
        Returns
        -------
        x0 : Tensor
            Original source samples
        x1_bary : Tensor
            Barycentric target samples
        """
        pi = self.get_map(x0, x1, kernel_fn)
        pi = torch.from_numpy(pi).to(x1.device).float()
        
        # Normalize rows for weighted average
        pi_normalized = pi / (pi.sum(dim=1, keepdim=True) + 1e-8)
        
        # Reshape for batch matmul: (B, B) @ (B, *dims) -> (B, *dims)
        original_shape = x1.shape
        x1_flat = x1.reshape(x1.shape[0], -1)  # (B, D)
        
        x1_bary = torch.mm(pi_normalized, x1_flat)  # (B, D)
        x1_bary = x1_bary.reshape(original_shape)
        
        return x0, x1_bary


# =============================================================================
# Gaussian OT (Bures-Wasserstein) - Closed Form Solution
# =============================================================================

class GaussianOTPlanSampler:
    """Closed-form OT for Gaussian distributions (Bures-Wasserstein).
    
    When both source and target distributions are Gaussian:
        X ~ N(μ₀, Σ₀)  (e.g., GP prior)
        Y ~ N(μ₁, Σ₁)  (e.g., data, approximated as Gaussian)
    
    The optimal transport map has closed form (McCann/Brenier):
        T(x) = μ₁ + A(x - μ₀)
    
    where A = Σ₀^{-1/2} (Σ₀^{1/2} Σ₁ Σ₀^{1/2})^{1/2} Σ₀^{-1/2}
    
    This is MUCH faster than numerical OT solvers and perfectly accurate
    when the Gaussian assumption holds. Uses POT's ot.gaussian module.
    
    References:
        - McCann (1997): A convexity principle for interacting gases
        - Peyré & Cuturi (2019): Computational Optimal Transport
        - POT documentation: https://pythonot.github.io/master/gen_modules/ot.gaussian.html
    """
    
    def __init__(
        self,
        reg: float = 1e-6,
        source_mean: Optional[torch.Tensor] = None,
        source_cov: Optional[torch.Tensor] = None,
        estimate_target_params: bool = True,
        warn: bool = True,
    ) -> None:
        """Initialize GaussianOTPlanSampler.
        
        Parameters
        ----------
        reg : float
            Regularization added to covariance diagonals for numerical stability.
            Important for high-dimensional functional data.
        source_mean : Tensor, optional
            Known mean of source distribution (e.g., GP prior mean).
            If None, estimated from samples.
        source_cov : Tensor, optional  
            Known covariance of source distribution (e.g., GP prior covariance).
            If None, estimated from samples.
        estimate_target_params : bool
            If True, estimate target mean/cov from samples each batch.
            If False, must provide target_mean/target_cov to methods.
        warn : bool
            Whether to warn on numerical issues.
        """
        self.reg = reg
        self.source_mean = source_mean
        self.source_cov = source_cov
        self.estimate_target_params = estimate_target_params
        self.warn = warn
        
        # Cache for computed mapping
        self._cached_A = None
        self._cached_b = None
    
    def _estimate_gaussian_params(
        self, 
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate mean and covariance from samples.
        
        Parameters
        ----------
        x : Tensor, shape (B, *dims)
            Samples
        weights : Tensor, optional, shape (B,)
            Sample weights
            
        Returns
        -------
        mean : Tensor, shape (D,)
            Estimated mean (flattened)
        cov : Tensor, shape (D, D)
            Estimated covariance
        """
        # Flatten to (B, D)
        x_flat = x.reshape(x.shape[0], -1)
        B, D = x_flat.shape
        
        if weights is None:
            weights = torch.ones(B, device=x.device, dtype=x.dtype) / B
        else:
            weights = weights / weights.sum()
        
        # Weighted mean
        mean = (weights.unsqueeze(1) * x_flat).sum(dim=0)
        
        # Weighted covariance
        x_centered = x_flat - mean.unsqueeze(0)
        cov = (x_centered.T * weights.unsqueeze(0)) @ x_centered
        
        # Add regularization for stability
        cov = cov + self.reg * torch.eye(D, device=x.device, dtype=x.dtype)
        
        return mean, cov
    
    def compute_mapping(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        source_mean: Optional[torch.Tensor] = None,
        source_cov: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the optimal linear mapping T(x) = Ax + b.
        
        Uses POT's bures_wasserstein_mapping for the closed-form solution.
        
        Parameters
        ----------
        x0 : Tensor, shape (B, *dims)
            Source samples (e.g., GP samples)
        x1 : Tensor, shape (B, *dims)
            Target samples (e.g., data)
        source_mean : Tensor, optional
            Known source mean. Overrides self.source_mean.
        source_cov : Tensor, optional
            Known source covariance. Overrides self.source_cov.
            
        Returns
        -------
        A : Tensor, shape (D, D)
            Linear map matrix
        b : Tensor, shape (D,)
            Bias vector
        """
        device = x0.device
        dtype = x0.dtype
        
        # Flatten samples
        x0_flat = x0.reshape(x0.shape[0], -1)
        x1_flat = x1.reshape(x1.shape[0], -1)
        
        # Get source parameters (use provided, cached, or estimate)
        if source_mean is not None and source_cov is not None:
            mu_s = source_mean.reshape(-1)
            Sigma_s = source_cov
        elif self.source_mean is not None and self.source_cov is not None:
            mu_s = self.source_mean.reshape(-1).to(device=device, dtype=dtype)
            Sigma_s = self.source_cov.to(device=device, dtype=dtype)
        else:
            mu_s, Sigma_s = self._estimate_gaussian_params(x0)
        
        # Estimate target parameters
        mu_t, Sigma_t = self._estimate_gaussian_params(x1)
        
        # Convert to numpy for POT
        mu_s_np = mu_s.detach().cpu().numpy()
        mu_t_np = mu_t.detach().cpu().numpy()
        Sigma_s_np = Sigma_s.detach().cpu().numpy()
        Sigma_t_np = Sigma_t.detach().cpu().numpy()
        
        D = x0_flat.shape[1]
        use_identity = False
        
        try:
            # Use POT's closed-form Gaussian OT mapping
            A_np, b_np = pot.gaussian.bures_wasserstein_mapping(
                mu_s_np, mu_t_np, Sigma_s_np, Sigma_t_np, log=False
            )
            
            # Check for NaN/Inf in results (can happen with ill-conditioned covariances)
            if np.any(np.isnan(A_np)) or np.any(np.isinf(A_np)):
                if self.warn:
                    warnings.warn(
                        f"Gaussian OT produced NaN/Inf (dim={D}, reg={self.reg}). "
                        f"Consider increasing regularization. Using identity."
                    )
                use_identity = True
            if np.any(np.isnan(b_np)) or np.any(np.isinf(b_np)):
                if self.warn:
                    warnings.warn(
                        f"Gaussian OT bias has NaN/Inf (dim={D}). Using mean shift only."
                    )
                b_np = (mu_t_np - mu_s_np)
                
        except Exception as e:
            if self.warn:
                warnings.warn(f"Gaussian OT mapping failed ({e}), using identity")
            use_identity = True
        
        if use_identity:
            A_np = np.eye(D)
            b_np = (mu_t_np - mu_s_np)
        
        # Convert back to torch
        A = torch.from_numpy(A_np).to(device=device, dtype=dtype)
        b = torch.from_numpy(b_np.flatten()).to(device=device, dtype=dtype)
        
        return A, b
    
    def transport(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        source_mean: Optional[torch.Tensor] = None,
        source_cov: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the optimal transport map to source samples.
        
        Parameters
        ----------
        x0 : Tensor, shape (B, *dims)
            Source samples to transport
        x1 : Tensor, shape (B, *dims)
            Target samples (for estimating target distribution)
        source_mean, source_cov : Tensor, optional
            Known source parameters
            
        Returns
        -------
        x0_transported : Tensor, shape (B, *dims)
            Transported source samples T(x0)
        """
        original_shape = x0.shape
        x0_flat = x0.reshape(x0.shape[0], -1)
        
        A, b = self.compute_mapping(x0, x1, source_mean, source_cov)
        
        # Apply affine map: T(x) = Ax + b
        x0_transported = x0_flat @ A.T + b.unsqueeze(0)
        
        return x0_transported.reshape(original_shape)
    
    def sample_plan(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        source_mean: Optional[torch.Tensor] = None,
        source_cov: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Gaussian OT to pair source and target samples.
        
        Unlike discrete OT which samples from a coupling, Gaussian OT
        applies a deterministic map. Returns (x0, T(x0)) where T is
        the optimal transport map.
        
        Parameters
        ----------
        x0 : Tensor, shape (B, *dims)
            Source samples (e.g., GP samples)
        x1 : Tensor, shape (B, *dims)
            Target samples (used to estimate target distribution)
        source_mean, source_cov : Tensor, optional
            Known source parameters
            
        Returns
        -------
        x0 : Tensor
            Original source samples
        x0_transported : Tensor  
            Transported source samples (paired targets)
        """
        x0_transported = self.transport(x0, x1, source_mean, source_cov)
        return x0, x0_transported
    
    def empirical_sample_plan(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience method using POT's empirical_bures_wasserstein_mapping.
        
        Estimates both source and target parameters from samples.
        
        Parameters
        ----------
        x0 : Tensor, shape (B, *dims)
            Source samples
        x1 : Tensor, shape (B, *dims)
            Target samples
            
        Returns
        -------
        x0 : Tensor
            Original source samples
        x0_transported : Tensor
            Transported source samples
        """
        original_shape = x0.shape
        x0_flat = x0.reshape(x0.shape[0], -1).detach().cpu().numpy()
        x1_flat = x1.reshape(x1.shape[0], -1).detach().cpu().numpy()
        
        try:
            A_np, b_np = pot.gaussian.empirical_bures_wasserstein_mapping(
                x0_flat, x1_flat, reg=self.reg, bias=True, log=False
            )
        except Exception as e:
            if self.warn:
                warnings.warn(f"Empirical Gaussian OT failed ({e}), using identity")
            D = x0_flat.shape[1]
            A_np = np.eye(D)
            b_np = x1_flat.mean(axis=0) - x0_flat.mean(axis=0)
        
        # Apply transport
        x0_transported_np = x0_flat @ A_np.T + b_np.flatten()
        
        x0_transported = torch.from_numpy(x0_transported_np).to(
            device=x0.device, dtype=x0.dtype
        ).reshape(original_shape)
        
        return x0, x0_transported


# =============================================================================
# Kernel Functions for OT Cost
# =============================================================================

class RBFKernel:
    """RBF (Gaussian) kernel for functional data.
    
    k(x, y) = exp(-||x - y||² / (2σ²))
    
    For functional data, operates on flattened representations.
    """
    
    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF Gram matrix.
        
        Parameters
        ----------
        x : Tensor, shape (B1, *)
        y : Tensor, shape (B2, *)
        
        Returns
        -------
        K : Tensor, shape (B1, B2)
        """
        # Flatten to (B, D)
        x_flat = x.reshape(x.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        
        # Pairwise squared distances
        dist_sq = torch.cdist(x_flat, y_flat) ** 2
        
        return torch.exp(-dist_sq / (2 * self.sigma ** 2))


class SignatureKernel:
    """Signature kernel for time series / path-valued data.
    
    Uses pySigLib for efficient signature kernel computation.
    Designed for paths/time series of shape (B, L, d) where:
        - B = batch size
        - L = sequence length  
        - d = channel dimension
    
    The signature kernel captures sequential structure:
    - Order of events
    - Area enclosed by paths
    - Higher-order path interactions
    
    Note: Requires pySigLib to be installed.
    """
    
    def __init__(
        self,
        time_aug: bool = True,
        lead_lag: bool = False,
        dyadic_order: int = 2,
        static_kernel_type: str = "rbf",
        static_kernel_sigma: float = 1.0,
        add_basepoint: bool = True,
        normalize: bool = False,
        data_scaling: Optional[float] = None,
        max_seq_len: int = 128,
        max_batch: int = 64,
    ):
        """
        Parameters
        ----------
        time_aug : bool
            Add time as an extra dimension (t, x_t). 
            Note: We add time manually for backward compatibility.
        lead_lag : bool
            Apply lead-lag transform (captures quadratic variation)
        dyadic_order : int
            Dyadic refinement level (default 2 for good accuracy)
        static_kernel_type : str
            Base kernel on path increments: "linear" or "rbf"
            RBF is more stable for optimization.
        static_kernel_sigma : float
            Bandwidth for RBF static kernel
        add_basepoint : bool
            Prepend zero row to paths (recommended)
        normalize : bool
            Normalize paths (mean=0, std=1)
        data_scaling : float, optional
            Scale factor for data channels. Scaling by s causes 
            level-k to scale as s^(2k).
        max_seq_len : int
            Maximum sequence length. Longer paths are subsampled to this length.
            This helps avoid GPU memory issues with long sequences.
        max_batch : int
            Maximum batch size for pysiglib computation. Controls GPU memory usage.
        """
        self.time_aug = time_aug
        self.lead_lag = lead_lag
        self.dyadic_order = dyadic_order
        self.static_kernel_type = static_kernel_type
        self.static_kernel_sigma = static_kernel_sigma
        self.add_basepoint = add_basepoint
        self.normalize = normalize
        self.data_scaling = data_scaling
        self.max_seq_len = max_seq_len
        self.max_batch = max_batch
        
        # Try to import pySigLib
        self._pysiglib_available = False
        self._sig_kernel_gram = None
        self._static_kernel = None
        self._fallback_kernel = None
        
        try:
            from pysiglib.torch_api import sig_kernel_gram
            from pysiglib.static_kernels import LinearKernel, RBFKernel as SigRBFKernel
            
            self._sig_kernel_gram = sig_kernel_gram
            
            # Create static kernel object
            if static_kernel_type == "linear":
                self._static_kernel = LinearKernel()
            else:  # "rbf"
                self._static_kernel = SigRBFKernel(static_kernel_sigma)
            
            self._pysiglib_available = True
            # Create fallback RBF kernel for error recovery
            self._fallback_kernel = RBFKernel(sigma=1.0)
        except ImportError:
            warnings.warn(
                "pySigLib not found. SignatureKernel will fall back to RBF. "
                "Install with: pip install pysiglib"
            )
            self._fallback_kernel = RBFKernel(sigma=1.0)
    
    def _prepare_paths(self, x: torch.Tensor) -> torch.Tensor:
        """Convert functional data to path format (B, L, d).
        
        Handles various input shapes:
        - (B, C, L): 1D functions with C channels -> (B, L, C)
        - (B, L): single-channel 1D functions -> (B, L, 1)
        
        Also subsamples paths if they exceed max_seq_len.
        """
        if x.dim() == 2:
            # (B, L) -> (B, L, 1)
            paths = x.unsqueeze(-1)
        elif x.dim() == 3:
            # (B, C, L) -> (B, L, C)
            paths = x.permute(0, 2, 1)
        else:
            raise ValueError(f"Unsupported shape {x.shape} for signature kernel")
        
        # Subsample if sequence is too long
        B, L, D = paths.shape
        if L > self.max_seq_len:
            # Uniform subsampling to max_seq_len points
            indices = torch.linspace(0, L - 1, self.max_seq_len, dtype=torch.long, device=paths.device)
            paths = paths[:, indices, :]
        
        return paths
    
    def _preprocess_paths(self, paths: torch.Tensor) -> torch.Tensor:
        """Preprocess paths: normalize, scale, add basepoint, add time.
        
        Parameters
        ----------
        paths : Tensor, shape (B, L, D)
        
        Returns
        -------
        processed : Tensor, shape (B, L', D') where L' and D' may differ
        """
        B, L, D = paths.shape
        
        # Normalize if requested
        if self.normalize:
            mean = paths.mean(dim=1, keepdim=True)
            std = paths.std(dim=1, keepdim=True) + 1e-8
            paths = (paths - mean) / std
        
        # Scale data channels
        if self.data_scaling is not None:
            paths = paths * self.data_scaling
        
        # Add basepoint (prepend zeros)
        if self.add_basepoint:
            basepoint = torch.zeros(B, 1, D, device=paths.device, dtype=paths.dtype)
            paths = torch.cat([basepoint, paths], dim=1)
            L = L + 1
        
        # Add time channel manually (more reliable than pysiglib's time_aug)
        if self.time_aug:
            time_channel = torch.linspace(0, 1, L, device=paths.device, dtype=paths.dtype)
            time_channel = time_channel.unsqueeze(0).unsqueeze(-1).expand(B, L, 1)
            paths = torch.cat([time_channel, paths], dim=-1)
        
        return paths
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute signature kernel Gram matrix.
        
        Parameters
        ----------
        x : Tensor, shape (B1, C, L) or (B1, L) 
        y : Tensor, shape (B2, C, L) or (B2, L)
        
        Returns
        -------
        K : Tensor, shape (B1, B2)
        """
        if not self._pysiglib_available:
            # Fallback to RBF
            return self._fallback_kernel(x, y)
        
        try:
            # Convert to path format (B, L, D) with subsampling
            x_paths = self._prepare_paths(x)
            y_paths = self._prepare_paths(y)
            
            # Preprocess (normalize, scale, add basepoint, add time)
            x_prep = self._preprocess_paths(x_paths)
            y_prep = self._preprocess_paths(y_paths)
            
            # Compute signature kernel Gram matrix
            # Note: time_aug=False because we added time manually
            K = self._sig_kernel_gram(
                x_prep, 
                y_prep,
                dyadic_order=self.dyadic_order,
                time_aug=False,  # We added time manually above
                static_kernel=self._static_kernel,
                max_batch=self.max_batch,  # Control memory usage
            )
            
            return K
        except Exception as e:
            # If signature kernel fails (e.g., CUDA error), fall back to RBF
            warnings.warn(
                f"SignatureKernel failed with error: {e}. "
                f"Falling back to RBF kernel."
            )
            return self._fallback_kernel(x, y)


def create_kernel(
    kernel_type: str = "rbf",
    **kwargs
) -> Callable:
    """Factory function to create kernel objects.
    
    Parameters
    ----------
    kernel_type : str
        "rbf", "signature", or "euclidean"
    **kwargs
        Kernel-specific parameters:
        
        For RBF:
            sigma : float (default 1.0)
        
        For Signature:
            time_aug : bool (default True) - add time dimension
            lead_lag : bool (default False) - lead-lag transform
            dyadic_order : int (default 2) - signature truncation level
            static_kernel_type : str (default "rbf") - "linear" or "rbf"
            static_kernel_sigma : float (default 1.0) - RBF bandwidth
            add_basepoint : bool (default True) - prepend zeros
            normalize : bool (default False) - normalize paths
            data_scaling : float (default None) - scale data channels
            max_seq_len : int (default 128) - max path length (longer paths subsampled)
            max_batch : int (default 64) - max batch size for pysiglib
        
    Returns
    -------
    kernel_fn : Callable or None
        Kernel function, or None for Euclidean cost
    """
    if kernel_type == "euclidean" or kernel_type is None:
        return None
    elif kernel_type == "rbf":
        sigma = kwargs.get("sigma", 1.0)
        return RBFKernel(sigma=sigma)
    elif kernel_type == "signature":
        return SignatureKernel(
            time_aug=kwargs.get("time_aug", True),
            lead_lag=kwargs.get("lead_lag", False),
            dyadic_order=kwargs.get("dyadic_order", 2),
            static_kernel_type=kwargs.get("static_kernel_type", "rbf"),
            static_kernel_sigma=kwargs.get("static_kernel_sigma", 1.0),
            add_basepoint=kwargs.get("add_basepoint", True),
            normalize=kwargs.get("normalize", False),
            data_scaling=kwargs.get("data_scaling", None),
            max_seq_len=kwargs.get("max_seq_len", 128),
            max_batch=kwargs.get("max_batch", 64),
        )
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

