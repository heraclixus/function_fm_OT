# Optimal Transport in Functional Flow Matching

This document explains the four different uses of "Optimal Transport" in this codebase and how they relate mathematically.

## Overview

| Concept | Purpose | Level | Code Location |
|---------|---------|-------|---------------|
| **FFM OT Path** | Interpolation trajectory | Continuous (path) | `functional_fm.py` |
| **Euclidean OT** | Sample pairing | Discrete (coupling) | `optimal_transport.py` |
| **Kernel OT** | Sample pairing (RKHS cost) | Discrete (coupling) | `optimal_transport.py` |
| **Gaussian OT** | Sample pairing (closed-form) | Continuous approximation | `optimal_transport.py` |

All four are rooted in **optimal transport theory**, but applied at different levels and with different cost functions.

---

## 1. FFM OT Path — The Interpolation Trajectory

### What It Does
Defines **how to interpolate** between a noise sample `z` and a data sample `f`:

$$x_t = t \cdot f + \sigma_t \cdot z, \quad \sigma_t = 1 - (1 - \sigma_{\min}) t$$

### Mathematical Background
This is the **McCann displacement interpolation** (Wasserstein geodesic) between two Gaussian measures:
- Source: $\mu_0 = \mathcal{N}(0, C_0)$ (GP prior)
- Target: $\mu_1^f = \mathcal{N}(f, \sigma_{\min}^2 C_0)$ (concentrated around data)

For Gaussians with the same covariance structure, the optimal transport map is **linear**, and the geodesic is simply linear interpolation.

### Code
```python
# functional_fm.py, lines 47-49
else:
    mu = t * x_data                          # m_t^f = tf
    sigma = 1. - (1. - self.sigma_min) * t   # σ_t = 1 - (1-σ_min)t
```

### Key Reference
- McCann, R. J. (1997). "A convexity principle for interacting gases"

---

## 2. Euclidean OT — Discrete Sample Pairing

### What It Does
Solves the **Kantorovich problem** to find optimal pairing between minibatch samples:

$$\pi^* = \arg\min_{\pi \in \Pi(a, b)} \sum_{i,j} \pi_{ij} \|f_i - z_j\|^2$$

where $\Pi(a,b)$ is the set of couplings with marginals $a$ and $b$.

### Mathematical Background
This is the classic **discrete optimal transport problem**:
- Given: batch of data samples $\{f_1, ..., f_n\}$ and noise samples $\{z_1, ..., z_n\}$
- Find: optimal matching that minimizes total squared Euclidean distance
- Solvers: Exact (EMD), Sinkhorn (entropic regularization)

### Code
```python
# optimal_transport.py, KernelOTPlanSampler
def get_map(self, x0, x1, kernel_fn=None):
    """Compute OT plan between x0 (noise) and x1 (data)."""
    if kernel_fn is None:
        # Euclidean cost: c(x,z) = ||x - z||²
        M = torch.cdist(x0_flat, x1_flat, p=2) ** 2
    else:
        # Kernel cost: c(x,z) = k(x,x) + k(z,z) - 2k(x,z)
        M = self.compute_kernel_cost(x0, x1, kernel_fn)
    
    # Solve OT problem
    pi = self.ot_fn(a, b, M.cpu().numpy())
    return torch.from_numpy(pi)
```

### Variants in Codebase
| Method | Description | When to Use |
|--------|-------------|-------------|
| `exact` | Earth Mover's Distance (linear programming) | Small batches, need exact solution |
| `sinkhorn` | Entropic regularization | Large batches, fast approximation |
| `unbalanced` | Allows mass variation | Outliers, varying sample sizes |

---

## 3. Kernel OT — RKHS-Based Optimal Transport

### What It Does

Instead of using squared Euclidean distance, kernel OT uses **RKHS (Reproducing Kernel Hilbert Space) distance** as the ground cost:

$$c(x, z) = \|\phi(x) - \phi(z)\|_{\mathcal{H}_k}^2 = k(x,x) + k(z,z) - 2k(x,z)$$

where:
- $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ is a positive definite kernel
- $\phi: \mathcal{X} \to \mathcal{H}_k$ is the induced feature map
- $\mathcal{H}_k$ is the RKHS associated with $k$

### Problem Formulation

The kernel OT problem is the standard **Kantorovich problem** with kernel-induced cost:

$$\pi^* = \arg\min_{\pi \in \Pi(a, b)} \sum_{i,j} \pi_{ij} \cdot c(x_i, z_j)$$

where:
- $\{x_1, \ldots, x_n\}$ and $\{z_1, \ldots, z_m\}$ are source and target samples
- $a \in \Delta^n$, $b \in \Delta^m$ are marginal distributions (typically uniform)
- $\Pi(a, b) = \{\pi \in \mathbb{R}_+^{n \times m} : \pi \mathbf{1} = a, \pi^T \mathbf{1} = b\}$ is the set of valid couplings

### Cost Matrix Computation

The kernel cost matrix is computed as:

```python
# optimal_transport.py, KernelOTPlanSampler.compute_kernel_cost()
K_00 = kernel_fn(x0, x0)  # Gram matrix (B, B)
K_11 = kernel_fn(x1, x1)  # Gram matrix (B, B)
K_01 = kernel_fn(x0, x1)  # Cross-Gram matrix (B, B)

# RKHS distance: c(x,z) = k(x,x) + k(z,z) - 2k(x,z)
C[i,j] = diag(K_00)[i] + diag(K_11)[j] - 2 * K_01[i,j]
```

### Why Kernel Costs?

Euclidean distance treats each coordinate independently and ignores functional structure:

| Aspect | Euclidean Cost | Kernel Cost |
|--------|----------------|-------------|
| Metric | $\|x - z\|^2 = \sum_t (x_t - z_t)^2$ | $\|\phi(x) - \phi(z)\|_{\mathcal{H}}^2$ |
| Structure | Pointwise | Feature-based |
| Time warping | Sensitive | Can be robust (signature) |
| Phase shifts | Penalized heavily | Can be handled |

### Available Kernels

| Kernel | Formula | Properties | Best For |
|--------|---------|------------|----------|
| **Euclidean** | $\|x - z\|^2$ | Baseline, coordinate-wise | General functions |
| **RBF** | $\exp(-\|x-z\|^2 / 2\sigma^2)$ | Smooth, localized similarity | Smooth functions |
| **Signature** | $\langle S(x), S(y) \rangle$ | Path-sensitive, order-aware | Time series, paths |

### RBF Kernel

The RBF (Radial Basis Function) kernel measures similarity in a Gaussian-weighted sense:

$$k_{\text{RBF}}(x, z) = \exp\left(-\frac{\|x - z\|^2}{2\sigma^2}\right)$$

- **Bandwidth σ**: Controls locality. Small σ = sharp, large σ = smooth
- **Feature space**: Infinite-dimensional Gaussian RKHS
- **RKHS distance**: As σ → ∞, approaches rescaled Euclidean distance

### Signature Kernel

The signature kernel captures **sequential/path structure**:

$$k_{\text{sig}}(x, z) = \langle S(x), S(z) \rangle$$

where $S(x)$ is the signature of path $x$—a sequence of iterated integrals encoding:
- Order of events
- Areas enclosed by path
- Higher-order path interactions

**Implementation** (`SignatureKernel` class):
- Uses `pySigLib` for efficient GPU computation
- Supports time augmentation: $(t, x_t)$ to make paths truly 1D-parametrized
- Lead-lag transform for capturing quadratic variation
- Static kernels (linear/RBF) on path increments

### Numerical Stability

Kernel costs can be large for functional data. The implementation includes:

1. **Cost normalization**: Divide by median cost value
2. **Log-domain Sinkhorn**: Avoids underflow/overflow
3. **Non-negativity clamping**: Handle numerical precision issues

```python
# Normalize for stability
if self.normalize_cost:
    cost_scale = M.median()
    M = M / cost_scale

# Use log-stabilized Sinkhorn
pot.sinkhorn(..., method='sinkhorn_log')
```

### Example: Comparing Costs

Consider two paths that are identical but time-shifted:

```
x: [0, 1, 2, 1, 0]       z: [0, 0, 1, 2, 1]
    ↗ peak at t=2           ↗ peak at t=3
```

| Cost | Value | Interpretation |
|------|-------|----------------|
| Euclidean | Large | Penalizes pointwise mismatch |
| RBF (σ large) | Similar to Euclidean | Still pointwise |
| Signature | Smaller | Recognizes similar dynamics |

Signature kernel "sees" that both paths rise, peak, and fall—despite the time shift.

---

## 4. Gaussian OT (Bures-Wasserstein) — Closed-Form Pairing

### What It Does
Assumes both distributions are Gaussian and uses the **closed-form optimal transport map**:

$$T^*(z) = \mu_1 + A(z - \mu_0)$$

where:
$$A = \Sigma_0^{-1/2} \left( \Sigma_0^{1/2} \Sigma_1 \Sigma_0^{1/2} \right)^{1/2} \Sigma_0^{-1/2}$$

### Mathematical Background
This is the **Bures-Wasserstein distance** between Gaussians:

$$W_2^2(\mathcal{N}(\mu_0, \Sigma_0), \mathcal{N}(\mu_1, \Sigma_1)) = \|\mu_0 - \mu_1\|^2 + \text{Tr}\left(\Sigma_0 + \Sigma_1 - 2(\Sigma_0^{1/2} \Sigma_1 \Sigma_0^{1/2})^{1/2}\right)$$

The optimal map $T^*$ is deterministic and affine.

### Code
```python
# optimal_transport.py, GaussianOTPlanSampler
def compute_mapping(self, x0, x1):
    """Compute optimal linear mapping T(x) = Ax + b."""
    # Estimate Gaussian parameters
    mu_s, Sigma_s = self._estimate_gaussian_params(x0)
    mu_t, Sigma_t = self._estimate_gaussian_params(x1)
    
    # POT's closed-form solution
    A, b = pot.gaussian.bures_wasserstein_mapping(
        mu_s, mu_t, Sigma_s, Sigma_t
    )
    return A, b

def transport(self, x0, x1):
    """Apply T(x) = Ax + b to source samples."""
    A, b = self.compute_mapping(x0, x1)
    return x0_flat @ A.T + b
```

### Advantages
- **Fast**: O(D³) vs O(n³) for exact OT
- **Deterministic**: No stochasticity in coupling
- **Differentiable**: Can backprop through mapping

### Limitations
- Assumes data is well-approximated by a Gaussian
- May not capture complex multimodal structure

---

## How They Work Together

```
                        STEP 1: PAIRING (Euclidean, Kernel, or Gaussian OT)
                        ═══════════════════════════════════════════════════
                        
    Data:    f₁ ──┐              ┌── f₃
                  │              │
    Data:    f₂ ──┼──────────────┼── f₁
                  │     π*       │
    Data:    f₃ ──┼──────────────┼── f₂
                  │              │
    Noise:   z₁ ──┘              └── z₃
    Noise:   z₂ ─────────────────── z₁
    Noise:   z₃ ─────────────────── z₂

                        STEP 2: INTERPOLATION (FFM OT Path)
                        ═══════════════════════════════════════════
                        
    For each matched pair (zⱼ, fᵢ):
    
    t=0                                                t=1
     zⱼ ●═══════════════════════════════════════════════● fᵢ
           x_t = t·fᵢ + σ_t·zⱼ  (OT geodesic)
```

### The Complete Pipeline

1. **Sample noise**: $z \sim \mathcal{N}(0, C_0)$ (GP prior)
2. **Sample data**: $f \sim \nu$ (training distribution)  
3. **Pair samples**: Find optimal coupling $\pi^*$ using one of:
   - **Euclidean OT**: $c(x,z) = \|x - z\|^2$
   - **Kernel OT**: $c(x,z) = k(x,x) + k(z,z) - 2k(x,z)$ (structure-aware)
   - **Gaussian OT**: Closed-form map $T^*(z) = Az + b$
4. **Interpolate**: For each pair $(z_j, f_i) \sim \pi^*$, compute $x_t$ along OT path
5. **Train**: Regress vector field $v_\theta(t, x_t) \approx \frac{d}{dt}x_t$

---

## Mathematical Relationships

### Theorem (McCann, 1997)
For probability measures $\mu_0, \mu_1$ on $\mathbb{R}^d$ with $\mu_0 \ll \mathcal{L}^d$, the Wasserstein-2 geodesic is:

$$\mu_t = [(1-t) \cdot \text{Id} + t \cdot T^*]_\# \mu_0$$

where $T^* = \nabla \phi$ is the Brenier optimal transport map.

### Corollary (Gaussian Case)
When $\mu_0 = \mathcal{N}(0, \Sigma_0)$ and $\mu_1 = \mathcal{N}(m, \Sigma_1)$:
- **Optimal map**: $T^*(z) = m + A(z - 0)$ is affine
- **Geodesic**: $\mu_t = \mathcal{N}((1-t)\cdot 0 + t \cdot m, \Sigma_t)$ where $\Sigma_t$ interpolates covariances

### FFM Connection
FFM's OT path with $m_t^f = tf$ and $\sigma_t = 1 - (1-\sigma_{\min})t$ is exactly the McCann interpolation when:
- Source: $\mu_0 = \mathcal{N}(0, C_0)$
- Target: $\mu_1^f = \mathcal{N}(f, \sigma_{\min}^2 C_0)$

---

## Comparison Table

| Aspect | FFM OT Path | Euclidean OT | Kernel OT | Gaussian OT |
|--------|-------------|--------------|-----------|-------------|
| **Question** | How to interpolate? | Which to pair? | Which to pair? (structure-aware) | Which to pair? (assuming Gaussian) |
| **Input** | Single pair $(z, f)$ | Batch $\{z_i\}, \{f_j\}$ | Batch $\{z_i\}, \{f_j\}$ + kernel $k$ | Batch $\{z_i\}, \{f_j\}$ |
| **Output** | Path $x_t$ | Coupling $\pi^*$ | Coupling $\pi^*$ | Map $T^*$ |
| **Cost** | N/A | $\|x - z\|^2$ | $k(x,x) + k(z,z) - 2k(x,z)$ | Bures-Wasserstein |
| **Computation** | $O(1)$ | $O(n^3)$ exact, $O(n^2)$ Sinkhorn | $O(n^2 \cdot L)$ + $O(n^2)$ Sinkhorn | $O(D^3)$ |
| **Assumption** | Gaussian conditionals | None | None (kernel choice matters) | Both distributions Gaussian |
| **Stochasticity** | Deterministic path | Stochastic sampling from $\pi^*$ | Stochastic sampling from $\pi^*$ | Deterministic map |
| **Best for** | Always used | General data | Sequential/path data | Gaussian-like data |

---

## Why Two Steps? Pairing + Interpolation

A natural question: **Why not just interpolate?** Does interpolation alone provide optimal pairing?

**Answer: No.** The interpolation formula `x_t = t·f + σ_t·z` doesn't determine which `z` pairs with which `f`. The two-step process is essential for optimal marginal flows.

### The Problem with Independent Sampling

When we sample `(f, z)` **independently**:
- Each **conditional path** `x_t = t·f + σ_t·z` is optimal (straight line)
- But the **marginal distribution** `μ_t` is NOT the Wasserstein geodesic!

```
        Marginal flow μ_t (NOT straight)
              ╱    ╲
             ╱      ╲
    z₁ ●━━━╳━━━━━━━━━● f₂    ← Paths CROSS!
           ╲        ╱
            ╲      ╱
    z₂ ●━━━━━╳━━━━━● f₁
```

**Problem**: Crossing paths create complex, multi-valued marginal vector fields that are harder to learn.

### The Solution: OT Pairing First

When we sample `(f, z) ~ π*` from the **optimal coupling**:

```
        Marginal flow μ_t (IS the Wasserstein geodesic)
    
    z₁ ●━━━━━━━━━━━━━━━● f₁    ← Paths DON'T cross!
    
    z₂ ●━━━━━━━━━━━━━━━● f₂
```

### Theorem (Tong et al., 2023)

Let `π*` be the optimal coupling between `μ₀` and `ν` for squared Euclidean cost. If:

$$x_t = (1-t)z + tf \quad \text{where } (z, f) \sim \pi^*$$

Then the marginal distribution satisfies:

$$\mu_t = [(1-t)\cdot\text{Id} + t\cdot T^*]_\# \mu_0$$

which is **exactly the McCann displacement interpolation** (Wasserstein geodesic).

### Comparison

| Sampling | Conditional Path | Marginal Path | Vector Field |
|----------|-----------------|---------------|--------------|
| Independent | Straight ✓ | Curved ✗ | Complex, multi-modal |
| OT Pairing | Straight ✓ | Straight ✓ | Simple, unimodal |

### Analogy: Moving Particles

Imagine moving a pile of sand from location A to location B:

- **Independent**: Each grain goes to a random target location
  - Some grains travel far, some short distances
  - Grains collide and interfere with each other
  
- **OT Pairing**: Each grain goes to its "nearest" target (Monge map)
  - Minimizes total work
  - No crossings = smooth flow

### Empirical Benefits

| Metric | Independent | OT Pairing |
|--------|-------------|------------|
| Training convergence | Slower | **Faster** |
| Vector field complexity | Higher | **Lower** |
| Sample quality | Good | **Better** |
| Path lengths | Variable | **Shorter** |

### Summary

```
┌────────────────────────────────────────────────────────────────┐
│  INDEPENDENT SAMPLING (Base FFM)                               │
│  ═══════════════════════════════                               │
│  Conditional: x_t = t·f + σ_t·z  ← Optimal (straight)          │
│  Marginal:    μ_t               ← NOT optimal (curved)         │
│                                                                │
│  Why? Random pairing → crossing paths → complex v_t(x)         │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  OT PAIRING (OT-FFM)                                           │
│  ═══════════════════                                           │
│  Step 1: (f, z) ~ π*            ← Find optimal coupling        │
│  Step 2: x_t = t·f + σ_t·z      ← Interpolate along geodesic   │
│                                                                │
│  Conditional: Optimal (straight)                               │
│  Marginal:    ALSO optimal (Wasserstein geodesic)              │
│                                                                │
│  Why? Non-crossing paths → simple v_t(x) → easier to learn     │
└────────────────────────────────────────────────────────────────┘
```

**Bottom line**: The two-step process ensures that BOTH the conditional paths AND the marginal flow are optimal. Interpolation alone only guarantees optimal conditional paths.

---

## References

1. **McCann, R. J.** (1997). A convexity principle for interacting gases. *Advances in Mathematics*.

2. **Lipman, Y., et al.** (2022). Flow Matching for Generative Modeling. *ICLR*.

3. **Tong, A., et al.** (2023). Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport. *ICML*.

4. **Peyré, G. & Cuturi, M.** (2019). Computational Optimal Transport. *Foundations and Trends in Machine Learning*.

5. **Kerrigan, G., et al.** (2024). Functional Flow Matching. *arXiv*.

