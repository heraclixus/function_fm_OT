# Kernel-OT Functional Flow Matching with Signature Kernels (pySigLib)

We propose an extension of **Functional Flow Matching (FFM)** that replaces the usual *random pairing* between **data functions** and **base Gaussian samples** with a **minibatch optimal-transport (OT) coupling**, where the OT **ground cost** is defined by a **signature kernel / RKHS geometry** computed using **pySigLib (torch_api)**.

---

## 1. Goal

Given:
- data distribution over (discretized) functions / time series: $X \sim \mu$,
- base distribution (Gaussian measure in function space, discretized): $Z \sim \nu$,

FFM trains a time-dependent vector field $v_\theta(t,\cdot)$ so that the ODE flow transports samples from $\nu$ to $\mu$ along a chosen probability path.

**Proposal:** Keep the *same* FFM probability path + conditional target construction, but change how endpoint pairs $(x,z)$ are drawn:
- instead of independent sampling, use **minibatch entropic OT pairing**
- with a **signature-kernel-induced RKHS distance** as OT ground cost.

---

## 2. Objects and notation

### 2.1 Discretized functional data
Represent each function / time series on a uniform grid:
- $x \in \mathbb{R}^{L \times d}$,
- $z \in \mathbb{R}^{L \times d}$,

and minibatches:
- $X = \{x_i\}_{i=1}^B$,
- $Z = \{z_j\}_{j=1}^B$.

(For 1D time series, typically $d=1$ unless you augment the channel dimension.)

### 2.2 FFM probability path and conditional sampling
FFM fixes:
- a probability path $\{\rho_t\}_{t\in[0,1]}$ between $\nu$ and $\mu$,
- a conditional sampling rule $f_t \sim q_t(\cdot\mid x,z)$,
- a conditional velocity target $u(t,f_t\mid x,z)$,

and learns $v_\theta$ by regression.

**This proposal does not alter** $\rho_t$, $q_t$, or $u$.  
It only changes the sampling distribution of $(x,z)$.

---

## 3. Signature kernel geometry from pySigLib

pySigLib provides a **signature kernel** on paths. Let $k(\cdot,\cdot)$ denote the signature kernel (implemented by `sig_kernel_gram` in torch_api).

### 3.1 Path augmentation choices (important!)
Given a raw series $x = (x_{t_1},\ldots,x_{t_L})\in\mathbb{R}^{L\times d}$, define a path representation $X(\cdot)$.

Common augmentations (supported via pySigLib flags):
- **Time augmentation:** $\widehat x_t = (t, x_t)$  (set `time_aug=True`)
- **Lead–lag transform:** promotes sensitivity to local ordering / “quadratic-variation-like” structure (set `lead_lag=True`)
- **Dyadic refinement:** refines discretization by a factor $2^\lambda$ (set `dyadic_order=λ`)

These control what “similarity” means in the signature kernel.

### 3.2 RKHS distance as OT ground cost
The signature kernel induces an RKHS $\mathcal K$ and feature map $\varphi(\cdot)$ with
$$
k(x,y) = \langle \varphi(x), \varphi(y)\rangle_{\mathcal K}.
$$

Define the OT ground cost as squared RKHS distance:
$$
c(x,z) = \|\varphi(x)-\varphi(z)\|_{\mathcal K}^2
       = k(x,x)+k(z,z)-2k(x,z).
$$

Given minibatches $X, Z$, compute Gram matrices:
- $K_{XX}[i,i'] = k(x_i,x_{i'})$,
- $K_{ZZ}[j,j'] = k(z_j,z_{j'})$,
- $K_{XZ}[i,j]  = k(x_i,z_j)$.

Then the **cost matrix** $C\in\mathbb{R}^{B\times B}$ is
$$
C_{ij} = (K_{XX})_{ii} + (K_{ZZ})_{jj} - 2 (K_{XZ})_{ij}.
$$

---

## 4. Minibatch entropic OT coupling

Define uniform weights $\mathbf u=\tfrac{1}{B}\mathbf 1$. The entropic OT plan $\pi^\star\in\mathbb{R}_+^{B\times B}$ solves:
$$
\pi^\star
= \arg\min_{\pi\in\Pi(\mathbf u,\mathbf u)}
\langle \pi, C\rangle
+ \varepsilon \sum_{i,j}\pi_{ij}\,(\log \pi_{ij}-1),
$$
where:
- $\Pi(\mathbf u,\mathbf u)$ are couplings with marginals $\mathbf u$,
- $\varepsilon>0$ is Sinkhorn (entropy) regularization.

### 4.1 Practical gradient handling (recommended)
Treat OT matching as a **non-differentiable** step:
- compute $\pi^\star$ from a detached $C$,
- backprop only through the FFM regression loss.

This usually stabilizes training and avoids making the Sinkhorn solver part of the gradient path.

### 4.2 Pairing strategies
Once $\pi^\star$ is computed, form endpoint pairs by either:

**(A) Sampling from the coupling**
- draw indices $(i,j)\sim\pi^\star$,
- use $(x,z)=(x_i,z_j)$.

**(B) Barycentric mapping (soft pairing)**
- define $\tilde z_i := \sum_j \pi^\star_{ij}\,z_j$,
- pair $(x_i,\tilde z_i)$.

Choice (A) is simplest; (B) can reduce variance when convex combinations in your representation are meaningful.

---

## 4.3 Theoretical Analysis: OT Simplifications Under Gaussian Measure Assumptions

FFM assumes that data lies in the Cameron-Martin space of the GP prior, and that all intermediate measures along the flow are Gaussian (see `gaussian_measure.tex`). This has important implications for optimal transport.

### Optimal Transport Between Gaussians (Gelbrich/Bures Formula)

For two Gaussian measures $\mu = \mathcal{N}(m_1, \Sigma_1)$ and $\nu = \mathcal{N}(m_2, \Sigma_2)$, the 2-Wasserstein distance has a closed-form solution:
$$
W_2^2(\mu, \nu) = \|m_1 - m_2\|^2 + \mathrm{Tr}\left(\Sigma_1 + \Sigma_2 - 2(\Sigma_2^{1/2} \Sigma_1 \Sigma_2^{1/2})^{1/2}\right)
$$

The optimal transport map $T: \mathbb{R}^n \to \mathbb{R}^n$ is also available in closed form:
$$
T(x) = m_2 + A(x - m_1), \quad A = \Sigma_1^{-1/2}(\Sigma_1^{1/2} \Sigma_2 \Sigma_1^{1/2})^{1/2} \Sigma_1^{-1/2}
$$

### Special Case: Equal Covariance

When $\Sigma_1 = \Sigma_2 = \Sigma$, the OT map simplifies dramatically to a **pure translation**:
$$
T(x) = x + (m_2 - m_1)
$$
and the Wasserstein distance reduces to:
$$
W_2^2(\mu, \nu) = \|m_1 - m_2\|^2
$$

### Connection to FFM Probability Path

In FFM, we have:
- Base measure: $\mu_0 = \mathcal{N}(0, C_0)$
- Conditional measure at time $t$: $\mu_t^f = \mathcal{N}(m_t^f, \sigma_t^2 C_0)$

The "OT" path parametrization $m_t^f = tf$ is precisely the **McCann displacement interpolation** (W₂-geodesic) between Gaussians with equal covariance structure. This means the FFM probability path is already optimal in the Wasserstein sense when data shares the GP prior's covariance.

### Implications for Minibatch OT Implementation

| Data Distribution | Recommended OT Cost | Rationale |
|-------------------|---------------------|-----------|
| Gaussian with covariance ≈ C₀ | **Euclidean (L²)** | OT between equal-covariance Gaussians uses squared L² distance |
| Mixture of Gaussians | Euclidean or RBF kernel | May benefit from kernel smoothing to handle modes |
| Non-Gaussian / structured time series | **Signature kernel** | Captures sequential structure beyond pointwise L² |

### When Does Signature Kernel OT Provide Value?

The signature kernel OT is most beneficial when the data distribution **deviates from Gaussian assumptions**:

1. **Mixture models**: Data consists of distinct modes/clusters that should be matched coherently
2. **Non-stationary dynamics**: Time series with regime changes, jumps, or non-Gaussian innovations
3. **Path-dependent structure**: When the *shape* of trajectories matters beyond pointwise values
4. **Invariances**: When you want robustness to time reparametrization (captured by signatures)

For data that is approximately Gaussian with covariance similar to the GP prior, **Euclidean OT is theoretically optimal** and computationally cheaper. The signature kernel OT serves as a more expressive alternative when Gaussian assumptions break down.

### FAQ: Does FFM Require Gaussian Data?

**Q: Does FFM work if data is NOT from a Gaussian measure?**

**A: Yes, practically. But theoretical guarantees weaken.**

The training objective (regressing on conditional vector fields) is well-defined for *any* data distribution:
- Sample data $x$, sample GP noise $z$, compute conditional field $v_t^x$, and regress
- The finite-dimensional discretization helps: on a grid of $L$ points, you're in $\mathbb{R}^L$ where absolute continuity is less problematic

What you lose theoretically:
- The marginal vector field formula requires **absolute continuity** between intermediate measures
- Gaussian measures guarantee this via Cameron-Martin/Feldman-Hájek theorems
- For non-Gaussian data, you lose the guarantee that the learned flow exactly transports $\mu_0 \to \nu$

The paper's pragmatic stance: *"In practice, we do not find it necessary"* to verify the Cameron-Martin space assumption.

### FAQ: Is OT Useful When Data IS Gaussian?

**Q: If data comes from a Gaussian measure, are the probability paths the same with/without OT?**

**A: The TARGET is identical. Training dynamics differ.**

The *conditional* vector field $v_t^f(g)$ and probability path $\mu_t^f$ are **identical** regardless of pairing strategy. What changes is which pairs $(x, z)$ you sample during training:

| Aspect | Same? | Explanation |
|--------|-------|-------------|
| Conditional path $\mu_t^f$ | ✅ Same | Defined by interpolation formula, not pairing |
| Conditional field $v_t^f$ | ✅ Same | Derived from $\mu_t^f$ |
| Marginal field $v_t$ (target) | ✅ Same | The integral $\int v_t^f \, d\nu(f)$ is pairing-independent |
| Training distribution | ❌ Different | Which $(x,z)$ pairs you see differs |
| Gradient variance | ❌ Different | OT typically reduces variance |
| Convergence speed | ❌ Different | OT often converges faster |

**Why OT can still help for Gaussian data:**
1. **Variance reduction**: Pairing "similar" samples reduces gradient noise
2. **Straighter paths**: Matched pairs yield shorter interpolations, easier to learn
3. **Finite sample effects**: Limited data and training budget favor OT's efficiency

**When OT provides NO additional benefit (limiting case):**
- Data is exactly $\mathcal{N}(m, C_0)$ with identical covariance to base GP
- Infinite samples and infinite training time
- Both methods converge to the *same* marginal vector field

**Conclusion**: For Gaussian data, OT is a **training efficiency improvement**, not a change to the target. For non-Gaussian data, OT may genuinely improve the learned transport by providing better inductive bias.

### Practical Recommendation

Start with **Euclidean OT** as a baseline (theoretically justified under Gaussian assumptions). Switch to **signature kernel OT** when:
- Visual inspection suggests non-Gaussian structure (multimodality, heavy tails, temporal patterns)
- Euclidean OT does not improve over independent pairing
- You specifically want to enforce path-aware matching

---

## 5. Training objective (unchanged FFM regression, OT-paired endpoints)

Sample:
- minibatches $X=\{x_i\}\sim\mu$, $Z=\{z_j\}\sim\nu$,
- compute $\pi^\star$ from signature-kernel OT,
- sample endpoint pair $(x,z)\sim\pi^\star$,
- sample $t\sim \mathrm{Unif}[0,1]$,
- sample $f_t \sim q_t(\cdot\mid x,z)$.

Then train the vector field $v_\theta$ with:
$$
\mathcal L_{\mathrm{FFM}}(\theta)
= \mathbb E\Big[\|v_\theta(t,f_t) - u(t,f_t\mid x,z)\|_{\mathcal H}^2\Big].
$$

**Key point:** Only the distribution of $(x,z)$ changes (now OT-coupled in signature RKHS).  
All other FFM mechanics remain intact.

---

## 6. Optional: signature MMD for monitoring and/or regularization

pySigLib also provides signature-kernel MMD. For distributions $\alpha,\beta$ on paths:
$$
\mathrm{MMD}_k^2(\alpha,\beta)
= \mathbb E[k(A,A')] - 2\mathbb E[k(A,B)] + \mathbb E[k(B,B')].
$$

Recommended uses:
- **Evaluation metric:** compare generated samples $\hat X$ vs real $X$ in the same signature kernel geometry.
- **Auxiliary regularizer:** add $\lambda_{\mathrm{mmd}}\mathrm{MMD}_k^2(\hat\mu_\theta,\mu)$ (typically at $t=1$).

---

## 7. Suggested signature-kernel configurations (time series)

These are “first try” settings; tune later.

### 7.1 Baseline (stable, general-purpose)
- `time_aug=True`
- `lead_lag=False`
- `dyadic_order=0` (or 1 if discretization is coarse)

### 7.2 For spiky / regime-switching series
- `time_aug=True`
- `lead_lag=True`  (often improves sensitivity to local structure)
- `dyadic_order=0` or 1

### 7.3 For short series (small L)
- keep `dyadic_order` small (0),
- consider `lead_lag=True` if you need extra expressivity.

---

## 8. Minimal algorithm sketch

At each training iteration:

1. Sample minibatch $X=\{x_i\}_{i=1}^B\sim\mu$, $Z=\{z_j\}_{j=1}^B\sim\nu$.
2. Build signature-kernel Gram matrices $K_{XX}, K_{ZZ}, K_{XZ}$.
3. Form cost matrix $C_{ij}=(K_{XX})_{ii}+(K_{ZZ})_{jj}-2(K_{XZ})_{ij}$.
4. Solve entropic OT to get $\pi^\star$.
5. Sample endpoint pairs $(x,z)$ using $\pi^\star$ (sampling or barycentric).
6. Run standard FFM:
   - sample $t$,
   - sample $f_t\sim q_t(\cdot\mid x,z)$,
   - compute target $u(t,f_t\mid x,z)$,
   - SGD step on $\|v_\theta(t,f_t)-u\|^2$.
7. (Optional) Monitor / regularize with signature MMD between generated vs real batches.

---

## 9. Integration points in `functional_flow_matching` repo

You should be able to integrate with minimal surface area by:

- locating the **training loop** where:
  - a batch of data functions $x$ is drawn,
  - a batch of base Gaussian samples $z$ is drawn,
  - the FFM loss is computed.

Then:
- insert a **Kernel-OT pairing module** before constructing the conditional FFM samples/targets,
- replace independent pairing with $(x,z)\sim\pi^\star$,
- keep the remainder of the pipeline unchanged.

Recommended initial strategy:
- compute OT plan $\pi^\star$ **without gradient**,
- do not introduce MMD regularization until OT-pairing baseline is stable.

---

## 10. Hyperparameters to tune (minimal set)

### Signature kernel / pySigLib
- `time_aug ∈ {True, False}` (usually True)
- `lead_lag ∈ {True, False}`
- `dyadic_order ∈ {0,1,2}`

### OT / Sinkhorn
- minibatch size $B$
- entropic regularization $\varepsilon$
- coupling strategy (sampling vs barycentric)

### Optional MMD
- $\lambda_{\mathrm{mmd}}$ weight
- evaluation batch size and frequency

---

## 11. Expected benefits (hypotheses)

- OT pairing under a **signature RKHS geometry** should match base samples to data samples in a way that respects **sequential structure**, potentially simplifying the learned vector field.
- The same signature kernel provides a coherent **evaluation metric** (signature MMD) aligned with the pairing geometry.

---

## 12. Next steps checklist

1. Start from a baseline FFM run (repo default config) to confirm reproducibility.
2. Add signature-kernel Gram computation (time_aug first).
3. Add entropic OT pairing (detach cost for stability).
4. Verify training still converges (loss curves, sample quality).
5. Sweep `lead_lag` and `dyadic_order`.
6. Add signature-MMD evaluation (optional: regularizer).
