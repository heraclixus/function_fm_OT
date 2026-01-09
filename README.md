# Functional Flow Matching with Optimal Transport

This repository implements **Kernel Functional Flow Matching (k-FFM)**, extending Functional Flow Matching with minibatch optimal transport coupling using kernel-induced ground costs.

## Overview

Standard FFM uses independent sampling between data functions and GP noise. We improve this by computing OT pairings under kernel-induced RKHS metrics:

- **Euclidean**: Standard L² distance
- **RBF**: Radial basis function kernel distance
- **Signature**: Signature kernel for time series (captures path geometry)

## Installation

```bash
pip install torch torchdiffeq gpytorch linear_operator scipy POT signatory
```

## Project Structure

```
├── functional_fm_ot.py     # Core FFMModelOT class
├── optimal_transport.py    # OT solvers and kernel implementations
├── diffusion.py            # DDPM/NCSN baselines
├── gano.py                 # GANO baseline
├── models/
│   ├── fno.py              # Fourier Neural Operator
│   └── gano_models.py      # Generator/Discriminator for GANO
├── scripts/
│   ├── *_ot.py             # Dataset-specific experiments
│   ├── run_seeded_experiments.py  # Multi-seed evaluation
│   └── generate_tables.py  # LaTeX table generation
└── data/                   # Datasets
```

## Datasets

| Type | Datasets |
|------|----------|
| **Sequence/Path** | AEMET (weather), Economy, Heston, rBergomi, Gene Expression |
| **1D PDE** | KdV, Stochastic KdV |
| **2D PDE** | Navier-Stokes, Stochastic NS, Ginzburg-Landau |

## Quick Start

### Single Experiment
```bash
# Run AEMET with signature kernel OT
python scripts/AEMET_ot.py

# Run Navier-Stokes with RBF kernel
python scripts/navier_stokes_ot.py
```

### Seeded Experiments (10 seeds)
```bash
# Run best config for a dataset/kernel combination
python scripts/run_seeded_experiments.py --dataset aemet --kernel signature

# Available kernels: none, signature, rbf, euclidean, ddpm, ncsn, gano
```

### Generate Results Tables
```bash
python scripts/generate_tables.py --type sequence
python scripts/generate_tables.py --type pde
```

## Key Features

- **Kernel OT Pairing**: Improved sample quality via structured transport
- **FNO Backbone**: Resolution-invariant generation (super-resolution capable)
- **Comprehensive Baselines**: FFM, DDPM, NCSN, GANO
- **Evaluation Metrics**: Mean, Variance, Skewness, Kurtosis, Autocorrelation, Spectrum MSE

## Citation

Based on:
- Kerrigan et al. "Functional Flow Matching"
- Tong et al. "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport"

## License

See [LICENSE](LICENSE) for details.
