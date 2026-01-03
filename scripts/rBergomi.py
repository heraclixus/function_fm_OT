"""
FFM Experiments on rBergomi Rough Volatility Data.

This script runs baseline FFM experiments (without OT) on rBergomi
log-variance paths, comparing:
- FFM-OT (OT-like interpolation path)
- FFM-VP (variance-preserving path)
- DDPM
- DDO (NCSN)
- GANO

Usage:
    python rBergomi.py
    python rBergomi.py --alpha -0.3  # Different roughness
"""

import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path

from util.gaussian_process import GPPrior
from util.util import make_grid

from functional_fm import FFMModel
from diffusion import DiffusionModel
from gano1d import GANO
from models.fno import FNO, FNO1dgano

# Import data generator
from data.rBergomi import rBergomi, generate_rBergomi_dataset

# =============================================================================
# Configuration
# =============================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Data params
n_samples = 5000
n_x = 100  # Time steps
alpha = -0.4  # Roughness: H = alpha + 0.5 = 0.1

# Generate or load dataset
print("Generating rBergomi dataset...")
dataset = generate_rBergomi_dataset(
    n_samples=n_samples,
    n_steps=n_x,
    T=1.0,
    alpha=alpha,
    rho=-0.7,
    eta=1.5,
    xi=0.04,
    seed=42,
)

# Use normalized log-variance for training
train_data = dataset['log_V_normalized']
print(f"Training data shape: {train_data.shape}")

batch_size = 64
train_loader = DataLoader(
    train_data.unsqueeze(1),  # Add channel dimension
    batch_size=batch_size,
    shuffle=True,
)

# Model hyperparameters
modes = 32
width = 256
mlp_width = 128

# GP hyperparameters (for base distribution)
kernel_length = 0.01
kernel_variance = 0.1

# Training params
epochs = 100
lr = 1e-3

# FFM params
sigma_min = 1e-4

# Output directory
spath = Path('../outputs/rBergomi_baseline/')
spath.mkdir(parents=True, exist_ok=True)

# Save dataset info
torch.save(dataset, spath / 'dataset.pt')

N = 5  # Number of random seeds
n_gen_samples = 500
random_seeds = [2**i for i in range(N)]

# =============================================================================
# FFM-OT (OT-like path)
# =============================================================================

print("\n" + "="*60)
print("Training FFM-OT")
print("="*60)

samplefmot = torch.zeros(N, n_gen_samples, n_x)

for i in range(N):
    print(f"\nSeed {i+1}/{N}")
    np.random.seed(random_seeds[i])
    torch.manual_seed(random_seeds[i])
    
    model = FNO(modes, vis_channels=1, hidden_channels=width, 
                proj_channels=mlp_width, x_dim=1, t_scaling=1000).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50)
    
    fmot = FFMModel(model, kernel_length=kernel_length, kernel_variance=kernel_variance,
                    sigma_min=sigma_min, device=device)
    fmot.train(train_loader, optimizer, epochs=epochs, scheduler=scheduler,
               eval_int=0, save_int=epochs, generate=False, save_path=spath)
    
    samplefmot[i, :, :] = fmot.sample([n_x], n_samples=n_gen_samples).cpu().squeeze()

torch.save(samplefmot, spath / 'samples_fmot.pt')
print(f"Saved FFM-OT samples to {spath / 'samples_fmot.pt'}")

# =============================================================================
# FFM-VP (Variance-Preserving path)
# =============================================================================

print("\n" + "="*60)
print("Training FFM-VP")
print("="*60)

samplefmvp = torch.zeros(N, n_gen_samples, n_x)

for i in range(N):
    print(f"\nSeed {i+1}/{N}")
    np.random.seed(random_seeds[i])
    torch.manual_seed(random_seeds[i])
    
    model = FNO(modes, vis_channels=1, hidden_channels=width,
                proj_channels=mlp_width, x_dim=1, t_scaling=1000).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50)
    
    fmvp = FFMModel(model, kernel_length=kernel_length, kernel_variance=kernel_variance,
                    sigma_min=sigma_min, device=device, vp=True)
    fmvp.train(train_loader, optimizer, epochs=epochs, scheduler=scheduler,
               eval_int=0, save_int=epochs, generate=False, save_path=spath)
    
    samplefmvp[i, :, :] = fmvp.sample([n_x], n_samples=n_gen_samples).cpu().squeeze()

torch.save(samplefmvp, spath / 'samples_fmvp.pt')
print(f"Saved FFM-VP samples to {spath / 'samples_fmvp.pt'}")

# =============================================================================
# DDPM
# =============================================================================

print("\n" + "="*60)
print("Training DDPM")
print("="*60)

method = 'DDPM'
T_ddpm = 1000
beta_min = 1e-4
beta_max = 0.02

sampleddpm = torch.zeros(N, n_gen_samples, n_x)

for i in range(N):
    print(f"\nSeed {i+1}/{N}")
    np.random.seed(random_seeds[i])
    torch.manual_seed(random_seeds[i])
    
    model = FNO(modes, vis_channels=1, hidden_channels=width,
                proj_channels=mlp_width, x_dim=1, t_scaling=1000).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50)
    
    ddpm = DiffusionModel(model, method, T=T_ddpm, device=device,
                          kernel_length=kernel_length, kernel_variance=kernel_variance,
                          beta_min=beta_min, beta_max=beta_max)
    ddpm.train(train_loader, optimizer, epochs=epochs, scheduler=scheduler,
               eval_int=0, save_int=epochs, generate=False, save_path=spath)
    
    sampleddpm[i, :, :] = ddpm.sample([n_x], n_samples=n_gen_samples).squeeze()

torch.save(sampleddpm, spath / 'samples_ddpm.pt')
print(f"Saved DDPM samples to {spath / 'samples_ddpm.pt'}")

# =============================================================================
# DDO (NCSN)
# =============================================================================

print("\n" + "="*60)
print("Training DDO (NCSN)")
print("="*60)

method = 'NCSN'
T_ncsn = 10
sigma1 = 1.0
sigmaT = 0.01
precondition = True

sampleddo = torch.zeros(N, n_gen_samples, n_x)

for i in range(N):
    print(f"\nSeed {i+1}/{N}")
    np.random.seed(random_seeds[i])
    torch.manual_seed(random_seeds[i])
    
    model = FNO(modes, vis_channels=1, hidden_channels=width,
                proj_channels=mlp_width, x_dim=1, t_scaling=1000).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50)
    
    ddo = DiffusionModel(model, method, T=T_ncsn, device=device,
                         kernel_length=kernel_length, kernel_variance=kernel_variance,
                         sigma1=sigma1, sigmaT=sigmaT, precondition=precondition)
    ddo.train(train_loader, optimizer, epochs=epochs, scheduler=scheduler,
              eval_int=0, save_int=epochs, generate=False, save_path=spath)
    
    sampleddo[i, :, :] = ddo.sample([n_x], n_samples=n_gen_samples).squeeze()

torch.save(sampleddo, spath / 'samples_ddo.pt')
print(f"Saved DDO samples to {spath / 'samples_ddo.pt'}")

# =============================================================================
# GANO
# =============================================================================

print("\n" + "="*60)
print("Training GANO")
print("="*60)

n_critic = 5
l_grad = 0.1

samplegano = torch.zeros(N, n_gen_samples, n_x)

for i in range(N):
    print(f"\nSeed {i+1}/{N}")
    np.random.seed(random_seeds[i])
    torch.manual_seed(random_seeds[i])
    
    D_model = FNO1dgano(modes, hidden_channels=width, proj_channels=mlp_width).to(device)
    G_model = FNO1dgano(modes, hidden_channels=width, proj_channels=mlp_width).to(device)
    
    optimizer_D = optim.Adam(D_model.parameters(), lr=lr)
    optimizer_G = optim.Adam(G_model.parameters(), lr=lr)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=50)
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=50)
    
    gano = GANO(G_model, D_model, device=device, l_grad=l_grad, n_critic=n_critic)
    gano.train(train_loader, epochs=epochs, D_optimizer=optimizer_D, G_optimizer=optimizer_G,
               D_scheduler=scheduler_D, G_scheduler=scheduler_G,
               eval_int=0, save_int=epochs, generate=False, save_path=spath)
    
    samplegano[i, :, :] = gano.sample([n_x], n_gen_samples).squeeze()

torch.save(samplegano, spath / 'samples_gano.pt')
print(f"Saved GANO samples to {spath / 'samples_gano.pt'}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*60)
print("All experiments complete!")
print("="*60)
print(f"Results saved to: {spath}")
print(f"  - samples_fmot.pt  (FFM-OT)")
print(f"  - samples_fmvp.pt  (FFM-VP)")
print(f"  - samples_ddpm.pt  (DDPM)")
print(f"  - samples_ddo.pt   (DDO/NCSN)")
print(f"  - samples_gano.pt  (GANO)")
print(f"  - dataset.pt       (Original data)")

