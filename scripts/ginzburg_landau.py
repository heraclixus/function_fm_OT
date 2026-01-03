"""
FFM Experiments on Stochastic Ginzburg-Landau Data.

This script runs baseline FFM experiments on stochastic Ginzburg-Landau
equation solutions (1D spatial), comparing:
- FFM-OT (OT-like interpolation path)
- FFM-VP (variance-preserving path)
- DDPM
- DDO (NCSN)
- GANO

Data shape: (1200, 129, 51) = (batch_size, N_spatial, T_time)

Usage:
    python ginzburg_landau.py
    python ginzburg_landau.py --epochs 100
"""

import sys
sys.path.append('../')

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from util.util import load_ginzburg_landau

from functional_fm import FFMModel
from diffusion import DiffusionModel
from gano1d import GANO
from models.fno import FNO, FNO1dgano

# =============================================================================
# Configuration
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser('ginzburg_landau_experiment')
    
    # Data params
    parser.add_argument('--dpath', help='Path to dataset', type=str,
                        default='../data/stochastic_ginzburg_landau.mat')
    parser.add_argument('--spath', help='Path to save outputs', type=str,
                        default='../outputs/ginzburg_landau_baseline/')
    parser.add_argument('--mode', help='Data mode: snapshot or trajectory', type=str,
                        default='snapshot')
    parser.add_argument('--subsample_time', help='Subsample time steps', type=int, default=1)
    
    # Training params
    parser.add_argument('--ntr', help='Number of training samples', type=int, default=40000)
    parser.add_argument('--nte', help='Number of test samples', type=int, default=10000)
    parser.add_argument('--bs', help='Batch size', type=int, default=512)
    parser.add_argument('--epochs', help='Training epochs', type=int, default=100)
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
    
    # Model params
    parser.add_argument('--modes', help='FNO modes', type=int, default=32)
    parser.add_argument('--width', help='Hidden channels', type=int, default=256)
    parser.add_argument('--mlp_width', help='MLP width', type=int, default=128)
    
    # GP params
    parser.add_argument('--kernel_length', help='Kernel lengthscale', type=float, default=0.01)
    parser.add_argument('--kernel_var', help='Kernel variance', type=float, default=0.1)
    
    # Experiment params
    parser.add_argument('--n_seeds', help='Number of random seeds', type=int, default=5)
    parser.add_argument('--n_gen', help='Number of samples to generate', type=int, default=500)
    
    # Method selection
    parser.add_argument('--methods', help='Methods to run (comma-separated)', type=str,
                        default='FFM-OT,FFM-VP,DDPM,DDO,GANO')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Parse methods to run
    methods_to_run = [m.strip() for m in args.methods.split(',')]
    
    # Load data
    print("\nLoading stochastic Ginzburg-Landau data...")
    data = load_ginzburg_landau(args.dpath, shuffle=True, mode=args.mode,
                                subsample_time=args.subsample_time)
    print(f"Data shape: {data.shape}")
    
    # Split into train/test
    n_total = data.shape[0]
    args.ntr = min(args.ntr, int(0.8 * n_total))
    args.nte = min(args.nte, n_total - args.ntr)
    
    train_data = data[:args.ntr]
    test_data = data[args.ntr:args.ntr + args.nte]
    
    print(f"Training samples: {train_data.shape[0]}")
    print(f"Test samples: {test_data.shape[0]}")
    
    # Get spatial dimension
    n_x = train_data.shape[-1]  # Should be 129 for Ginzburg-Landau
    print(f"Spatial dimension: {n_x}")
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.bs, shuffle=False)
    
    # Output directory
    spath = Path(args.spath)
    spath.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args)
    config['n_x'] = n_x
    config['device'] = device
    torch.save(config, spath / 'config.pt')
    
    # Random seeds
    random_seeds = [2**i for i in range(args.n_seeds)]
    N = args.n_seeds
    n_gen_samples = args.n_gen
    
    # Common hyperparameters
    modes = args.modes
    width = args.width
    mlp_width = args.mlp_width
    kernel_length = args.kernel_length
    kernel_variance = args.kernel_var
    epochs = args.epochs
    lr = args.lr
    sigma_min = 1e-4
    
    # =========================================================================
    # FFM-OT (OT-like path)
    # =========================================================================
    
    if 'FFM-OT' in methods_to_run:
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
    
    # =========================================================================
    # FFM-VP (Variance-Preserving path)
    # =========================================================================
    
    if 'FFM-VP' in methods_to_run:
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
    
    # =========================================================================
    # DDPM
    # =========================================================================
    
    if 'DDPM' in methods_to_run:
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
    
    # =========================================================================
    # DDO (NCSN)
    # =========================================================================
    
    if 'DDO' in methods_to_run:
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
    
    # =========================================================================
    # GANO
    # =========================================================================
    
    if 'GANO' in methods_to_run:
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
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "="*60)
    print("All experiments complete!")
    print("="*60)
    print(f"Results saved to: {spath}")


if __name__ == '__main__':
    main()

