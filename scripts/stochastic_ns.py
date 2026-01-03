"""
FFM Experiments on Stochastic Navier-Stokes Data.

This script runs baseline FFM experiments on stochastic Navier-Stokes 
equation solutions (2D spatial), comparing:
- FFM-OT (OT-like interpolation path)
- FFM-VP (variance-preserving path)
- DDPM
- DDO (NCSN)
- GANO

Data shape: (10, 64, 64, 15001) = (batch_size, N, N, T_time)

Note: With 10 trajectories and 15001 time steps, we have 150,010 spatial snapshots.
We subsample in time to reduce this.

Usage:
    python stochastic_ns.py
    python stochastic_ns.py --subsample_time 100 --epochs 50
"""

import sys
sys.path.append('../')

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from torch.optim.lr_scheduler import StepLR

from models.fno import FNO
from models.gano_models import Generator, Discriminator
from gano import GANO
from functional_fm import FFMModel
from diffusion import DiffusionModel
from util.util import load_stochastic_ns, plot_samples

# =============================================================================
# Configuration
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser('stochastic_ns_experiment')
    
    # Data params
    parser.add_argument('--dpath', help='Path to dataset', type=str,
                        default='../data/stochastic_ns_64.mat')
    parser.add_argument('--spath', help='Path to save outputs', type=str,
                        default='../outputs/stochastic_ns_baseline/')
    parser.add_argument('--subsample_time', help='Subsample time steps (reduces 15001 samples)', 
                        type=int, default=5)  # 15001/5 * 10 trajectories ≈ 30,000 samples
    
    # Training params
    parser.add_argument('--ntr', help='Number of training samples', type=int, default=20000)
    parser.add_argument('--nte', help='Number of test samples', type=int, default=5000)
    parser.add_argument('--bs', help='Batch size', type=int, default=512)
    parser.add_argument('--epochs', help='Training epochs', type=int, default=100)
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
    
    # Model params (2D FNO)
    parser.add_argument('--modes', help='FNO modes', type=int, default=16)
    parser.add_argument('--hch', help='Hidden channels', type=int, default=32)
    parser.add_argument('--pch', help='Projection channels', type=int, default=64)
    parser.add_argument('--tscale', help='Time scaling', type=float, default=1000.0)
    
    # GP params
    parser.add_argument('--kernel_length', help='Kernel lengthscale', type=float, default=0.001)
    parser.add_argument('--kernel_var', help='Kernel variance', type=float, default=1.0)
    
    # Experiment params
    parser.add_argument('--n_seeds', help='Number of random seeds', type=int, default=3)
    parser.add_argument('--n_gen', help='Number of samples to generate', type=int, default=100)
    
    # Method selection
    parser.add_argument('--methods', help='Methods to run (comma-separated)', type=str,
                        default='FFM-OT,FFM-VP,DDPM,DDO,GANO')
    
    # GANO specific
    parser.add_argument('--lgrad', help='Gradient penalty for GANO', type=float, default=10)
    parser.add_argument('--ncritic', help='Critic updates per generator update', type=int, default=5)
    parser.add_argument('--pad', help='Padding for GANO', type=int, default=0)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Parse methods to run
    methods_to_run = [m.strip() for m in args.methods.split(',')]
    
    # Load data
    print("\nLoading stochastic Navier-Stokes data...")
    data = load_stochastic_ns(args.dpath, shuffle=True, subsample_time=args.subsample_time)
    print(f"Data shape: {data.shape}")  # (n_samples, 1, 64, 64)
    
    # Split into train/test
    n_total = data.shape[0]
    args.ntr = min(args.ntr, int(0.8 * n_total))
    args.nte = min(args.nte, n_total - args.ntr)
    
    train_data = data[:args.ntr]
    test_data = data[args.ntr:args.ntr + args.nte]
    
    print(f"Training samples: {train_data.shape[0]}")
    print(f"Test samples: {test_data.shape[0]}")
    
    # Get spatial dimensions
    n_channels = train_data.shape[1]  # 1
    spatial_dims = train_data.shape[2:]  # (64, 64)
    print(f"Spatial dimensions: {spatial_dims}")
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.bs, shuffle=False)
    
    # Output directory
    spath = Path(args.spath)
    spath.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args)
    config['spatial_dims'] = list(spatial_dims)
    config['device'] = device
    torch.save(config, spath / 'config.pt')
    
    # Random seeds
    random_seeds = [2**i for i in range(args.n_seeds)]
    N = args.n_seeds
    n_gen_samples = args.n_gen
    
    # Common hyperparameters
    modes = args.modes
    hch = args.hch
    pch = args.pch
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
        
        samplefmot = torch.zeros(N, n_gen_samples, *spatial_dims)
        
        for i in range(N):
            print(f"\nSeed {i+1}/{N}")
            np.random.seed(random_seeds[i])
            torch.manual_seed(random_seeds[i])
            
            model = FNO(modes, vis_channels=n_channels, hidden_channels=hch,
                        proj_channels=pch, x_dim=2, t_scaling=args.tscale).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
            
            fmot = FFMModel(model, kernel_length=kernel_length, kernel_variance=kernel_variance,
                            sigma_min=sigma_min, device=device)
            fmot.train(train_loader, optimizer, epochs=epochs, scheduler=scheduler,
                       test_loader=test_loader, eval_int=10, save_int=epochs,
                       generate=False, save_path=spath)
            
            samples = fmot.sample(list(spatial_dims), n_samples=n_gen_samples, 
                                  n_channels=n_channels).cpu().squeeze(1)
            samplefmot[i] = samples
            
            # Save some sample visualizations
            if i == 0:
                plot_samples(samples[:16].unsqueeze(1), spath / 'samples_fmot_viz.pdf')
        
        torch.save(samplefmot, spath / 'samples_fmot.pt')
        print(f"Saved FFM-OT samples to {spath / 'samples_fmot.pt'}")
    
    # =========================================================================
    # FFM-VP (Variance-Preserving path)
    # =========================================================================
    
    if 'FFM-VP' in methods_to_run:
        print("\n" + "="*60)
        print("Training FFM-VP")
        print("="*60)
        
        samplefmvp = torch.zeros(N, n_gen_samples, *spatial_dims)
        
        for i in range(N):
            print(f"\nSeed {i+1}/{N}")
            np.random.seed(random_seeds[i])
            torch.manual_seed(random_seeds[i])
            
            model = FNO(modes, vis_channels=n_channels, hidden_channels=hch,
                        proj_channels=pch, x_dim=2, t_scaling=args.tscale).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
            
            fmvp = FFMModel(model, kernel_length=kernel_length, kernel_variance=kernel_variance,
                            sigma_min=sigma_min, device=device, vp=True)
            fmvp.train(train_loader, optimizer, epochs=epochs, scheduler=scheduler,
                       test_loader=test_loader, eval_int=10, save_int=epochs,
                       generate=False, save_path=spath)
            
            samples = fmvp.sample(list(spatial_dims), n_samples=n_gen_samples,
                                  n_channels=n_channels).cpu().squeeze(1)
            samplefmvp[i] = samples
        
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
        
        sampleddpm = torch.zeros(N, n_gen_samples, *spatial_dims)
        
        for i in range(N):
            print(f"\nSeed {i+1}/{N}")
            np.random.seed(random_seeds[i])
            torch.manual_seed(random_seeds[i])
            
            model = FNO(modes, vis_channels=n_channels, hidden_channels=hch,
                        proj_channels=pch, x_dim=2, t_scaling=args.tscale).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
            
            ddpm = DiffusionModel(model, method, T=T_ddpm, device=device,
                                  kernel_length=kernel_length, kernel_variance=kernel_variance,
                                  beta_min=beta_min, beta_max=beta_max)
            ddpm.train(train_loader, optimizer, epochs=epochs, scheduler=scheduler,
                       test_loader=test_loader, eval_int=10, save_int=epochs,
                       generate=False, save_path=spath)
            
            samples = ddpm.sample(list(spatial_dims), n_samples=n_gen_samples,
                                  n_channels=n_channels).squeeze(1)
            sampleddpm[i] = samples
        
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
        sigma1 = 100.0  # Larger for 2D data
        sigmaT = 0.01
        precondition = True
        
        sampleddo = torch.zeros(N, n_gen_samples, *spatial_dims)
        
        for i in range(N):
            print(f"\nSeed {i+1}/{N}")
            np.random.seed(random_seeds[i])
            torch.manual_seed(random_seeds[i])
            
            model = FNO(modes, vis_channels=n_channels, hidden_channels=hch,
                        proj_channels=pch, x_dim=2, t_scaling=args.tscale).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
            
            ddo = DiffusionModel(model, method, T=T_ncsn, device=device,
                                 kernel_length=kernel_length, kernel_variance=kernel_variance,
                                 sigma1=sigma1, sigmaT=sigmaT, precondition=precondition)
            ddo.train(train_loader, optimizer, epochs=epochs, scheduler=scheduler,
                      test_loader=test_loader, eval_int=10, save_int=epochs,
                      generate=False, save_path=spath)
            
            samples = ddo.sample(list(spatial_dims), n_samples=n_gen_samples,
                                 n_channels=n_channels).squeeze(1)
            sampleddo[i] = samples
        
        torch.save(sampleddo, spath / 'samples_ddo.pt')
        print(f"Saved DDO samples to {spath / 'samples_ddo.pt'}")
    
    # =========================================================================
    # GANO (2D version)
    # =========================================================================
    
    if 'GANO' in methods_to_run:
        print("\n" + "="*60)
        print("Training GANO")
        print("="*60)
        
        in_channels = n_channels + 2  # visual channels + 2D spatial embedding
        out_channels = n_channels
        
        samplegano = torch.zeros(N, n_gen_samples, *spatial_dims)
        
        for i in range(N):
            print(f"\nSeed {i+1}/{N}")
            np.random.seed(random_seeds[i])
            torch.manual_seed(random_seeds[i])
            
            model_g = Generator(in_channels, out_channels, hch, pad=args.pad).to(device)
            model_d = Discriminator(in_channels, out_channels, hch, pad=args.pad).to(device)
            
            optimizer_g = optim.Adam(model_g.parameters(), lr=lr)
            optimizer_d = optim.Adam(model_d.parameters(), lr=lr)
            scheduler_g = StepLR(optimizer_g, step_size=25, gamma=0.1)
            scheduler_d = StepLR(optimizer_d, step_size=25, gamma=0.1)
            
            gano = GANO(model_d, model_g, args.lgrad, args.ncritic,
                        kernel_length=kernel_length, kernel_variance=kernel_variance,
                        device=device)
            gano.train(train_loader, optimizer_d, optimizer_g, epochs=epochs,
                       D_scheduler=scheduler_d, G_scheduler=scheduler_g,
                       test_loader=None, eval_int=0, save_int=epochs,
                       generate=False, save_path=spath)
            
            # Note: GANO sampling interface may differ
            try:
                samples = gano.sample(list(spatial_dims), n_gen_samples)
                if samples.dim() == 4:
                    samples = samples.squeeze(1)
                samplegano[i] = samples.cpu()
            except Exception as e:
                print(f"GANO sampling failed: {e}")
        
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

