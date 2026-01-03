import torch
import numpy as np
import matplotlib.pyplot as plt
from util.util2 import MatReader


def make_grid(dims, x_min=0, x_max=1):
    """ Creates a 1D or 2D grid based on the list of dimensions in dims.

    Example: dims = [64, 64] returns a grid of shape (64*64, 2)
    Example: dims = [100] returns a grid of shape (100, 1)
    """
    if len(dims) == 1:
        grid = torch.linspace(x_min, x_max, dims[0])
        grid = grid.unsqueeze(-1)
    elif len(dims) == 2:
        _, _, grid = make_2d_grid(dims)
    return grid


def make_2d_grid(dims, x_min=0, x_max=1):
    # Makes a 2D grid in the format of (n_grid, 2)
    x1 = torch.linspace(x_min, x_max, dims[0])
    x2 = torch.linspace(x_min, x_max, dims[1])
    x1, x2 = torch.meshgrid(x1, x2, indexing='ij')
    grid = torch.cat((
        x1.contiguous().view(x1.numel(), 1),
        x2.contiguous().view(x2.numel(), 1)),
        dim=1)
    return x1, x2, grid

def reshape_for_batchwise(x, k):
        # need to do some ugly shape-hacking here to get appropriate number of dims
        # maps tensor (n,) to (n, 1, 1, ..., 1) where there are k 1's
        return x.view(-1, *[1]*k)

def reshape_channel_last(x):
    # maps a tensor (B, C, *dims) to (B, *dims, C)
    k = x.ndim
    idx = list(range(k))
    idx.append(idx.pop(1))
    return x.permute(idx)

def reshape_channel_first(x):
    # maps a tensor (B, *dims, C) to (B, C, *dims)
    k = x.ndim
    idx = list(range(k))
    idx.insert(1, idx.pop())
    return x.permute(idx)


def load_navier_stokes(path=None, shuffle=True, subsample_time=1):
    """Load Navier-Stokes data from .mat file.
    
    Args:
        path: Path to .mat file
        shuffle: Whether to shuffle the samples
        subsample_time: Subsample every n-th time step (default 1 = no subsampling)
        
    Returns:
        Tensor of shape (n_samples, 1, 64, 64) where each sample is a 2D snapshot
    """
    if not path:
        path = '../data/ns.mat'
    r = MatReader(path)
    r._load_file()
    u = r.read_field('sol')  # (batch_size, N, N, T)
    
    # Subsample time if needed
    if subsample_time > 1:
        u = u[..., ::subsample_time]
     
    u = u.permute(0, -1, 1, 2).reshape(-1, 64, 64).unsqueeze(1)
    
    # Ensure float32 for GPU efficiency
    u = u.float()
    
    if shuffle:
        idx = torch.randperm(u.shape[0])
        u = u[idx]
        
    return u


def load_stochastic_ns(path=None, shuffle=True, subsample_time=1):
    """Load stochastic Navier-Stokes data from .mat file.
    
    Data shape: (batch_size, N, N, T) = (10, 64, 64, 15001)
    
    Args:
        path: Path to .mat file
        shuffle: Whether to shuffle the samples
        subsample_time: Subsample every n-th time step (default 1 = no subsampling)
        
    Returns:
        Tensor of shape (n_samples, 1, 64, 64) where each sample is a 2D snapshot
    """
    if not path:
        path = '../data/stochastic_ns_64.mat'
    r = MatReader(path)
    r._load_file()
    u = r.read_field('sol')  # (batch_size=10, N=64, N=64, T=15001)
    
    # Subsample time if needed (reduces from 15001 to fewer time points)
    if subsample_time > 1:
        u = u[..., ::subsample_time]
    
    # Reshape: (batch, N, N, T) -> (batch, T, N, N) -> (batch*T, N, N) -> (batch*T, 1, N, N)
    u = u.permute(0, -1, 1, 2).reshape(-1, 64, 64).unsqueeze(1)
    
    # Ensure float32 for GPU efficiency
    u = u.float()
    
    if shuffle:
        idx = torch.randperm(u.shape[0])
        u = u[idx]
        
    return u


def load_stochastic_kdv(path=None, shuffle=True, mode='snapshot', subsample_time=1):
    """Load stochastic KdV data from .mat file.
    
    Data shape: (batch_size, N, T) = (1200, 128, 101)
    
    Args:
        path: Path to .mat file
        shuffle: Whether to shuffle the samples
        mode: 'snapshot' treats each (batch, time) as independent 1D sample
              'trajectory' keeps full time series per batch element
        subsample_time: Subsample every n-th time step
        
    Returns:
        If mode='snapshot': Tensor of shape (n_samples, 1, 128) 
        If mode='trajectory': Tensor of shape (1200, 101, 128) for sequence modeling
    """
    if not path:
        path = '../data/stochastic_kdv.mat'
    r = MatReader(path)
    r._load_file()
    u = r.read_field('sol')  # (batch_size=1200, N=128, T=101)
    
    if mode == 'snapshot':
        # Subsample time if needed
        if subsample_time > 1:
            u = u[..., ::subsample_time]
        # Treat each time slice as independent sample
        # (batch, N, T) -> (batch, T, N) -> (batch*T, N) -> (batch*T, 1, N)
        u = u.permute(0, 2, 1).reshape(-1, u.shape[1]).unsqueeze(1)
    elif mode == 'trajectory':
        # Keep as trajectories: (batch, N, T) -> (batch, T, N)
        if subsample_time > 1:
            u = u[..., ::subsample_time]
        u = u.permute(0, 2, 1)  # (batch, T, N)
    
    # Ensure float32 for GPU efficiency
    u = u.float()
    
    if shuffle:
        idx = torch.randperm(u.shape[0])
        u = u[idx]
        
    return u


def load_ginzburg_landau(path=None, shuffle=True, mode='snapshot', subsample_time=1):
    """Load stochastic Ginzburg-Landau data from .mat file.
    
    Data shape: (batch_size, N, T) = (1200, 129, 51)
    
    Args:
        path: Path to .mat file
        shuffle: Whether to shuffle the samples
        mode: 'snapshot' treats each (batch, time) as independent 1D sample
              'trajectory' keeps full time series per batch element
        subsample_time: Subsample every n-th time step
        
    Returns:
        If mode='snapshot': Tensor of shape (n_samples, 1, 129)
        If mode='trajectory': Tensor of shape (1200, 51, 129) for sequence modeling
    """
    if not path:
        path = '../data/stochastic_ginzburg_landau.mat'
    r = MatReader(path)
    r._load_file()
    u = r.read_field('sol')  # (batch_size=1200, N=129, T=51)
    
    if mode == 'snapshot':
        # Subsample time if needed
        if subsample_time > 1:
            u = u[..., ::subsample_time]
        # Treat each time slice as independent sample
        # (batch, N, T) -> (batch, T, N) -> (batch*T, N) -> (batch*T, 1, N)
        u = u.permute(0, 2, 1).reshape(-1, u.shape[1]).unsqueeze(1)
    elif mode == 'trajectory':
        # Keep as trajectories: (batch, N, T) -> (batch, T, N)
        if subsample_time > 1:
            u = u[..., ::subsample_time]
        u = u.permute(0, 2, 1)  # (batch, T, N)
    
    # Ensure float32 for GPU efficiency
    u = u.float()
    
    if shuffle:
        idx = torch.randperm(u.shape[0])
        u = u[idx]
        
    return u


def load_kdv(path=None, mode='snapshot'):
    """Load KdV data from .mat file.
    
    Note: This is a SINGLE trajectory, not batched data!
    Data shape: (N, T) = (512, 201) - one trajectory with 512 spatial points and 201 time steps
    
    Args:
        path: Path to .mat file
        mode: 'snapshot' treats each time step as independent 1D sample (201 samples of size 512)
              'sliding_window' uses overlapping windows for more samples
        
    Returns:
        If mode='snapshot': Tensor of shape (201, 1, 512) - each time slice as a sample
    """
    if not path:
        path = '../data/KdV.mat'
    r = MatReader(path)
    r._load_file()
    u = r.read_field('uu')  # (N=512, T=201)
    
    if mode == 'snapshot':
        # Transpose to (T, N) and add channel dimension -> (T, 1, N)
        u = u.T.unsqueeze(1)  # (201, 1, 512)
    
    # Ensure float32 for GPU efficiency
    u = u.float()
    
    # Note: KdV has only 201 samples from a single trajectory, no shuffling by default
    return u


def plot_loss_curve(tr_loss, save_path, te_loss=None, te_epochs=None, logscale=True):
    fig, ax = plt.subplots()

    if logscale:
        ax.semilogy(tr_loss, label='tr')
    else:
        ax.plot(tr_loss, label='tr')
    if te_loss is not None:
        te_epochs = np.asarray(te_epochs)
        if logscale:
            ax.semilogy(te_epochs-1, te_loss, label='te')  # assume te_epochs is 1-indexed
        else:
            ax.plot(te_epochs-1, te_loss, label='te')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc='upper right')

    plt.savefig(save_path)
    plt.close(fig)


def plot_samples(samples, save_path):
    n = samples.shape[0]
    sqrt_n = int(np.sqrt(n))

    fig, axs = plt.subplots(sqrt_n, sqrt_n, figsize=(8,8))

    samples = samples.permute(0, 2, 3, 1)  # (b, c, h, w) --> (b, h, w, c)
    samples = samples.detach().cpu()

    for i in range(n):
        j, k = i//sqrt_n, i%sqrt_n
        
        axs[j, k].imshow(samples[i])
        
        axs[j, k].set_xticks([])
        axs[j, k].set_yticks([])

    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.savefig(save_path)
    plt.close(fig)


def sample_many(wrapper, n_samples, dims, n_channels=1, batch_size=500, save_path=None):
    n_batches = n_samples // batch_size
    n_samples = n_batches * batch_size
    print(f'Generating {n_samples} samples')


    samples = []
    generated = 0
    while generated < n_samples:
        print(f'... generated {generated}/{n_samples}')
        try:
            sample = wrapper.sample(dims, n_samples=batch_size, n_channels=n_channels)
            samples.append(sample.detach().cpu())
            del sample
            torch.cuda.empty_cache()
            generated += batch_size
        except:
            print('NaN, retry')

    samples = torch.stack(samples)

    if save_path:
        torch.save(samples, save_path)
    return samples