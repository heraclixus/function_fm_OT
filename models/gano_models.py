import torch
import numpy as np
import pylab as plt
import torch.nn.functional as F
import torch.nn as nn

from util.config import load_config
from util.gaussian_process import GPPrior
from util.util import make_grid, reshape_channel_last

import time

def kernel(in_chan=2, up_dim=32):
    """
        Kernel network apply on grid
    """
    layers = nn.Sequential(
                nn.Linear(in_chan, up_dim, bias=True), torch.nn.GELU(),
                nn.Linear(up_dim, up_dim, bias=True), torch.nn.GELU(),
                nn.Linear(up_dim, 1, bias=False)
            )
    return layers


def kernel_1d(in_chan=1, up_dim=32):
    """
        Kernel network apply on 1D grid
    """
    layers = nn.Sequential(
                nn.Linear(in_chan, up_dim, bias=True), torch.nn.GELU(),
                nn.Linear(up_dim, up_dim, bias=True), torch.nn.GELU(),
                nn.Linear(up_dim, 1, bias=False)
            )
    return layers


# =============================================================================
# 1D Spectral Convolution and Pointwise Operations
# =============================================================================

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        modes1: Number of Fourier modes to multiply (should be <= input_length // 2)
        """
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = (1 / (2 * in_channels)) ** (1.0 / 2.0)
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, out_dim=None):
        if out_dim is None:
            out_dim = x.shape[-1]
        batchsize = x.shape[0]
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x)
        
        # Determine how many modes we can actually use (limited by input FFT size)
        n_fft_modes = x_ft.shape[-1]
        actual_modes = min(self.modes1, n_fft_modes)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, out_dim // 2 + 1, 
                            dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :actual_modes] = self.compl_mul1d(
            x_ft[:, :, :actual_modes], 
            self.weights1[:, :, :actual_modes]
        )

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=out_dim)
        return x


class pointwise_op_1d(nn.Module):
    def __init__(self, in_channel, out_channel, dim1):
        super(pointwise_op_1d, self).__init__()
        self.conv = nn.Conv1d(int(in_channel), int(out_channel), 1)
        self.dim1 = int(dim1)

    def forward(self, x, dim1=None):
        if dim1 is None:
            dim1 = self.dim1
        x_out = self.conv(x)
        x_out = torch.nn.functional.interpolate(x_out, size=dim1, mode='linear', align_corners=True)
        return x_out


class Generator1D(nn.Module):
    def __init__(self, in_channels, out_channels, d_co_domain, pad=0, factor=3/4, base_modes=16):
        super(Generator1D, self).__init__()
        """
        1D Generator using Fourier Neural Operator architecture.
        
        Similar to 2D version but operates on 1D sequences.
        Uses adaptive modes that work with various input sizes.
        
        input: (batchsize, n_points, in_channels)
        output: (batchsize, n_points, out_channels)
        
        Args:
            base_modes: Base number of Fourier modes (will be scaled by layer)
        """
        self.in_channels = in_channels  # input channels (data + grid = 2 for 1D)
        self.out_channels = out_channels
        self.d_co_domain = d_co_domain
        self.factor = factor
        self.padding = pad

        self.fc0 = nn.Linear(self.in_channels, self.d_co_domain)

        # Spectral convolution layers - use small base modes that work with any input size
        # Modes should be small enough to work even with downsampled features
        m = base_modes
        self.conv0 = SpectralConv1d(self.d_co_domain, int(2 * factor * self.d_co_domain), m)
        self.conv1 = SpectralConv1d(int(2 * factor * self.d_co_domain), int(4 * factor * self.d_co_domain), m)
        self.conv2 = SpectralConv1d(int(4 * factor * self.d_co_domain), int(8 * factor * self.d_co_domain), m // 2)
        self.conv2_1 = SpectralConv1d(int(8 * factor * self.d_co_domain), int(16 * factor * self.d_co_domain), m // 4)
        self.conv2_9 = SpectralConv1d(int(16 * factor * self.d_co_domain), int(8 * factor * self.d_co_domain), m // 4)
        self.conv3 = SpectralConv1d(int(16 * factor * self.d_co_domain), int(4 * factor * self.d_co_domain), m // 2)
        self.conv4 = SpectralConv1d(int(8 * factor * self.d_co_domain), int(2 * factor * self.d_co_domain), m)
        self.conv5 = SpectralConv1d(int(4 * factor * self.d_co_domain), self.d_co_domain, m)

        # Pointwise operations - dimensions will be set dynamically
        self.w0 = pointwise_op_1d(self.d_co_domain, int(2 * factor * self.d_co_domain), 64)
        self.w1 = pointwise_op_1d(int(2 * factor * self.d_co_domain), int(4 * factor * self.d_co_domain), 32)
        self.w2 = pointwise_op_1d(int(4 * factor * self.d_co_domain), int(8 * factor * self.d_co_domain), 16)
        self.w2_1 = pointwise_op_1d(int(8 * factor * self.d_co_domain), int(16 * factor * self.d_co_domain), 8)
        self.w2_9 = pointwise_op_1d(int(16 * factor * self.d_co_domain), int(8 * factor * self.d_co_domain), 16)
        self.w3 = pointwise_op_1d(int(16 * factor * self.d_co_domain), int(4 * factor * self.d_co_domain), 32)
        self.w4 = pointwise_op_1d(int(8 * factor * self.d_co_domain), int(2 * factor * self.d_co_domain), 64)
        self.w5 = pointwise_op_1d(int(4 * factor * self.d_co_domain), self.d_co_domain, 128)

        self.fc1 = nn.Linear(2 * self.d_co_domain, 4 * self.d_co_domain)
        self.fc2 = nn.Linear(4 * self.d_co_domain, self.out_channels)

    def forward(self, x):
        # x: (batch, n_points, channels) - should be 3D
        # Handle case where input might have extra dimensions
        if x.ndim == 4:
            # Squeeze out singleton dimension if present
            if x.shape[1] == 1:
                x = x.squeeze(1)  # (B, 1, N, C) -> (B, N, C)
            elif x.shape[-1] == 1:
                x = x.squeeze(-1)  # (B, N, C, 1) -> (B, N, C)
        
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x_fc0 = self.fc0(x)
        x_fc0 = F.gelu(x_fc0)

        # Permute to (batch, channels, n_points) for conv operations
        x_fc0 = x_fc0.permute(0, 2, 1)

        if self.padding > 0:
            x_fc0 = F.pad(x_fc0, [0, self.padding])

        D1 = x_fc0.shape[-1]
        
        # Compute intermediate dimensions
        d_0 = int(D1 * self.factor)
        d_1 = max(D1 // 2, 4)
        d_2 = max(D1 // 4, 4)
        d_3 = max(D1 // 8, 4)

        # Encoder path
        x1_c0 = self.conv0(x_fc0, d_0)
        x2_c0 = self.w0(x_fc0, d_0)
        x_c0 = x1_c0 + x2_c0
        x_c0 = F.gelu(x_c0)

        x1_c1 = self.conv1(x_c0, d_1)
        x2_c1 = self.w1(x_c0, d_1)
        x_c1 = x1_c1 + x2_c1
        x_c1 = F.gelu(x_c1)

        x1_c2 = self.conv2(x_c1, d_2)
        x2_c2 = self.w2(x_c1, d_2)
        x_c2 = x1_c2 + x2_c2
        x_c2 = F.gelu(x_c2)

        x1_c2_1 = self.conv2_1(x_c2, d_3)
        x2_c2_1 = self.w2_1(x_c2, d_3)
        x_c2_1 = x1_c2_1 + x2_c2_1
        x_c2_1 = F.gelu(x_c2_1)

        # Decoder path with skip connections
        x1_c2_9 = self.conv2_9(x_c2_1, d_2)
        x2_c2_9 = self.w2_9(x_c2_1, d_2)
        x_c2_9 = x1_c2_9 + x2_c2_9
        x_c2_9 = F.gelu(x_c2_9)
        x_c2_9 = torch.cat([x_c2_9, x_c2], dim=1)

        x1_c3 = self.conv3(x_c2_9, d_1)
        x2_c3 = self.w3(x_c2_9, d_1)
        x_c3 = x1_c3 + x2_c3
        x_c3 = F.gelu(x_c3)
        x_c3 = torch.cat([x_c3, x_c1], dim=1)

        x1_c4 = self.conv4(x_c3, d_0)
        x2_c4 = self.w4(x_c3, d_0)
        x_c4 = x1_c4 + x2_c4
        x_c4 = F.gelu(x_c4)
        x_c4 = torch.cat([x_c4, x_c0], dim=1)

        x1_c5 = self.conv5(x_c4, D1)
        x2_c5 = self.w5(x_c4, D1)
        x_c5 = x1_c5 + x2_c5
        x_c5 = F.gelu(x_c5)

        x_c5 = torch.cat([x_c5, x_fc0], dim=1)
        
        if self.padding > 0:
            x_c5 = x_c5[..., :-self.padding]

        # Permute back to (batch, n_points, channels)
        x_c5 = x_c5.permute(0, 2, 1)

        x_fc1 = self.fc1(x_c5)
        x_fc1 = F.gelu(x_fc1)

        x_out = self.fc2(x_fc1)
        
        # Ensure output is 3D (batch, n_points, channels)
        while x_out.ndim > 3:
            x_out = x_out.squeeze(-1)

        return x_out

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx


class Discriminator1D(nn.Module):
    def __init__(self, in_channels, out_channels, d_co_domain, kernel_dim=16, pad=0, factor=3/4, base_modes=16):
        super(Discriminator1D, self).__init__()
        """
        1D Discriminator using Fourier Neural Operator architecture.
        
        Similar to 2D version but operates on 1D sequences.
        Outputs a scalar discriminator score per sample.
        Uses adaptive modes that work with various input sizes.
        
        input: (batchsize, n_points, in_channels)
        output: (batchsize, 1)
        
        Args:
            base_modes: Base number of Fourier modes (will be scaled by layer)
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_co_domain = d_co_domain
        self.factor = factor
        self.padding = pad
        self.kernel_dim = kernel_dim

        self.fc0 = nn.Linear(self.in_channels, self.d_co_domain)

        # Spectral convolution layers - use small base modes that work with any input size
        m = base_modes
        self.conv0 = SpectralConv1d(self.d_co_domain, int(2 * factor * self.d_co_domain), m)
        self.conv1 = SpectralConv1d(int(2 * factor * self.d_co_domain), int(4 * factor * self.d_co_domain), m)
        self.conv2 = SpectralConv1d(int(4 * factor * self.d_co_domain), int(8 * factor * self.d_co_domain), m // 2)
        self.conv2_1 = SpectralConv1d(int(8 * factor * self.d_co_domain), int(16 * factor * self.d_co_domain), m // 4)
        self.conv2_9 = SpectralConv1d(int(16 * factor * self.d_co_domain), int(8 * factor * self.d_co_domain), m // 4)
        self.conv3 = SpectralConv1d(int(16 * factor * self.d_co_domain), int(4 * factor * self.d_co_domain), m // 2)
        self.conv4 = SpectralConv1d(int(8 * factor * self.d_co_domain), int(2 * factor * self.d_co_domain), m)
        self.conv5 = SpectralConv1d(int(4 * factor * self.d_co_domain), self.d_co_domain, m)

        # Pointwise operations - dimensions will be set dynamically
        self.w0 = pointwise_op_1d(self.d_co_domain, int(2 * factor * self.d_co_domain), 64)
        self.w1 = pointwise_op_1d(int(2 * factor * self.d_co_domain), int(4 * factor * self.d_co_domain), 32)
        self.w2 = pointwise_op_1d(int(4 * factor * self.d_co_domain), int(8 * factor * self.d_co_domain), 16)
        self.w2_1 = pointwise_op_1d(int(8 * factor * self.d_co_domain), int(16 * factor * self.d_co_domain), 8)
        self.w2_9 = pointwise_op_1d(int(16 * factor * self.d_co_domain), int(8 * factor * self.d_co_domain), 16)
        self.w3 = pointwise_op_1d(int(16 * factor * self.d_co_domain), int(4 * factor * self.d_co_domain), 32)
        self.w4 = pointwise_op_1d(int(8 * factor * self.d_co_domain), int(2 * factor * self.d_co_domain), 64)
        self.w5 = pointwise_op_1d(int(4 * factor * self.d_co_domain), self.d_co_domain, 128)

        self.fc1 = nn.Linear(2 * self.d_co_domain, 4 * self.d_co_domain)
        self.fc2 = nn.Linear(4 * self.d_co_domain, self.out_channels)

        # Kernel for last functional operation (1D grid input)
        self.knet = kernel_1d(1, self.kernel_dim)

    def forward(self, x):
        # x: (batch, n_points, channels) - should be 3D
        # Handle case where input might have extra dimensions
        if x.ndim == 4:
            # Squeeze out singleton dimension if present
            if x.shape[1] == 1:
                x = x.squeeze(1)  # (B, 1, N, C) -> (B, N, C)
            elif x.shape[-1] == 1:
                x = x.squeeze(-1)  # (B, N, C, 1) -> (B, N, C)
        
        batch_size = x.shape[0]
        n_points = x.shape[1]
        
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x_fc0 = self.fc0(x)
        x_fc0 = F.gelu(x_fc0)

        # Permute to (batch, channels, n_points)
        x_fc0 = x_fc0.permute(0, 2, 1)

        if self.padding > 0:
            x_fc0 = F.pad(x_fc0, [0, self.padding])

        D1 = x_fc0.shape[-1]
        
        # Compute intermediate dimensions
        d_0 = int(D1 * self.factor)
        d_1 = max(D1 // 2, 4)
        d_2 = max(D1 // 4, 4)
        d_3 = max(D1 // 8, 4)

        # Encoder path
        x1_c0 = self.conv0(x_fc0, d_0)
        x2_c0 = self.w0(x_fc0, d_0)
        x_c0 = x1_c0 + x2_c0
        x_c0 = F.gelu(x_c0)

        x1_c1 = self.conv1(x_c0, d_1)
        x2_c1 = self.w1(x_c0, d_1)
        x_c1 = x1_c1 + x2_c1
        x_c1 = F.gelu(x_c1)

        x1_c2 = self.conv2(x_c1, d_2)
        x2_c2 = self.w2(x_c1, d_2)
        x_c2 = x1_c2 + x2_c2
        x_c2 = F.gelu(x_c2)

        x1_c2_1 = self.conv2_1(x_c2, d_3)
        x2_c2_1 = self.w2_1(x_c2, d_3)
        x_c2_1 = x1_c2_1 + x2_c2_1
        x_c2_1 = F.gelu(x_c2_1)

        # Decoder path with skip connections
        x1_c2_9 = self.conv2_9(x_c2_1, d_2)
        x2_c2_9 = self.w2_9(x_c2_1, d_2)
        x_c2_9 = x1_c2_9 + x2_c2_9
        x_c2_9 = F.gelu(x_c2_9)
        x_c2_9 = torch.cat([x_c2_9, x_c2], dim=1)

        x1_c3 = self.conv3(x_c2_9, d_1)
        x2_c3 = self.w3(x_c2_9, d_1)
        x_c3 = x1_c3 + x2_c3
        x_c3 = F.gelu(x_c3)
        x_c3 = torch.cat([x_c3, x_c1], dim=1)

        x1_c4 = self.conv4(x_c3, d_0)
        x2_c4 = self.w4(x_c3, d_0)
        x_c4 = x1_c4 + x2_c4
        x_c4 = F.gelu(x_c4)
        x_c4 = torch.cat([x_c4, x_c0], dim=1)

        x1_c5 = self.conv5(x_c4, D1)
        x2_c5 = self.w5(x_c4, D1)
        x_c5 = x1_c5 + x2_c5
        x_c5 = F.gelu(x_c5)

        x_c5 = torch.cat([x_c5, x_fc0], dim=1)

        if self.padding > 0:
            x_c5 = x_c5[..., :-self.padding]

        # Permute back
        x = x_c5.permute(0, 2, 1)

        x = self.fc1(x)
        x = F.gelu(x)

        x = self.fc2(x)

        # Apply kernel for functional integral
        kx = self.knet(grid)
        kx = kx.view(batch_size, -1, 1)
        x = x.view(batch_size, -1, 1)
        x = torch.einsum('bik,bik->bk', kx, x) / n_points

        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx


# =============================================================================
# 2D Spectral Convolution and Pointwise Operations (Original)
# =============================================================================
    
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, dim1, dim2,modes1 = None, modes2 = None):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim1 = dim1 #output dimensions
        self.dim2 = dim2
        if modes1 is not None:
            self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
            self.modes2 = modes2
        else:
            self.modes1 = dim1//2-1 #if not given take the highest number of modes can be taken
            self.modes2 = dim2//2 
        self.scale = (1 / (2*in_channels))**(1.0/2.0)
        self.weights1 = nn.Parameter(self.scale * (torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))
        self.weights2 = nn.Parameter(self.scale * (torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, dim1 = None,dim2 = None):
        if dim1 is not None:
            self.dim1 = dim1
            self.dim2 = dim2
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        
        # Determine how many modes we can actually use
        # Must be limited by BOTH input FFT size AND output FFT size
        n_fft_modes1_in = x_ft.shape[-2]
        n_fft_modes2_in = x_ft.shape[-1]
        n_fft_modes1_out = self.dim1
        n_fft_modes2_out = self.dim2 // 2 + 1
        
        # Limit modes by: weight modes, input modes, and output modes
        actual_modes1 = min(self.modes1, n_fft_modes1_in // 2, n_fft_modes1_out // 2)
        actual_modes2 = min(self.modes2, n_fft_modes2_in, n_fft_modes2_out)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, self.dim1, self.dim2//2 + 1, dtype=torch.cfloat, device=x.device)
        
        if actual_modes1 > 0 and actual_modes2 > 0:
            # Slice input FFT and weights to matching sizes - use contiguous to ensure proper slicing
            x_ft_pos = x_ft[:, :, :actual_modes1, :actual_modes2].contiguous()
            x_ft_neg = x_ft[:, :, -actual_modes1:, :actual_modes2].contiguous()
            w1 = self.weights1[:, :, :actual_modes1, :actual_modes2].contiguous()
            w2 = self.weights2[:, :, :actual_modes1, :actual_modes2].contiguous()
            
            # Compute Fourier mode multiplication
            result1 = torch.einsum("bixy,ioxy->boxy", x_ft_pos, w1)
            result2 = torch.einsum("bixy,ioxy->boxy", x_ft_neg, w2)
            
            out_ft[:, :, :actual_modes1, :actual_modes2] = result1
            out_ft[:, :, -actual_modes1:, :actual_modes2] = result2

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(self.dim1, self.dim2))
        return x


class pointwise_op(nn.Module):
    def __init__(self, in_channel, out_channel,dim1, dim2):
        super(pointwise_op,self).__init__()
        self.conv = nn.Conv2d(int(in_channel), int(out_channel), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)

    def forward(self,x, dim1 = None, dim2 = None):
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
        x_out = self.conv(x)
        x_out = torch.nn.functional.interpolate(x_out, size = (dim1, dim2),mode = 'bicubic',align_corners=True)
        return x_out

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, d_co_domain, pad = 0, factor = 3/4):
        super(Generator, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.in_channels = in_channels # input channel
        self.out_channels = out_channels

        self.d_co_domain = d_co_domain 
        self.factor = factor
        self.padding = pad  # pad the domain if input is non-periodic

        self.fc0 = nn.Linear(self.in_channels, self.d_co_domain) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.d_co_domain, 2*factor*self.d_co_domain, 48, 48, 24, 24)

        self.conv1 = SpectralConv2d(2*factor*self.d_co_domain, 4*factor*self.d_co_domain, 32, 32, 16,16)

        self.conv2 = SpectralConv2d(4*factor*self.d_co_domain, 8*factor*self.d_co_domain, 16, 16,8,8)
        
        self.conv2_1 = SpectralConv2d(8*factor*self.d_co_domain, 16*factor*self.d_co_domain, 8, 8,4,4)
        
        self.conv2_9 = SpectralConv2d(16*factor*self.d_co_domain, 8*factor*self.d_co_domain, 16, 16,4,4)
        

        self.conv3 = SpectralConv2d(16*factor*self.d_co_domain, 4*factor*self.d_co_domain, 32, 32,8,8)

        self.conv4 = SpectralConv2d(8*factor*self.d_co_domain, 2*factor*self.d_co_domain, 48, 48,16,16)

        self.conv5 = SpectralConv2d(4*factor*self.d_co_domain, self.d_co_domain, 64, 64,24,24) # will be reshaped

        self.w0 = pointwise_op(self.d_co_domain,2*factor*self.d_co_domain,48, 48) #
        
        self.w1 = pointwise_op(2*factor*self.d_co_domain, 4*factor*self.d_co_domain, 32, 32) #
        
        self.w2 = pointwise_op(4*factor*self.d_co_domain, 8*factor*self.d_co_domain, 16, 16) #
        
        self.w2_1 = pointwise_op(8*factor*self.d_co_domain, 16*factor*self.d_co_domain, 8, 8)
        
        self.w2_9 = pointwise_op(16*factor*self.d_co_domain, 8*factor*self.d_co_domain, 16, 16)
        
        self.w3 = pointwise_op(16*factor*self.d_co_domain, 4*factor*self.d_co_domain, 32, 32) #
        
        self.w4 = pointwise_op(8*factor*self.d_co_domain, 2*factor*self.d_co_domain, 48, 48)
        
        self.w5 = pointwise_op(4*factor*self.d_co_domain, self.d_co_domain, 64, 64) # will be reshaped

        self.fc1 = nn.Linear(2*self.d_co_domain, 4*self.d_co_domain)
        self.fc2 = nn.Linear(4*self.d_co_domain, self.out_channels)

    def forward(self, x):

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x_fc0 = self.fc0(x)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 3, 1, 2)
        
        
        x_fc0 = F.pad(x_fc0, [0,self.padding, 0,self.padding])
        
        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]
        

        x1_c0 = self.conv0(x_fc0,int(D1*self.factor),int(D2*self.factor))
        x2_c0 = self.w0(x_fc0,int(D1*self.factor),int(D2*self.factor))
        x_c0 = x1_c0 + x2_c0
        x_c0 = F.gelu(x_c0)
        #print(x.shape)

        x1_c1 = self.conv1(x_c0 ,D1//2,D2//2)
        x2_c1 = self.w1(x_c0 ,D1//2,D2//2)
        x_c1 = x1_c1 + x2_c1
        x_c1 = F.gelu(x_c1)
        #print(x.shape)

        x1_c2 = self.conv2(x_c1 ,D1//4,D2//4)
        x2_c2 = self.w2(x_c1 ,D1//4,D2//4)
        x_c2 = x1_c2 + x2_c2
        x_c2 = F.gelu(x_c2 )
        #print(x.shape)
        
        x1_c2_1 = self.conv2_1(x_c2,D1//8,D2//8)
        x2_c2_1 = self.w2_1(x_c2,D1//8,D2//8)
        x_c2_1 = x1_c2_1 + x2_c2_1
        x_c2_1 = F.gelu(x_c2_1)
        
        x1_c2_9 = self.conv2_9(x_c2_1,D1//4,D2//4)
        x2_c2_9 = self.w2_9(x_c2_1,D1//4,D2//4)
        x_c2_9 = x1_c2_9 + x2_c2_9
        x_c2_9 = F.gelu(x_c2_9)
        x_c2_9 = torch.cat([x_c2_9, x_c2], dim=1) 

        x1_c3 = self.conv3(x_c2_9,D1//2,D2//2)
        x2_c3 = self.w3(x_c2_9,D1//2,D2//2)
        x_c3 = x1_c3 + x2_c3
        x_c3 = F.gelu(x_c3)
        x_c3 = torch.cat([x_c3, x_c1], dim=1)

        x1_c4 = self.conv4(x_c3,int(D1*self.factor),int(D2*self.factor))
        x2_c4 = self.w4(x_c3,int(D1*self.factor),int(D2*self.factor))
        x_c4 = x1_c4 + x2_c4
        x_c4 = F.gelu(x_c4)
        x_c4 = torch.cat([x_c4, x_c0], dim=1)

        x1_c5 = self.conv5(x_c4,D1,D2)
        x2_c5 = self.w5(x_c4,D1,D2)
        x_c5 = x1_c5 + x2_c5
        x_c5 = F.gelu(x_c5)
        

        x_c5 = torch.cat([x_c5, x_fc0], dim=1)
        if self.padding!=0:
            x_c5 = x_c5[..., :-self.padding, :-self.padding]

        x_c5 = x_c5.permute(0, 2, 3, 1)
        
        x_fc1 = self.fc1(x_c5)
        x_fc1 = F.gelu(x_fc1)
        
        x_out = self.fc2(x_fc1)
        
        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)



class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, d_co_domain, kernel_dim=16, pad = 0, factor = 3/4):
        super(Discriminator, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.in_channels = in_channels # input channel
        self.out_channels = out_channels 
        self.d_co_domain = d_co_domain 
        self.factor = factor
        self.padding = pad  # pad the domain if input is non-periodic
        self.kernel_dim = kernel_dim
        self.fc0 = nn.Linear(self.in_channels, self.d_co_domain) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.d_co_domain, 2*factor*self.d_co_domain, 48, 48, 24, 24)

        self.conv1 = SpectralConv2d(2*factor*self.d_co_domain, 4*factor*self.d_co_domain, 32, 32, 16,16)

        self.conv2 = SpectralConv2d(4*factor*self.d_co_domain, 8*factor*self.d_co_domain, 16, 16,8,8)
        
        self.conv2_1 = SpectralConv2d(8*factor*self.d_co_domain, 16*factor*self.d_co_domain, 8, 8,4,4)
        
        self.conv2_9 = SpectralConv2d(16*factor*self.d_co_domain, 8*factor*self.d_co_domain, 16, 16,4,4)
        

        self.conv3 = SpectralConv2d(16*factor*self.d_co_domain, 4*factor*self.d_co_domain, 32, 32,8,8)

        self.conv4 = SpectralConv2d(8*factor*self.d_co_domain, 2*factor*self.d_co_domain, 48, 48,16,16)

        self.conv5 = SpectralConv2d(4*factor*self.d_co_domain, self.d_co_domain, 64, 64,24,24) # will be reshaped

        self.w0 = pointwise_op(self.d_co_domain,2*factor*self.d_co_domain,48, 48) #
        
        self.w1 = pointwise_op(2*factor*self.d_co_domain, 4*factor*self.d_co_domain, 32, 32) #
        
        self.w2 = pointwise_op(4*factor*self.d_co_domain, 8*factor*self.d_co_domain, 16, 16) #
        
        self.w2_1 = pointwise_op(8*factor*self.d_co_domain, 16*factor*self.d_co_domain, 8, 8)
        
        self.w2_9 = pointwise_op(16*factor*self.d_co_domain, 8*factor*self.d_co_domain, 16, 16)
        
        self.w3 = pointwise_op(16*factor*self.d_co_domain, 4*factor*self.d_co_domain, 32, 32) #
        
        self.w4 = pointwise_op(8*factor*self.d_co_domain, 2*factor*self.d_co_domain, 48, 48)
        
        self.w5 = pointwise_op(4*factor*self.d_co_domain, self.d_co_domain, 64, 64) # will be reshaped

        self.fc1 = nn.Linear(2*self.d_co_domain, 4*self.d_co_domain)
        self.fc2 = nn.Linear(4*self.d_co_domain, self.out_channels)
        
        # kernel for last functional operation

        self.knet = kernel(2, self.kernel_dim)


    def forward(self, x):
        # x has shape (batch_size, dim0, dim1, ..., channels)
        # Handle case where channel dimension is missing
        if x.ndim == 3:
            x = x.unsqueeze(-1)  # Add channel dimension: [B, H, W] -> [B, H, W, 1]
        
        batch_size = x.shape[0]
        dims = x.shape[1:-1]
        prod_dims = dims.numel()
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
         
        x_fc0 = self.fc0(x)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 3, 1, 2)
        
        
        x_fc0 = F.pad(x_fc0, [0,self.padding, 0,self.padding])
        
        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]
        

        x1_c0 = self.conv0(x_fc0,int(D1*self.factor),int(D2*self.factor))
        x2_c0 = self.w0(x_fc0,int(D1*self.factor),int(D2*self.factor))
        x_c0 = x1_c0 + x2_c0
        x_c0 = F.gelu(x_c0)
        #print(x.shape)

        x1_c1 = self.conv1(x_c0 ,D1//2,D2//2)
        x2_c1 = self.w1(x_c0 ,D1//2,D2//2)
        x_c1 = x1_c1 + x2_c1
        x_c1 = F.gelu(x_c1)
        #print(x.shape)

        x1_c2 = self.conv2(x_c1 ,D1//4,D2//4)
        x2_c2 = self.w2(x_c1 ,D1//4,D2//4)
        x_c2 = x1_c2 + x2_c2
        x_c2 = F.gelu(x_c2 )
        #print(x.shape)
        
        x1_c2_1 = self.conv2_1(x_c2,D1//8,D2//8)
        x2_c2_1 = self.w2_1(x_c2,D1//8,D2//8)
        x_c2_1 = x1_c2_1 + x2_c2_1
        x_c2_1 = F.gelu(x_c2_1)
        
        x1_c2_9 = self.conv2_9(x_c2_1,D1//4,D2//4)
        x2_c2_9 = self.w2_9(x_c2_1,D1//4,D2//4)
        x_c2_9 = x1_c2_9 + x2_c2_9
        x_c2_9 = F.gelu(x_c2_9)
        x_c2_9 = torch.cat([x_c2_9, x_c2], dim=1) 

        x1_c3 = self.conv3(x_c2_9,D1//2,D2//2)
        x2_c3 = self.w3(x_c2_9,D1//2,D2//2)
        x_c3 = x1_c3 + x2_c3
        x_c3 = F.gelu(x_c3)
        x_c3 = torch.cat([x_c3, x_c1], dim=1)

        x1_c4 = self.conv4(x_c3,int(D1*self.factor),int(D2*self.factor))
        x2_c4 = self.w4(x_c3,int(D1*self.factor),int(D2*self.factor))
        x_c4 = x1_c4 + x2_c4
        x_c4 = F.gelu(x_c4)
        x_c4 = torch.cat([x_c4, x_c0], dim=1)

        x1_c5 = self.conv5(x_c4,D1,D2)
        x2_c5 = self.w5(x_c4,D1,D2)
        x_c5 = x1_c5 + x2_c5
        x_c5 = F.gelu(x_c5)
        

        x_c5 = torch.cat([x_c5, x_fc0], dim=1)
        if self.padding!=0:
            x_c5 = x_c5[..., :-self.padding, :-self.padding]

        x = x_c5
        x = x.permute(0, 2, 3, 1)

        x = self.fc1(x)
        x = F.gelu(x)

        x = self.fc2(x)
        
        kx = self.knet(grid)
        kx = kx.view(batch_size,-1, 1)
        x = x.view(batch_size,-1, 1)
        x = torch.einsum('bik,bik->bk', kx, x)/(prod_dims)

        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
