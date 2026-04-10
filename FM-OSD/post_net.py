"""
Global and local decoders
"""

from torch import nn
import torch

class GRN(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        # x: [B,C,H,W]
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)               # [B,C,1,1]
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)             # normalize across channels
        return x + self.gamma * (x * nx) + self.beta

class Upnet_v3(nn.Module):
    def __init__(self, size, in_channels, out_channels = 128, grn=False):
        super().__init__()
        self.size = size
        self.conv_out = nn.Conv2d(in_channels, out_channels, 3, padding = 1)

        self.grn_global = None
        if grn:
            self.grn_global = GRN(out_channels)

    def forward(self, x, num_patches):
        # x shape: (B, 1, T, D)
        x = x.squeeze(1) # x shape: (B, T, D)       
        b, n, c = x.shape         
        x = x.reshape(b, num_patches[0], num_patches[1], c) # x shape: (B, H_p, W_p, D)
        x = x.permute(0, 3, 1, 2) # x shape: (B, D, H_p, W_p)

        out = torch.nn.functional.interpolate(x, self.size, mode = 'bilinear') # x shape: (B, D, H, W)
        out = self.conv_out(out) # x shape: (B, out_channels, H, W)

        if self.grn_global is not None:
            out = self.grn_global(out)

        return out

class Upnet_v3_coarsetofine2_tran_new(nn.Module): 
    def __init__(self, size, in_channels, out_channels = 128, grn=False):
        super().__init__()
        self.size = size
        self.conv_out1 = nn.Conv2d(in_channels, out_channels, 3, padding = 1)  # global
        self.conv_out2 = nn.Conv2d(out_channels, out_channels, 5, padding = 2)  # local

        # Does upsampling
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(),
        )

        self.grn_local = None
        self.grn_global = None
        if grn:
            self.grn_local = GRN(out_channels)
            self.grn_global = GRN(out_channels)

    def forward(self, x, num_patches, size, islocal = False):
        # x shape: (B, 1, T, D)
        x = x.squeeze(1) # x shape: (B, T, D)
        b, n, c = x.shape
        x = x.reshape(b, num_patches[0], num_patches[1], c) # x shape: (B, H_p, W_p, D)
        x = x.permute(0, 3, 1, 2)  # x shape: (B, D, H_p, W_p)
        if islocal:
            x = self.up(x) # x shape: (B, out_channels, a, b)
            out = torch.nn.functional.interpolate(x, size, mode = 'bilinear') # x shape: (B, out_channels, H, W)
            out_fine = self.conv_out2(out) # x shape: (B, out_channels, H, W)
            if self.grn_local is not None:
                out_fine = self.grn_local(out_fine)
            return out_fine
        else: # Same as the previous forward
            out = torch.nn.functional.interpolate(x, size, mode = 'bilinear')
            out_coarse = self.conv_out1(out)
            if self.grn_global is not None:
                out_coarse = self.grn_global(out_coarse)
            return out_coarse  


