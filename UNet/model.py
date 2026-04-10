"""
UNet model code
"""

import torch

import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self, cfg_model, no_of_landmarks):
        super(Unet, self).__init__()
             
        self.unet = smp.Unet(
            encoder_name=cfg_model.ENCODER_NAME,
            encoder_weights=cfg_model.ENCODER_WEIGHTS,  
            decoder_channels=cfg_model.DECODER_CHANNELS,
            in_channels=cfg_model.IN_CHANNELS,
            classes=no_of_landmarks,
        )        
        
    def forward(self, x):
        return self.unet(x)
    

def two_d_softmax(x):
    x_stable = x - torch.amax(x, dim=(2, 3), keepdim=True)
    exp_y = torch.exp(x_stable)
    return exp_y / torch.sum(exp_y, dim=(2, 3), keepdim=True)


def nll_across_batch(output, target, mask=None):    

    nll = -target * torch.log(output.double()) #.clamp(min=1e-9)

    if mask is not None:
        mask = mask.to(dtype=nll.dtype, device=nll.device)
        nll = nll * mask

    return torch.mean(torch.sum(nll, dim=(2, 3)))
    
        
