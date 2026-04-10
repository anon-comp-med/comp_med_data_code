"""
Inference and metric helper functions
"""

import numpy as np
import math as math
import torch

# Get predicted landmarks
def get_model_landmarks(heatmap):
    max_coords = []
    for i in range(heatmap.shape[0]):
        max_index = np.argmax(heatmap[i])
        coords = np.unravel_index(max_index, heatmap[i].shape)
        max_coords.append(coords)

    max_coords = np.array(max_coords)

    # Flip to (x,y)
    return np.fliplr(max_coords)


# Get predicted landmarks
def get_model_landmarks_batch(heatmap):
    B, K, H, W = heatmap.shape
    # flatten spatial dims
    flat = heatmap.view(B * K, -1)   # (B*K, H*W)
    # argmax along spatial dimension
    idx = flat.argmax(dim=1)         # (B*K,), stays on GPU
    # convert flat idx -> (x, y)
    y = idx // W
    x = idx %  W
    coords = torch.stack((x, y), dim=1)  # (B*K, 2)
    return coords.view(B, K, 2)
       


# Convert output heatmap to one-hot at maximum
def output_to_gt_batch(batch_heatmaps):
    B, K, H, W = batch_heatmaps.shape
    flat = batch_heatmaps.view(B, K, -1)
    idx = flat.argmax(dim=2)
    out = torch.zeros_like(flat, dtype=torch.double, device=batch_heatmaps.device)
    out.scatter_(2, idx.unsqueeze(2), 1.0)
    return out.view(B, K, H, W)


# Calculate the radial error for each landmark
def get_radial_error(output_heat, truth_heat, meta, mm=True, scale=True):

    # Get landmarks
    output_lands = get_model_landmarks(output_heat)
    truth_lands = get_model_landmarks(truth_heat)

    # Convert to mm
    if mm:
        output_lands = output_lands * meta["mre_per_pixel"] 
        truth_lands = truth_lands * meta["mre_per_pixel"] 

    # Normalize to original space
    if scale:
        output_lands = output_lands * meta["scale_factor"]
        truth_lands = truth_lands *  meta["scale_factor"]

    return np.linalg.norm((output_lands - truth_lands), axis = 1)


def get_ere(heatmaps, meta, significant_pixel_cutoff=0.05, mm=True, scale=True):    

    N, H, W = heatmaps.shape

    max_vals_per_land = np.max(heatmaps, axis=(1,2), keepdims=True)
    normalized = heatmaps / max_vals_per_land

    # Apply mask
    normalized *= (normalized > significant_pixel_cutoff)

    # Normalize
    total_weights = np.sum(normalized, axis=(1, 2), keepdims=True)
    empty_mask = total_weights.squeeze() == 0
    total_weights[empty_mask] = 1  
    normalized /= total_weights  

    predictions = get_model_landmarks(heatmaps).astype(np.float64)

    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')  # (H, W)
    coords = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float64)  # (H*W, 2)

    # Flatten normalized heatmaps for dot product
    flat_weights = normalized.reshape(N, -1)  # (N, H*W)

    # Convert to mm
    if mm:
        coords *= meta["mre_per_pixel"] 
        predictions *= meta["mre_per_pixel"] 

    # Normalize to original space
    if scale:
        coords *= meta["scale_factor"]
        predictions *=  meta["scale_factor"]

    # Compute distances from all coordinates to each predicted landmark
    dists = np.linalg.norm(coords[None, :, :] - predictions[:, None, :], axis=2)  # (N, H*W)

    # Weighted sum of distances = ERE
    eres = np.sum(flat_weights * dists, axis=1)  # (N,)

    # Handle empty masks by setting ERE = inf
    eres[empty_mask] = float('inf')

    return eres.tolist()



def get_mode_prob(output):
    return np.amax(output, axis = (1,2))


def get_sdr(radial_errors, thresholds):
    successful_detection_rates = []
    for threshold in thresholds:
        sdr = 100 * np.sum(radial_errors < threshold) / len(radial_errors)
        successful_detection_rates.append(sdr)
    return successful_detection_rates