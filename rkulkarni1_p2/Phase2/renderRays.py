import argparse
import glob
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
import imageio
import torch
import matplotlib.pyplot as plt
import os
import json
import cv2 
import math
import numpy as np
import random
import torch.nn.functional as F
from loadDataset import device
from NeRFModel import compute_accumulated_transmittance

# # Setting Random Seed for reproducibility
# # np.random.seed(0)

# ####################################
# # Normalizing an Image to [0,255]
# ####################################
# def normalize(image):                                                                                                                                                             
#     return cv2.normalize(image,dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

# ######################################################
# # COMPUTE RAY DIRECTIONS AND ORIGINS FOR AN IMAGE
# ######################################################
# def generate_ray_points(ray_directions, ray_origins, num_ray_samples, t_near, t_far, num_rand_rays=None, n_frequencies=4, rand=False):
    
#     time_intervals = torch.linspace(t_near, t_far, num_ray_samples)
#     noise_shape = list(ray_origins.shape[:-1]) + [num_ray_samples]
#     noise = torch.rand(size = noise_shape) * (t_far - t_near)/num_ray_samples
#     time_intervals = time_intervals + noise
#     time_intervals = time_intervals.to(device)
    
#     rays = ray_origins[..., None, :] + ray_directions[..., None, :]* time_intervals[...,:, None]  # (n_rays x n_samples x 3) 
    
#     return time_intervals, rays


###############################
# RENDER RAYS
############################### 

def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    
    # perturb sampling along each ray
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)

    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)
    
    # Compute 3D points along each ray
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)   # [batch_size, nb_bins, 3]
    # Expand the ray_directions tensor to match the shape of x
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1) 
    
    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])
    
    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    # Compute the pixel values as a weighted sum of colors along each ray
    c = (weights * colors).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background 
    return c + 1 - weight_sum.unsqueeze(-1)

