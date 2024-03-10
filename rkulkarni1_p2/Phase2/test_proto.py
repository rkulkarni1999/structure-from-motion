import numpy as np 
import torch 
import cv2
import torch
import torch.nn as nn
import json
import os
import math
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from loadDataset import device
from NeRFModel import NerfModel 
from renderRays import render_rays

model = NerfModel(hidden_dim=256).to(device)
testing_dataset = torch.from_numpy(np.load('./rkulkarni1_p2/Phase2/Data/testing_data.pkl', allow_pickle=True))
hn = 2 
hf = 6 


@torch.no_grad
def test(hn, hf, dataset, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400):
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]
    
    data = []
    for i in range(int(np.ceil(H / chunk_size))):   # iterate over chunks
        # Get chunk of rays
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)        
        regenerated_px_values = render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(regenerated_px_values)
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)
    
    plt.figure()
    plt.imshow(img)
    plt.savefig(f'novel_views/img_{img_index}.png', bbox_inches='tight')
    plt.close()

# test(hn, hf, testing_dataset)