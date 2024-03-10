from tqdm.notebook import tqdm
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn 


def train(nerf_model, optimizer, scheduler, data_loader, device, hn=0, hf=1, epochs=int(1e5), nb_bins=192, H=400, W=400):
    
    training_loss = []
    for _ in tqdm(range(epochs)):
        for batch in data_loader
        