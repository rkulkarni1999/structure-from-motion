from tqdm.notebook import tqdm
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn 

from NeRFModel import NerfModel 
from loadDataset import device
from renderRays import render_rays

@torch.no_grad()
def test(hn, hf, dataset, chunk_size=10, img_idx=0, nb_bins=192, H=400, W=400):
    ray_origins = dataset[img_idx * H * W: (img_idx + 1) * H * W, :3]
    ray_directions = dataset[img_idx * H * W: (img_idx + 1) * H * W, 3:6]
    
    # # empty list for regenerated pixels
    # data = [] 
    
    # for ii in range(int(np.ceil(H / chunk_size))):
    #     # Getting a chunk of rays
    #     ray_origins_ = ray_origins[i * W * chunk_size: (ii + 1) * W * chunk_size].to(device)
    #     ray_directions_ = ray_directions[i * W * chunk_size: (ii + 1) * W * chunk_size].to(device)
    


def train(nerf_model, optimizer, scheduler, data_loader, device, hn=0, hf=1, epochs=int(1e5), nb_bins=192, H=400, W=400):
    
    training_loss = []
    
    for _ in tqdm(range(epochs)):
        for batch in data_loader:
            
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            ground_truth_pixel_values = batch[:, 6:].to(device)
            
            regenerated_pixel_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins) 
            loss = ((ground_truth_pixel_values - regenerated_pixel_values) ** 2).sum()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
        
        scheduler.step()
        
        for img_idx in range(200):
            test(hn, hf, testing_dataset, )
            
            
            

def main():
    # importing the datasets {TODO: Integrate the custom dataset}
    training_dataset = torch.from_numpy(np.load('./rkulkarni1_p2/Phase2/Data/training_data.pkl', allow_pickle=True))
    testing_dataset = torch.from_numpy(np.load('testing_data.pkl', allow_pickle=True))
    
    # Defining the model
    model = NerfModel(hidden_dim=256).to(device)
    
    # Defining the optimizer
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    # defining scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)
    
    # loading training data into the dataloader
    data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)
    
    # Train
    train()    

if __name__ == "__main__":
    main()