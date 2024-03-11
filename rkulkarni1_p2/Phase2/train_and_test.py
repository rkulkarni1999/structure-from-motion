import torch
import numpy as np
from torch.utils.data import DataLoader
import os
import json
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from renderRays import render_rays
from NeRFModel import NerfModel, device
from loadDataset import loadDataset

###################
# LOAD CHECKPOINTS
###################
def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

###################
# FUNCTION 2 TRAIN
###################
def train(nerf_model, optimizer, scheduler, data_loader, device='cpu', hn=0, hf=1, nb_epochs=int(1e5),nb_bins=192, H=400, W=400, checkpoint_dir="./rkulkarni1_p2/Phase2/checkpoints", data_flag="lego"):
    
    # Initialize TensorBoard SummaryWriter
    current_time = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f'runs/nerf_experiment_{data_flag}_{current_time}')
    
    training_loss = []
    
    total_iterations = 0
        
    os.makedirs(checkpoint_dir, exist_ok=True) # create directory if there exist none. 

    for epoch in tqdm(range(nb_epochs), desc="Epoch Progress"):
        
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(data_loader):
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            ground_truth_px_values = batch[:, 6:].to(device)
            
            regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins) 
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update total and epoch losses
            epoch_loss += loss.item()
            training_loss.append(loss.item())
            
            # Log loss for each iteration
            writer.add_scalar('Loss/Iteration', loss.item(), total_iterations)
            total_iterations += 1

        # Log average epoch loss
        epoch_loss_avg = epoch_loss / len(data_loader)
        writer.add_scalar('Loss/Epoch', epoch_loss_avg, epoch+1)
        scheduler.step()
        
        # Save checkpoint after every epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"nerf_model_{data_flag}_epoch_{epoch+1}.ckpt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': nerf_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': training_loss,
        }, checkpoint_path)
        
        print(f"Checkpoint saved at {checkpoint_path}")

           
    writer.close() # close tensorboard
    return training_loss

#########################################
# FUNCTION 2 TEST {Generate Novel Views}
#########################################
@torch.no_grad()
def test(hn, hf, dataset, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400, data_flag="lego"):
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]

    data = []   # list of regenerated pixel values
    for i in range(int(np.ceil(H / chunk_size))):   # iterate over chunks
        
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)        
        regenerated_px_values = render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(regenerated_px_values)
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)

    plt.figure()
    plt.imshow(img)
    plt.savefig(f'./rkulkarni1_p2/Phase2/outputs/novel_views_{data_flag}/img_{img_index}.png', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    
    # Check device 
    print(f"This model is being trained/tested on {device}")
    
    ##########################
    # CHOOSE Dataset and Mode 
    # 1. "lego" or "ship"
    # 2. True or False 
    ##########################
    data_flag = "ship" 
    train_model = True

    # Load datasets
    data_path = f"./rkulkarni1_p2/Phase2/Data/{data_flag}/"
    training_dataset = loadDataset(data_path=data_path, mode="train")    
    testing_dataset = loadDataset(data_path=data_path, mode="test")
         
    # Initialize model, optimizer and scheduler
    model = NerfModel(hidden_dim=256).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)
    
    ##############################
    # CHANGE FLAG TO CHANGE MODE
    ##############################
    checkpoint_path = f"./rkulkarni1_p2/Phase2/checkpoints/nerf_model_{data_flag}_epoch_1.ckpt"  
    
    if train_model:
        data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)
        train(model, model_optimizer, scheduler, data_loader, device=device, hn=2, hf=6, nb_bins=192, H=400, W=400,data_flag=data_flag)
    else:
        # Load the checkpoint
        load_checkpoint(checkpoint_path, model, model_optimizer, scheduler)
    
    # Run test loop after loading the checkpoint
    for img_index in range(200):  # Adjust the range based on your specific requirements
        test(hn=2, hf=6, dataset=testing_dataset, chunk_size=10, img_index=img_index, nb_bins=192, H=400, W=400, data_flag=data_flag)
    
    