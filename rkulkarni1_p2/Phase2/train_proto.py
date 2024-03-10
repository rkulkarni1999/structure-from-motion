import numpy as np 
import torch 
import cv2
import torch
import torch.nn as nn
import json
import os
import math
from torch.utils.data import DataLoader

from loadDataset import device
from NeRFModel import NerfModel 
from renderRays import render_rays
###################################################################
# SANITY CHECK FOR THE PICKLE DATASET
##################################################################
hn = 0 
hf = 1
nb_bins = 192
H = 400
W = 400
epochs = int(1e5)

training_dataset = torch.from_numpy(np.load('./rkulkarni1_p2/Phase2/Data/training_data.pkl', allow_pickle=True))
testing_dataset = torch.from_numpy(np.load('./rkulkarni1_p2/Phase2/Data/testing_data.pkl', allow_pickle=True))
# print(training_dataset.shape)
print(testing_dataset.shape)
# model = NerfModel(hidden_dim=256).to(device)
# model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)
# data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)

# training_loss = [] 
# # Iterate through the DataLoader
# for batch in data_loader:
    
#     ray_origins = batch[:,:3].to(device)
#     print(f"Ray Origins : {ray_origins}")
    
#     ray_directions = batch[:,3:6].to(device)
#     print(f"Ray Directions : {ray_directions}")
    
#     ground_truth_pixel_values = batch[:,6:].to(device)
#     print(f"Ground Pixel Values : {ground_truth_pixel_values}")
    
#     regenerated_pixel_values = render_rays(model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
#     print(f"Regenerated Pixel Values : {regenerated_pixel_values}")
    
#     loss = ((ground_truth_pixel_values - regenerated_pixel_values) ** 2).sum()
#     print(f"loss : {loss}")
    
#     model_optimizer.zero_grad()
#     loss.backward()
#     model_optimizer.step()
#     training_loss.append(loss.item())
#     print(f"Training Losses : {training_loss}")
    
#     break
    


# ###################################################################
# # Dataset Computed using the {Direction, Origin and RGB Values}
# ################################ ##################################
# def decompose_transform_matrix(transform):
#     rotation = transform[:3, :3]
#     translation = transform[:3, 3]
#     return rotation, translation

# def loadDataset(data_path, mode):
#     if mode in ["train", "val", "test"]:
#         json_file = f"transforms_{mode}.json"
#     else:
#         raise ValueError("Mode must be 'train', 'val', or 'test'.")
    
#     transforms_path = os.path.join(data_path, json_file)
    
#     with open(transforms_path) as file:
#         data = json.load(file)
        
#     images     = []
#     transforms = []
    
#     for frame in data["frames"]:
#         img_file = os.path.join(data_path, frame["file_path"] + ".png")
#         image = cv2.imread(img_file)  # {REMEMBER} : Image is read in BGR 
#         image = cv2.resize(image, (400, 400), interpolation=cv2.INTER_AREA)
#         images.append(torch.tensor(image).permute(2, 0, 1).to(device))
        
#         transform = torch.tensor(frame["transform_matrix"]).to(device)
#         transforms.append(transform)
        
#     camera_angle_x = data["camera_angle_x"]
#     focal_length = 0.5 * images[0].shape[1] / math.tan(0.5 * camera_angle_x)
#     focal_length = torch.tensor(focal_length).to(device)
    
#     return focal_length, torch.stack(transforms), torch.stack(images)
#
# #############################
# # Convert Each Pixel to Rays
# #############################
# def PixelsToRays(focal, height, width, rotation, translation):
#     """
#     Function : Computes the ray direction and origin for all pixels in an image. 
#     Input:
#         camera_info: image width, height, focal length {constant throughout the network} 
#         pose: rotation and translation. Camera pose in world frame
#         args: get near and far range, sample rate ...{TODO : Implemented in the Next Function}
#     Outputs:
#         ray origin and direction
#     """
#     # Seperating the Image into Pixel Intervals
#     x_intervals = torch.linspace(0, height-1, height).to(device)  
#     y_intervals = torch.linspace(0, width-1, width).to(device)
    
#     x,y = torch.meshgrid(x_intervals, y_intervals, indexing='xy')
    
#     # shifting center and normalize by focal length. 
#     x_norm = (x- (height*0.5))/focal
#     y_norm = (y- (width*0.5))/focal
    
#     # direction vectors for each pixel
#     directions = torch.stack([x_norm, - y_norm, -torch.ones_like(x)], dim = -1)
#     directions = directions[..., None,:]
#     directions = directions * rotation
#     ray_directions = torch.sum(directions, dim = -1)
#     # ray_directions = ray_directions/torch.linalg.norm(ray_directions, dim = -1, keepdims = True)
#     ray_origins = torch.broadcast_to(translation, ray_directions.shape)
    
#     return ray_directions, ray_origins

# # Load the dataset
# data_path = "./rkulkarni1_p2/Phase2/Data"  # Set this to the actual path of your dataset
# mode = "train"  # Could be "train", "val", or "test"
# focal_length, transforms, images = loadDataset(data_path, mode)

# # Initialize tensors to store all rays origins and directions
# # all_rays_origins = torch.empty((0, 3), device=device)
# # all_rays_directions = torch.empty((0, 3), device=device)
# all_rays = torch.empty((0, 6), device=device)

# for i in range(len(images)):
#     height, width = images[i].shape[1], images[i].shape[2]  # Assuming image tensor is in CxHxW format
#     rotation, translation = decompose_transform_matrix(transforms[i])

#     ray_directions, ray_origins = PixelsToRays(focal_length, height, width, rotation, translation)

#     # Reshape and concatenate origins and directions side by side for each ray
#     rays_combined = torch.cat((ray_origins.reshape(-1, 3), ray_directions.reshape(-1, 3)), dim=1)

#     # Stack them vertically
#     all_rays = torch.cat((all_rays, rays_combined), dim=0)
    

# print("Hello")
# print(all_rays[590:600,:])