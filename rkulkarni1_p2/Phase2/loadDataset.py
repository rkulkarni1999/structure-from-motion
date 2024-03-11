import cv2
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from kornia import create_meshgrid
import pickle
import numpy as np
from PIL import Image
from torchvision import transforms

from torch.utils.data import DataLoader
from NeRFModel import NerfModel

#################################################################################### 
# DATASET GENERATION FOR NeRF {TODO: Compare the whole tensor with Pickle Dataset}
####################################################################################
# Making dataset out of data.
def loadDataset(data_path, mode):
    # PARAMETERS
    NEAR = 2.0
    FAR = 6.0
    IMG_HEIGHT = 400
    IMG_WIDTH = 400
    BOUNDS = np.array([NEAR, FAR])

    # LOADING THE JSON FILE
    if mode in ["train", "val", "test"]:
        json_file = f"transforms_{mode}.json"
    else:
        raise ValueError("Mode must be 'train', 'val', or 'test'.")

    transforms_path = os.path.join(data_path, json_file)

    with open(transforms_path) as file:
        data = json.load(file)

    # COMPUTING THE FOCAL LENGTH
    camera_angle_x = data["camera_angle_x"]
    focal_length = 0.5 * 400 / np.tan(0.5 * camera_angle_x)

    # COMPUTING DIRECTIONS
    directions = get_ray_directions(IMG_HEIGHT, IMG_WIDTH, focal_length)

    # EMPTY LISTS FOR STORING OUTPUTS
    poses = []
    all_rays = []
    all_rgbs = []

    for frame in data["frames"]:
        pose = np.array(frame["transform_matrix"])[:3, :4]
        poses += [pose]
        # c2w = torch.FloatTensor(pose).to(device)
        c2w = torch.FloatTensor(pose)

        image_path = os.path.join(data_path, f"{frame['file_path']}.png")
        img = Image.open(image_path)
        img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.LANCZOS)
        transform = transforms.ToTensor()
        # img_tensor = transform(img).to(device)  # Converts to tensor and sends to device
        img_tensor = transform(img)

        img_tensor = img_tensor.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
        img_tensor= img_tensor[:, :3]*img_tensor[:, -1:] + (1-img_tensor[:, -1:]) # blend A to RGB
        all_rgbs += [img_tensor]

        # Handle RGBA images by blending A into RGB or just use RGB
        # if img_tensor.size(0) == 4:  # If image is RGBA
        #     img_tensor = img_tensor[:3, :, :] * img_tensor[3:4, :, :] + (1.0 - img_tensor[3:4, :, :])  # Blend A into RGB
        # img = img_tensor.view(-1, 3)  # Flatten to shape (H*W, 3)
        # all_rgbs.append(img)

        rays_o, rays_d = get_rays(directions, c2w)
        all_rays.append(torch.cat([rays_o, rays_d, NEAR * torch.ones_like(rays_o[:, :1]), FAR * torch.ones_like(rays_o[:, :1])], dim=1))

    # Concatenating lists of tensors into a single tensor
    all_rays = torch.cat(all_rays, dim=0)  # (N, 8), where N is the total number of rays across all images
    all_rays = all_rays[:, :6]
    all_rgbs = torch.cat(all_rgbs, dim=0)  # (N, 3), where N is the total number of pixels across all images

    dataset = concatenate_tensors(all_rays, all_rgbs)

    return dataset

def get_ray_directions(H, W, focal):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions

def get_rays(directions, c2w):
    # Ensure both tensors are on the same device
    # directions = directions.to(device)
    directions = directions
    # c2w = c2w.to(device)
    c2w = c2w
    
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = torch.matmul(directions.reshape(-1, 3), c2w[:, :3].T)  # (H*W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape)  # (H*W, 3)
    return rays_o, rays_d

def concatenate_tensors(tensor1, tensor2):
    return torch.cat((tensor1, tensor2), dim=1)

###################################
# EXAMPLE USAGE TO VIEW THE DATASET
###################################
# data_path = "./rkulkarni1_p2/Phase2/Data/lego"  # Update this to the actual path
# mode = "train"  # Can be "train", "val", or "test"

# custom_dataset = loadDataset(data_path, mode)
# pickle_dataset = torch.from_numpy(np.load('./rkulkarni1_p2/Phase2/Data/lego/training_data.pkl', allow_pickle=True))

# print(custom_dataset[7410:7420,:])
# print(pickle_dataset[7410:7420,:])

# a = custom_dataset[7410:7420,:]
# b = pickle_dataset[7410:7420,:]

# mask = (a == b)

# print(mask)

# close = torch.isclose(custom_dataset, pickle_dataset, atol=1e-4, rtol=1e-3)
# all_close = torch.all(close)
# print(all_close)

# # model = NerfModel(hidden_dim=256).to(device)
# # model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
# # scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)
# # data_loader = DataLoader(tensor_dataset, batch_size=1024, shuffle=True)

# # # Iterate through the DataLoader
# # for batch in data_loader:
# #     print(batch)
# #     break  


# print(all_rays[:10, :])
# print(all_rgbs[:10, :])
# print(tensor_dataset[:10, :])







