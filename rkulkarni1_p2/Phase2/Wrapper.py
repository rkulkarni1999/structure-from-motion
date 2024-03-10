################################
# IMPORTING LIBRARIES
################################
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


# Positional Encoding
def positional_encoding(x, L):
        
        out = [x]
        for jj in range(L):
            out.append(torch.sin(2**jj*x))
            out.append(torch.cos(2**jj*x))
        return torch.cat(out, dim=1)
    
# Convert to MiniBatches
def mini_batches(inputs, batch_size):
    return [inputs[i:i + batch_size] for i in range(0, inputs.shape[0], batch_size)]

# Volumetric Rendering
def volumetric_rendering(pred, time_intervals, ray_directions, ray_origins):
    
    rgb = torch.relu(pred[...,:3])
    sigma_a = torch.relu(pred[..., 3])
    
    delta_time_interval = time_intervals[..., 1:] - time_intervals[..., :-1]
    
    t1 = torch.Tensor([1e10]).expand(delta_time_interval[...,:1].shape).to(device)
    delta_time_interval = torch.cat([delta_time_interval, t1], -1) 

##################################
# Sanity Check for PixelsToRays()
##################################
# pose  = poses[0]
# image = images[0]
# height, width = image.shape[1], image.shape[2] 
# rotation, translation = pose[:3, :3], pose[:3, 3]
# PixelsToRays(focal, height, width, rotation, translation)

# def PixelToRay(camera_info, pose, pixelPosition, args):
#     """
#     Input:
#         camera_info: image width, height, camera matrix 
#         pose: camera pose in world frame
#         pixelPosition: pixel position in the image
#         args: get near and far range, sample rate ...
#     Outputs:
#         ray origin and direction
#     """
#     pass

# def generateBatch(images, poses, camera_info, args):
#     """
#     Input:
#         images: all images in dataset
#         poses: corresponding camera pose in world frame
#         camera_info: image width, height, camera matrix
#         args: get batch size related information
#     Outputs:
#         A set of rays
#     """
#     pass 

# def render(model, rays_origin, rays_direction, args):
#     """
#     Input:
#         model: NeRF model
#         rays_origin: origins of input rays
#         rays_direction: direction of input rays
#     Outputs:
#         rgb values of input rays
#     """
#     pass

# def loss(groundtruth, prediction):
#     pass

# def train(images, poses, camera_info, args):
#     pass 

# def test(images, poses, camera_info, args):
#     pass

# def main(args):
#     # load data
#     print("Loading data...")
#     images, poses, camera_info = loadDataset(args.data_path, args.mode)

#     if args.mode == 'train':
#         print("Start training")
#         train(images, poses, camera_info, args)
#     elif args.mode == 'test':
#         print("Start testing")
#         args.load_checkpoint = True
#         test(images, poses, camera_info, args)

# def configParser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path',default="./Phase2/Data/",help="dataset path")
#     parser.add_argument('--mode',default='train',help="train/test/val")
#     parser.add_argument('--lrate',default=5e-4,help="training learning rate")
#     parser.add_argument('--n_pos_freq',default=10,help="number of positional encoding frequencies for position")
#     parser.add_argument('--n_dirc_freq',default=4,help="number of positional encoding frequencies for viewing direction")
#     parser.add_argument('--n_rays_batch',default=32*32*4,help="number of rays per batch")
#     parser.add_argument('--n_sample',default=400,help="number of sample per ray")
#     parser.add_argument('--max_iters',default=10000,help="number of max iterations for training")
#     parser.add_argument('--logs_path',default="./logs/",help="logs path")
#     parser.add_argument('--checkpoint_path',default="./Phase2/example_checkpoint/",help="checkpoints path")
#     parser.add_argument('--load_checkpoint',default=True,help="whether to load checkpoint or not")
#     parser.add_argument('--save_ckpt_iter',default=1000,help="num of iteration to save checkpoint")
#     parser.add_argument('--images_path', default="./image/",help="folder to store images")
#     return parser

# if __name__ == "__main__":
#     parser = configParser()
#     args = parser.parse_args()
#     main(args)