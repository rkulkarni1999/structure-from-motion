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
from NeRFModel import *
import cv2 
import math

# Assigning GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setting Random Seed for reproducibility
np.random.seed(0)

# Making dataset out of data. 
def loadDataset(data_path, mode):
    """
    Input:
        data_path: dataset path
        mode: train or test
    Outputs:
        camera_info: image width, height, camera matrix 
        images: images
        pose: corresponding camera pose in world frame
    """
    if mode in ["train", "val", "test"]:
        json_file = f"transforms_{mode}.json"
    else:
        raise ValueError("Mode must be 'train', 'val', or 'test'.")
    
    transforms_path = os.path.join(data_path, json_file)
    
    with open(transforms_path) as file:
        data = json.load(file)
        
    images     = []
    transforms = []
    
    for frame in data["frames"]:
        img_file = os.path.join(data_path, frame["file_path"] + ".png")
        image = cv2.imread(img_file)  # {REMEMBER} : Image is read in BGR 
        image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_AREA)
        images.append(torch.tensor(image).permute(2, 0, 1).to(device))
        
        transform = torch.tensor(frame["transform_matrix"]).to(device)
        transforms.append(transform)
        
    camera_angle_x = data["camera_angle_x"]
    focal_length = 0.5 * images[0].shape[1] / math.tan(0.5 * camera_angle_x)
    focal_length = torch.tensor(focal_length).to(device)
    
    return focal_length, torch.stack(transforms), torch.stack(images)

# fl, mats, imgs = loadDataset(data_path= "./rkulkarni1_p2\Phase2\Data", mode="train") # {SANITTY CHECK FOR FUNCTION}


def PixelToRay(camera_info, pose, pixelPosition, args):
    """
    Input:
        camera_info: image width, height, camera matrix 
        pose: camera pose in world frame
        pixelPosition: pixel position in the image
        args: get near and far range, sample rate ...
    Outputs:
        ray origin and direction
    """
    pass

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