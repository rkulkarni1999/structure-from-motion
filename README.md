# 3D Reconstruction using Classical Structure from Motion (SfM). 

This project showcases the development of an advanced system for 3D scene reconstruction using Structure from Motion (SfM). The system processes a sequence of images to reconstruct a 3D model of a scene. 

## Methodology

Structure from Motion is a process that uses a series of 2D images to reconstruct a 3D scene. This phase involved the following steps:

- Feature Matching and Outlier Rejection with RANSAC
- Fundamental Matrix Estimation
- Essential Matrix Estimation from the Fundamental Matrix
- Camera Pose Estimation from the Essential Matrix
- Cheirality Condition Check with Triangulation
- Perspective-n-Point
- Bundle Adjustment

### Input

The dataset consists of 5 images of Unity Hall at WPI, captured using a Samsung S22 Ultra with the following settings: f/1.8 aperture, ISO 50, and 1/500 sec shutter speed. Four matching files (`matching*.txt`) detail feature correspondences between image pairs.

![Initial Images](./rkulkarni1_p2/Phase1/Data/Imgs.png)

### Initial Feature Matching

![Feature Matching](./rkulkarni1_p2/Phase1/Data/IntermediateOutputImages/beforeransac.png)

- RANSAC results showing inliers:

![Feature Matching](./rkulkarni1_p2/Phase1/Data/IntermediateOutputImages/afterransac.png)

- Cheirality check visualizing all possible camera poses:

![Cheirality Check](rkulkarni1_p2\Phase1\Data\IntermediateOutputImages\allpossible.png)

- Triangulation using the correct camera pose:

![Triangulation](rkulkarni1_p2\Phase1\Data\IntermediateOutputImages\Figure_1.png)

- Linear Triangulation vs Non-Linear Triangular Traingulation for Set 1 and Set 2. 

![Linear vs Non-Linear Triangulation](./rkulkarni1_p2/Phase1/Data/IntermediateOutputImages/beforeandafter_nonliner.png)

- Before and After Bundle Adjustment for sets 1 and 2. 

![Before and After Bundle Adjustment](rkulkarni1_p2\Phase1\Data\IntermediateOutputImages/beforeandafterbundle.png)

### Usage

To run the SfM pipeline:

```bash
python3 Wrapper.py
```

# Novel View Synthesis using Neural Radiance Fields (NeRF):
Implemented the original NERF method [from this paper](https://arxiv.org/abs/2003.08934).

### Input:
Download the lego data for NeRF from the original author’s link [here](https://drive.google.com/drive/folders/1lrDkQanWtTznf48FCaW5lX9ToRdNDF1a)

#### Sample input

<img src="./rkulkarni1_p2/Phase2/inputs/input.png"  align="center" alt="Undistorted" width="500"/>

### Neural Network used
<img src="./rkulkarni1_p2/Phase2/inputs/NETWROK.png"  align="center" alt="Undistorted" width="300"/>

### Training without Positional Encoding
<img src="rkulkarni1_p2\Phase2\outputs\training_loss_data\loss_vs_iteration_lego_3_epochs_without_encoding.png"  align="center" alt="Undistorted" width="400"/>

### Training with Positional Encoding
<img src="rkulkarni1_p2\Phase2\outputs\training_loss_data\loss_vs_iteration_lego_3_epochs.png"  align="center" alt="Undistorted" width="400"/>

### Result on Test without positional encoding
<img src="./rkulkarni1_p2/Phase2/NeRF_lego_no_p_enc.gif"  align="center" alt="Undistorted" width="500"/>

### Result on Test set with positional encoding
<img src="./rkulkarni1_p2/Phase2/NeRF_lego.gif"  align="center" alt="Undistorted" width="500"/>

<img src="./rkulkarni1_p2/Phase2/NeRF_ship.gif"  align="center" alt="Undistorted" width="500"/>

### Usage Guidelines:

#### Training:
1. Change the directory to Phase 2.
2. To train the NeRF model on GPU:

```
python3 Wrapper,py
```
3. Output of Loss plot will be saved in Results folder.

#### Testing
1. Change the flag in the ```Wrapper.py``` script and follow the training instructions. 

