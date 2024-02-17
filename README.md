# Buildings Built in Minutes - Structure from Motion

## 1. Introduction

Structure from Motion (SfM) is a technique for reconstructing a 3D scene and simultaneously obtaining the camera poses with respect to the scene using a series of images from different viewpoints. This process was famously utilized in projects like "Building Rome in a Day" by Agarwal et al., and is also the principle behind Microsoft Photosynth. Open-source algorithms such as VisualSFM make this technology accessible for experimentation.

## 2. Dataset for Classical SfM

For our classical SfM algorithm demonstration, we use a dataset comprising 5 images of Unity Hall at Worcester Polytechnic Institute, captured with a Samsung S22 Ultra camera. The images were resized to 800×600 pixels before camera calibration was performed using a Radial-Tangential model in MATLAB R2022a's Camera Calibrator Application.

## 4. Classical Approach to the SfM Problem

### 4.1. Feature Matching, Fundamental Matrix, and RANSAC

Using SIFT keypoints and descriptors, we can match features across images. The Fundamental matrix (*F*) plays a crucial role here, representing the epipolar geometry between two views. It's essential to refine these matches and reject outliers using RANSAC to improve the accuracy of *F* estimation.

### 4.2. Estimating Fundamental Matrix

The Fundamental matrix (*F*) is a 3×3 rank 2 matrix that relates corresponding points in two images from different views. Understanding epipolar geometry, which is the intrinsic projective geometry between two views, is crucial for grasping the concept of *F*.

#### 4.2.1. Epipolar Geometry

Epipolar geometry involves understanding how points in 3D space (captured in two images) relate to each other through their images points, denoted as **x** and **x'**. The key takeaway is that the search for corresponding points can be limited to the epipolar line in the other image, simplifying the matching process.

#### 4.2.2. The Fundamental Matrix (*F*)

*F* represents epipolar geometry algebraically. For *m* correspondences, the epipolar constraint **x'i^T F xi = 0** must be satisfied. Using the Eight-point algorithm, *F* can be estimated from at least 8 point correspondences between two images. This process involves setting up a homogeneous linear system and solving it using Singular Value Decomposition (SVD) to enforce the rank 2 constraint on *F*.

#### 4.2.3. Match Outlier Rejection via RANSAC

To handle noisy data and outliers from feature matching, RANSAC is employed to select the model of *F* with the maximum number of inliers, improving the robustness of the fundamental matrix estimation.

### 4.3. Estimate Essential Matrix from Fundamental Matrix

The Essential Matrix (*E*) relates corresponding points assuming pinhole camera models. It can be derived from *F* and the camera calibration matrix (*K*), with *E* = *K^T F K*. Correcting *E* to have singular values (1,1,0) ensures it adheres to its constraint of having 5 degrees of freedom.

### 4.4. Estimate Camera Pose from Essential Matrix

Four possible camera pose configurations can be derived from *E*. The correct configuration ensures the reconstructed 3D points are in front of both cameras, determined by the cheirality condition.

### 4.5. Triangulation Check for Cheirality Condition

The correct camera pose is identified by triangulating 3D points and ensuring they satisfy the cheirality condition, which confirms that the points are in front of the camera. This step is crucial for resolving the ambiguity in camera poses obtained from the essential matrix.
