import numpy as np
from LinerPnP import LinerPnP

def ProjectionMatrix(R, C, K):
    """Compute the camera projection matrix from rotation matrix R, translation vector C, and intrinsic matrix K."""
    C = C.reshape((3, 1))  # Ensure C is a column vector.
    P = K @ (R @ np.hstack((np.eye(3), -C)))  # Simplified matrix multiplication.
    return P

def PnPError(feature, X, R, C, K):
    """Calculate the reprojection error for a single 3D point and its corresponding 2D feature."""
    X_homo = np.hstack((X, 1)).reshape((4, 1))  # Directly create homogeneous coordinate.
    P = ProjectionMatrix(R, C, K)
    x_proj = P @ X_homo
    x_proj /= x_proj[2]  # Normalize to convert from homogeneous to Cartesian coordinates.

    u, v = feature  # 2D feature coordinates.
    error = np.linalg.norm([u, v] - x_proj[:2].ravel())  # Compute the L2 norm as error.

    return error

def PnPRANSAC(K, features, x3D, iter=1000, thresh=350):
    """Robustly estimate camera pose using the RANSAC algorithm."""
    inliers_thresh = 0
    R_best, C_best = None, None
    n_rows = x3D.shape[0]

    for _ in range(iter):
        # Randomly select 6 points.
        rand_indices = np.random.choice(n_rows, size=6, replace=False)
        X_set, x_set = x3D[rand_indices], features[rand_indices]

        # Get R and C from PnP function.
        R, C = LinerPnP(X_set, x_set, K)

        inliers = []
        if R is not None:
            for j in range(n_rows):
                error = PnPError(features[j], x3D[j], R, C, K)
                if error < thresh:
                    inliers.append(j)

        if len(inliers) > inliers_thresh:
            inliers_thresh = len(inliers)
            R_best, C_best = R, C

    return R_best, C_best