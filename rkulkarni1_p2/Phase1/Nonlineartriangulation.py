import numpy as np
import scipy.optimize as optimize

def ProjectionMatrix(R, C, K):
    """Compute the projection matrix given rotation, translation, and camera intrinsic."""
    C = np.reshape(C, (3, 1))        
    I = np.identity(3)
    P = K @ np.hstack((R, -R @ C))
    # P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return P

def NonLinearTriangulation(K, pts1, pts2, x3D, R1, C1, R2, C2):
    """
    Perform non-linear optimization to refine 3D points.
    """
    P1 = ProjectionMatrix(R1, C1, K) 
    P2 = ProjectionMatrix(R2, C2, K)
    
    if pts1.shape[0] != pts2.shape[0] or pts1.shape[0] != x3D.shape[0]:
        raise ValueError("Check point dimensions - level nlt")

    def optimizePoint(X0, pt1, pt2, P1, P2):
        """Optimize a single 3D point."""
        optimized = optimize.least_squares(fun=ReprojectionLoss, x0=X0, method="trf", args=(pt1, pt2, P1, P2))
        return optimized.x

    # Optimize each 3D point
    x3D_optimized = np.array([optimizePoint(x3D[i], pts1[i], pts2[i], P1, P2) for i in range(len(x3D))])
    return x3D_optimized

def ReprojectionLoss(X, pts1, pts2, P1, P2):
    """
    Calculate the reprojection error for both cameras.
    """
    # Convert X to homogeneous coordinates
    X_homo = np.append(X, 1)
    
    # Project X onto both image planes
    x1_proj = P1 @ X_homo
    x2_proj = P2 @ X_homo

    # Convert from homogeneous to Cartesian coordinates
    x1_proj /= x1_proj[2]
    x2_proj /= x2_proj[2]

    # Compute reprojection errors
    error1 = np.sum((pts1 - x1_proj[:2])**2)
    error2 = np.sum((pts2 - x2_proj[:2])**2)

    return error1 + error2