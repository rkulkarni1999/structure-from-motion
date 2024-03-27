import numpy as np

def homo(pts):
    """Add a column of ones to pts to make them homogeneous coordinates."""
    return np.hstack((pts, np.ones((pts.shape[0], 1))))

def ProjectionMatrix(R, C, K):
    """Assuming this function is defined correctly to compute the projection matrix."""
    C = np.reshape(C, (3, 1))        
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return P # Replace with your actual implementation

def reprojectionErrorPnP(x3D, pts, K, R, C):
    """Compute the mean reprojection error."""
    """Compute the mean reprojection error using vectorized operations where possible."""
    # Compute the projection matrix from R, C, and K
    P = ProjectionMatrix(R, C, K)
    
    # Convert 3D points to homogeneous coordinates
    X_homo = homo(x3D)  # Assuming x3D is (N, 3)
    
    # Project 3D points to 2D using the camera projection matrix
    pts_proj_homo = (P @ X_homo.T).T  # Resulting shape should be (N, 3) in homogeneous coordinates
    
    # Convert projected points from homogeneous to Cartesian coordinates
    pts_proj = pts_proj_homo[:, :2] / pts_proj_homo[:, [2]]
    
    # Compute reprojection errors (squared differences) for each point
    errors = np.sum((pts - pts_proj)**2, axis=1)
    
    # Return the mean reprojection error
    return np.mean(errors)


def LinerPnP(X_set, x_set, K):
    """Solve the Perspective-n-Point problem."""
    X_4 = homo(X_set)
    x_3 = homo(x_set)

    # Normalize x
    K_inv = np.linalg.inv(K)
    x_n = (K_inv @ x_3.T).T

    A = []
    for X, x in zip(X_4, x_n):
        u, v, _ = x
        u_cross = np.array([[0, -1, v], [1, 0, -u], [-v, u, 0]])
        X_tilde = np.kron(np.eye(3), X.reshape(1, 4))
        A.append(u_cross @ X_tilde)

    A = np.concatenate(A, axis=0)
    _, _, VT = np.linalg.svd(A)
    P = VT[-1].reshape((3, 4))
    R, C = P[:, :3], P[:, 3]

    # Correct R to ensure it is a valid rotation matrix
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    C = -np.linalg.inv(R) @ C

    # Ensure right-handed coordinate system
    if np.linalg.det(R) < 0:
        R, C = -R, -C

    return R, C
