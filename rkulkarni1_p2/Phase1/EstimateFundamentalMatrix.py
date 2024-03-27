import numpy as np

def normalise(uv):
    """
    Normalize a set of 2D points (uv) for computer vision algorithms.
    The points are translated so their centroid is at the origin and scaled so
    that their average distance from the origin is sqrt(2).
    """

    """
    https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/sdai30/index.html
    """

    # Compute centroid of the points
    centroid = np.mean(uv, axis=0)
    
    # Translate points to have centroid at the origin
    uv_centered = uv - centroid
    
    # Compute scale factor to ensure the average distance from the origin is sqrt(2)
    scale = np.sqrt(2 / np.mean(np.sum(uv_centered**2, axis=1)))
    
    # Construct the normalization matrix
    T_scale = np.diag([scale, scale, 1])
    T_trans = np.array([[1, 0, -centroid[0]], [0, 1, -centroid[1]], [0, 0, 1]])
    T = T_scale @ T_trans  # Matrix multiplication for transformation
    
    # Apply normalization transformation
    x_augmented = np.hstack((uv, np.ones((len(uv), 1))))  # Augment points with a column of ones
    uv_normalized = (T @ x_augmented.T).T
    
    return uv_normalized, T

def get_fundamental_matrix(pts1, pts2, normalized = True):
    
    x1, x2 = pts1, pts2
    n = x1.shape[0]
    if n > 7:
        if normalized:
            x1 , T1 = normalise(x1)
            x2 , T2 = normalise(x2)

        A = np.zeros((n,9))
        for i in range(n):
            x_1,y_1 = x1[i][0], x1[i][1]
            x_2,y_2 = x2[i][0], x2[i][1]
            A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])

        U,S,VT = np.linalg.svd(A, full_matrices= True)
        F = VT.T[:,-1]
        F = F.reshape(3,3)

        # Due to Noise F can be full rank i.e 3, 
        # but we need to make it rank 2 by assigning zero to last diagonal element and thus we get the epipoles

        u, s, vt = np.linalg.svd(F)
        s = np.diag(s)
        s[2,2] = 0                     
        F = np.dot(u, np.dot(s, vt))

        #This is given in algorithm for normalization

        if normalise:
            F = np.dot(T2.T, np.dot(F, T1))   
            F = F / F[2,2]

        return F

    else:
        return None