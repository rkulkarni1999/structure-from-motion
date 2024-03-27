import numpy as np

def linearTriangulation(K, C1, R1, C2, R2, x1, x2):
    # Identity matrix for constructing camera matrices
    I = np.identity(3)
    
    # Reshape camera centers for proper matrix operations
    C1 = C1.reshape(3, 1)
    C2 = C2.reshape(3, 1)

    # Compute camera projection matrices for both camera views
    P1 = K @ R1 @ np.hstack((I, -C1))
    P2 = K @ R2 @ np.hstack((I, -C2))

    # Initialize list to hold 3D points
    X = []
    uv_X3D =[]
    # Convert x1 and x2 to homogeneous coordinates by adding a 1
    x1_homogeneous = np.hstack((x1, np.ones((x1.shape[0], 1))))
    x2_homogeneous = np.hstack((x2, np.ones((x2.shape[0], 1))))

    # Process each pair of points
    for point1, point2 in zip(x1_homogeneous, x2_homogeneous):
        # Construct matrix A as per the direct linear transform algorithm
        # A = np.vstack([
        #     (point1[1] * P1[2, :]) - P1[1, :],
        #     P1[0, :] - (point1[0] * P1[2, :]),
        #     (point2[1] * P2[2, :]) - P2[1, :],
        #     P2[0, :] - (point2[0] * P2[2, :])
        # ])
        A = np.vstack([
            (point1[1] * P1[2, :]) - P1[1, :],
            (point1[0] * P1[2, :]) - P1[0, :],
            (point2[1] * P2[2, :]) - P2[1, :],
            (point2[0] * P2[2, :]) - P2[0, :]
        ])

        # Solve for X using SVD
        _, _, V = np.linalg.svd(A)
        X_homogeneous = V[-1]  # Take the last row from V
        
        # Normalize to convert from homogeneous to Cartesian coordinates
        X_cartesian = X_homogeneous[:3] / X_homogeneous[3]
        X.append(X_cartesian)

        # Concatenate all arrays horizontally to form a single 3x3 array
        # combined_array = np.vstack((point1, point2, X_cartesian)).T
        # uv_X3D.append(combined_array)
    # Convert list of 3D points to a numpy array for convenience
    return np.array(X)

def linearTriangulation1(K, C1, R1, C2, R2, x1, x2):

    I = np.identity(3)
    C1 = np.reshape(C1, (3,1))
    C2 = np.reshape(C2, (3,1))

    P1 = np.dot(K, np.dot(R1, np.hstack((I,-C1))))
    P2 = np.dot(K, np.dot(R2, np.hstack((I,-C2))))

    p1T = P1[0,:].reshape(1,4)
    p2T = P1[1,:].reshape(1,4)
    p3T = P1[2,:].reshape(1,4)

    p_1T = P2[0,:].reshape(1,4)
    p_2T = P2[1,:].reshape(1,4)
    p_3T = P2[2,:].reshape(1,4)

    X = []
    for i in range(x1.shape[0]):
       x = x1[i,0] 
       y = x1[i,1]
       x_ = x2[i,0]
       y_ = x2[i,1]

       A = []
       A.append((y * p3T) - p2T)
       A.append(p1T - (x * p3T))
       A.append((y_ * p_3T) - p_2T)
       A.append(p_1T - (x_ * p_3T))

       A = np.array(A).reshape(4,4)

       _,_,vt = np.linalg.svd(A)
       v = vt.T
       x = v[:,-1]
       x = x[0:3]/x[3]
       X.append(x)

    X = np.array(X)
    # X = X/X[:,3].reshape(-1,1)
    return X