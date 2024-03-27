import numpy as np

def projectionMatrix(R,C,K):
    C = np.reshape(C,(3,1))
    I = np.identity(3)
    P = K @ np.hstack((R, -R @ C))
    return P

def ReProjectionError(X,pts1, pts2, R1, C1, R2, C2, K):
    P1 = projectionMatrix(R1, C1, K)
    P2 = projectionMatrix(R2, C2, K)

    # print("This i s p1",p1,p1.shape)
    # print("This i s p2",p2,p2.shape)
    # Convert X to homogeneous coordinates
    X_homo = np.append(X, 1)
    
    # Project X onto both image planes
    x1_proj = P1 @ X_homo
    x2_proj = P2 @ X_homo

    # Convert from homogeneous to Cartesian coordinates
    x1_proj /= x1_proj[2]
    x2_proj /= x2_proj[2]

    # Compute reprojection errors
    error1 = np.sqrt(np.sum((pts1 - x1_proj[:2])**2))
    error2 = np.sqrt(np.sum((pts2 - x2_proj[:2])**2))

    return error1 + error2


def get_reprojection_error(pts1,pts2,X,X_refined,R1,C1,R_best,C_best,K):

    total_err1 = []
    for pt1, pt2, X_3d in zip(pts1,pts2,X):
        error1 = ReProjectionError(X_3d,pt1,pt2,R1,C1,R_best,C_best,K)
        total_err1.append(error1)
    
    mean_err1 = np.mean(total_err1)

    total_err2 = []
    for pt1, pt2, X_3d in zip(pts1,pts2,X_refined):
        error2 = ReProjectionError(X_3d,pt1,pt2,R1,C1,R_best,C_best,K)
        total_err2.append(error2)
    
    mean_err2 = np.mean(total_err2)

    print("Between images A & B Linear Training: ", mean_err1, "Non-Linear Training: ", mean_err2)

