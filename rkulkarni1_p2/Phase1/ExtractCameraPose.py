import numpy as np

def get_camera_pose(E):
    U,D,V_T = np.linalg.svd(E)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    R=[]
    C=[]

    # Possible rotation
    R.append(np.dot(U,np.dot(W,V_T)))
    R.append(np.dot(U,np.dot(W,V_T)))
    R.append(np.dot(U,np.dot(W.T,V_T)))
    R.append(np.dot(U,np.dot(W.T,V_T)))
    
    # Possible translation
    C.append(U[:,2])
    C.append(-U[:,2])
    C.append(U[:,2])
    C.append(-U[:,2])

    for i in range(4):
        if (np.linalg.det(R[i])<0):
            R[i] = -R[i]
            C[i] = -C[i]
    
    return R, C