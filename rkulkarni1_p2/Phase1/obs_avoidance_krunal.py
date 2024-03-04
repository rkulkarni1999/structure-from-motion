import numpy as np
import os
import bpy

def Exr2Depth():
    pass


def compute_vector(GoalLocation):
    pixelSize = 8 
    focalLengthMetric = bpy.context.scene.camera.data.lens
    sensorSizeMetric = bpy.context.scene.camera.data.sensor_width
    fx = focalLengthMetric*pixelSize/sensorSizeMetric
    fy = focalLengthMetric*pixelSize/sensorSizeMetric

    cx = pixelSize/2
    cy = pixelSize/2   
        
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    # Grid of vectors. one time calculation. so no need to vectorize
    vectorGrid = []
    for u in range(8):
        row = []
        for v in range(8): 
            v_ = np.linalg.inv(K)@np.array([u, v, 1]) # gives raw pixel values. 
            row.append(v_)
        
        vectorGrid.append(row)
    vectorGrid = np.array(vectorGrid)   
    
    D_front = Exr2Depth(os.path.join(path_dir, 'Frame%04d.exr'%(bpy.data.scenes[1].frame_current)))

    depthWeightedVectors_front = vectorGrid/D_front[:, :, np.newaxis]
    finalDirection_front = np.sum(depthWeightedVectors_front, axis=(0, 1))

    R_quad__front = np.array([[1, 0, 0],     
                            [0, 0, -1],
                            [0, 1, 0]])
    frontcam_dir = frontcam_dir@R_quad__front                
    
    obs_dir = frontcam_dir 

    lamda = 2 # This is a scaling parameter
    control_dir = GoalLocation - lamda*obs_dir
    # delta_dir = control_dir*0.01
    delta_dir = control_dir
    
    return delta_dir
    
    

