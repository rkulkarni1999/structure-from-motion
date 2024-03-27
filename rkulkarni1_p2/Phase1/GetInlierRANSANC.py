import numpy as np
from EstimateFundamentalMatrix import get_fundamental_matrix

def ransac_inliers(pts1, pts2, idx):
    iteration = 2000
    thresold = 0.005     #choosed this value after going through all image pairs rejecting most of the outliers.
    max_inliers = 0
    inliers_index =[]

    for i in range(iteration):
        
        total_feature_points = pts1.shape[0]
        random_index = np.random.choice(total_feature_points,8)
        x1 = pts1[random_index,:]
        x2 = pts2[random_index,:]

        F = get_fundamental_matrix(x1,x2)
        index =[]
        count = 0
        if F is not None:
            for j in range(total_feature_points):
                x1 = np.array([pts1[j,0],pts1[j,1],1])
                x2 = np.array([pts2[j,0],pts2[j,1],1]).T
                error = np.dot(x2,np.dot(F,x1))          #did not used absolute here
                if np.abs(error)< thresold:
                    count +=1
                    index.append(idx[j])
            
        if(count>max_inliers):
            inliers_index = index
            max_inliers = count
            F_inliers = F                               # TODO calculate F based on all inliners

    return F_inliers, inliers_index

