import numpy as np
from GetInlierRANSANC import ransac_inliers
from drawmatches import draw_matches

def Filter_features_usingransac(u_x,v_y,flag,images):
    RANSAC_feature_flag = np.zeros_like(flag) #np.zeros has limit which is solve by zeros_like
    all_fundamental_matrix = np.empty(shape=(5,5), dtype=object)

    for i in range(4):  # Iterating from the first image to the fourth
        for j in range(i + 1, 5):  # Ensuring we only pair each image with those following it

            # Find indices where both conditions are met, simplifying the extraction of indices
            idx = np.where((flag[:, i]) & (flag[:, j]))[0]
            # Directly access and pair x, y coordinates using advanced indexing
            pts1 = np.column_stack((u_x[idx, i], v_y[idx, i]))
            pts2 = np.column_stack((u_x[idx, j], v_y[idx, j]))
            idx = np.array(idx).reshape(-1)

            # draw_matches(images[i], pts1, images[j], pts2)

            if len(idx)>8:
                F , in_idx = ransac_inliers(pts1, pts2, idx)
                print("Between Images: ",i,"and",j,"NO of Inliers: ", len(in_idx), "/", len(idx) )
                all_fundamental_matrix[i,j] = F
                RANSAC_feature_flag[in_idx,i]=1
                RANSAC_feature_flag[in_idx,j]=1
        #     f_pts1 = np.column_stack((u_x[in_idx, i], v_y[in_idx, i]))
        #     f_pts2 = np.column_stack((u_x[in_idx, j], v_y[in_idx, j]))
        #     draw_matches(images[i], f_pts1, images[j], f_pts2)
        # break                                                                                   ## TO BE DELETED
        
    return all_fundamental_matrix,RANSAC_feature_flag