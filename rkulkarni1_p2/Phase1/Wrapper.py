# Create Your Own Starter Code :)
import numpy as np
import argparse
import cv2 as cv
from GetInlierRANSANC import ransac_inliers
from drawmatches import draw_matches
import numpy as np
from EssentialMatrixFromFundamentalMatrix import get_essential_matrix
from ExtractCameraPose import get_camera_pose
from LInerTriangulation import linearTriangulation, linearTriangulation1
from plotxzcoordinate import plotxz
from DisambiguateCameraPose import DisambiguatePose
from Nonlineartriangulation import NonLinearTriangulation
from Reprojectionerror import ReProjectionError,get_reprojection_error
from PnPRansac import PnPRANSAC
from LinerPnP import reprojectionErrorPnP
from Utils.dataloder import features_extraction
from NonlinerPnP import NonLinearPnP
from Utils.ransac_filterfeatures import Filter_features_usingransac
from Utils.VisibilityMatrix import getObservationsIndexAndVizMat
from BundleAdjustment import BundleAdjustment

def get_X_lineartriangulation(R12_possible,C12_possible,K,R1,C1,pts1,pts2):
    X_RC_possible =[]
    for i in range(len(C12_possible)):
        x1 = pts1
        x2 = pts2
        X = linearTriangulation(K, C1, R1, C12_possible[i], R12_possible[i], x1, x2)
        X_RC_possible.append(X)
    return X_RC_possible


def main(Args):

    Data = Args.Data
    Output = Args.Outputs

    images = []
    for i in range(1,6): #6 images given
        path = Data + "/" + str(i) + ".png"
        image = cv.imread(path)
        if image is not None:
            images.append(image)
        else:
            print("No image is found")

    #Feature Correspondence
    u_x,v_y,flag,rgb_values =features_extraction(Data)

    print("Removing all outlier matches using ransac......")

    # idx = np.where((flag[:, 0]) & (flag[:, 1]))[0]
    # pts1 = np.column_stack((u_x[idx, 0], v_y[idx, 0]))
    # pts2 = np.column_stack((u_x[idx, 1], v_y[idx, 1]))
    # draw_matches(images[0], pts1, images[1], pts2)

    all_fundamental_matrix,RANSAC_feature_flag = Filter_features_usingransac(u_x,v_y,flag,images)

    print("Outlier removed")

    print("#####Obtating 3D feature point########")


    print("##performing for image 1 and 2###")
    F12 = all_fundamental_matrix[0,1]
    K = np.array([[531.122155322710, 0 ,407.192550839899],
                  [0, 531.541737503901, 313.308715048366],
                  [0,0,1]])
    
    E12 = get_essential_matrix(K,F12)
    R12_possible,C12_possible = get_camera_pose(E12)  ## return 4 set of poses

    idx = np.where(RANSAC_feature_flag[:,0] & RANSAC_feature_flag[:,1])[0]
    pts1 = np.column_stack((u_x[idx, 0], v_y[idx, 0]))
    pts2 = np.column_stack((u_x[idx, 1], v_y[idx, 1]))

    R1 = np.identity(3)
    C1 = np.zeros((3,1))

    X_RC_possible = get_X_lineartriangulation(R12_possible,C12_possible,K,R1,C1,pts1,pts2)
    # plotxz(X_RC_possible)
 
    R12, C12, X12 = DisambiguatePose(R12_possible,C12_possible,X_RC_possible)
    print(R12)
    print(C12)
    # FOR PLOTTING X-Z coordinates for image 1 and 2 pair after chailerity check
    # pts3D_1 =[]
    # pts3D_1.append(X12)
    # pts3D_1.append(X12)
    # plotxz(pts3D_1)

    X12_refined = NonLinearTriangulation(K,pts1,pts2,X12,R1,C1,R12,C12)

    # pts3D_refined =[]
    # pts3D_refined.append(X12_refined)
    # pts3D_refined.append(X12)
    # plotxz(pts3D_refined)

    get_reprojection_error(pts1,pts2,X12,X12_refined,R1,C1,R12,C12,K)

    "Resistering Cam 1 and 2"
    X_all = np.zeros((u_x.shape[0],3))
    cam_indices = np.zeros((u_x.shape[0],1), dtype = int)
    X_found = np.zeros((u_x.shape[0],1), dtype = int)

    X_all[idx] = X12_refined[:,:3]
    X_found[idx] = 1
    cam_indices[idx] = 1
    X_found[np.where(X_all[:2]<0)] = 0

    C_set = []
    R_set = []

    C0 = np.zeros(3)
    R0 = np.identity(3)
    C_set.append(C0)
    R_set.append(R0)
    C_set.append(C12)
    R_set.append(R12)

    print("#########Registered Cam 1 and Cam 2 ############")

    for i in range(2,5):
        print("Registering Image: ", str(i+1))
        feature_idx_i = np.where(X_found[:,0] & RANSAC_feature_flag[:,i])[0]
        if len(feature_idx_i) < 8:
            print("Got ", len(feature_idx_i), "common points between X and ", i, "image")
            continue

        ptsi = np.column_stack((u_x[feature_idx_i, i], v_y[feature_idx_i, i]))
        X_common = X_all[feature_idx_i,:].reshape(-1,3) #X_COMMON between image 1 2 and 3

        # estimating pose of next cam based on already calculated X word points uisng Pnp
        R_init, C_init = PnPRANSAC(K,ptsi,X_common, iter=1000, thresh=100)
        linear_error_pnp = reprojectionErrorPnP(X_common, ptsi, K, R_init, C_init)

        Ri, Ci = NonLinearPnP(K, ptsi, X_common, R_init, C_init)
        non_linear_error_pnp = reprojectionErrorPnP(X_common, ptsi, K, Ri, Ci)
        print("Initial linear PnP error: ", linear_error_pnp, " Final Non-linear PnP error: ", non_linear_error_pnp)
        C_set.append(Ci)
        R_set.append(Ri)
        for k in range(0,i):
            idx_i_to_k = np.where(RANSAC_feature_flag[:,k] & RANSAC_feature_flag[:,i])[0]
            if (len(idx_i_to_k)<8):
                continue
            x1 = np.column_stack((u_x[idx_i_to_k, 0], v_y[idx_i_to_k, 0]))
            x2 = np.column_stack((u_x[idx_i_to_k, 1], v_y[idx_i_to_k, 1]))
            Xik = linearTriangulation(K,C_set[k],R_set[k],Ci,Ri,x1,x2)
            Xik_refined = NonLinearTriangulation(K,x1,x2,Xik,R_set[k],C_set[k],Ri,Ci)
            get_reprojection_error(x1,x2,Xik,Xik_refined,R_set[k],C_set[k],Ri,Ci,K)

            X_all[idx_i_to_k] = Xik_refined[:,:3]
            X_found[idx_i_to_k] = 1
            
            X_index, visibility_matrix = getObservationsIndexAndVizMat(X_found,RANSAC_feature_flag,nCam=i)

            print("########Bundle Adjustment Started")
            R_set_, C_set_, Xb_all = BundleAdjustment(X_index, visibility_matrix,X_all,X_found,u_x,v_y,RANSAC_feature_flag,R_set,C_set,K,nCam=i)
            # print(np.array(R_set).shape,np.array(C_set).shape,X_all.shape)
            
            for k in range(0,i+1):
                idx_X_pts = np.where(X_found[:,0] & RANSAC_feature_flag[:,k])[0]
                x = np.column_stack((u_x[idx_X_pts,k], v_y[idx_X_pts,k]))
                X = Xb_all[idx_X_pts]
                BundAdj_error = reprojectionErrorPnP(X,x,K,R_set_[k],C_set_[k])
                print("########Error after Bundle Adjustment: ", BundAdj_error)

            print("############Regestired camera: ", i+1,"############################")


    pts3D_refined =[]
    pts3D_refined.append(X_all)
    pts3D_refined.append(Xb_all)
    plotxz(pts3D_refined)

if __name__ == '__main__':
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Outputs', default='../Outputs/', help='Outputs are saved here')
    Parser.add_argument('--Data', default='./P3Data', help='Data')

    Args = Parser.parse_args()
    main(Args)
