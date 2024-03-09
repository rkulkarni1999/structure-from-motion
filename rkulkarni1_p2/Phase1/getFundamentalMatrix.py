import numpy as np
import cv2

def normalize_points(points_set):
    
    points_set_ = np.mean(points_set, axis=0)
    points_left, points_right = points_set_[0], points_set_[1]
    

# points left and points_right -> matched features across two image views. There should be atleast 8 correspondences
def get_fundamental_matrix(points_left, points_right):
    
    pts_lft, pts_rgt = points_left, points_right
    
    
