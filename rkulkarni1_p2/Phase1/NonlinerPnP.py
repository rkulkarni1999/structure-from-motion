from scipy.spatial.transform import Rotation
import scipy.optimize as optimize
import numpy as np

def get_rotation_from_matrix(rotation_matrix):
    """Converts a rotation matrix to a quaternion."""
    return Rotation.from_matrix(rotation_matrix).as_quat()

def projection_matrix(rotation_matrix, camera_center, camera_matrix):
    """Calculates the projection matrix given rotation matrix, camera center, and camera matrix."""
    extrinsic_matrix = np.hstack((rotation_matrix, -rotation_matrix @ camera_center.reshape(-1, 1)))
    return camera_matrix @ extrinsic_matrix

def reprojection_error(params, object_points, image_points, camera_matrix):
    """Calculates the reprojection error."""
    quaternion = params[:4]
    camera_center = params[4:]
    rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
    proj_matrix = projection_matrix(rotation_matrix, camera_center, camera_matrix)

    projected_points = proj_matrix @ np.hstack((object_points, np.ones((object_points.shape[0], 1)))).T
    projected_points /= projected_points[2, :]
    reprojection_errors = np.linalg.norm(image_points.T - projected_points[:2, :], axis=0)
    
    return np.sum(reprojection_errors**2)  # Minimize the sum of squared errors

def NonLinearPnP(camera_matrix, image_points, object_points, initial_rotation_matrix, initial_camera_center):
    """Performs non-linear optimization to refine the camera pose using initial estimates."""
    initial_quaternion = get_rotation_from_matrix(initial_rotation_matrix)
    initial_guess = np.hstack((initial_quaternion, initial_camera_center))

    result = optimize.least_squares(reprojection_error, initial_guess, method='trf', args=(object_points, image_points, camera_matrix))
    optimized_params = result.x
    optimized_quaternion = optimized_params[:4]
    optimized_camera_center = optimized_params[4:]
    optimized_rotation_matrix = Rotation.from_quat(optimized_quaternion).as_matrix()
    return optimized_rotation_matrix, optimized_camera_center