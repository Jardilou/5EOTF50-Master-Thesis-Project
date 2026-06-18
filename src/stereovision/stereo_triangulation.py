import cv2
import numpy as np

def load_stereo_matrices(npz_filepath):
    """Loads the calibration matrices saved from Step 2 Notebook."""
    data = np.load(npz_filepath)
    # The essential projection matrices for Left (P1) and Right (P2) cameras
    return data['P1'], data['P2']

def triangulate_point(pt_left, pt_right, P1, P2):
    """
    Triangulates a single 2D point pair into 3D space.
    pt_left, pt_right: tuples (x, y)
    Returns: 3D numpy array (X, Y, Z)
    """
    pts_left = np.array([[pt_left]], dtype=np.float64)
    pts_right = np.array([[pt_right]], dtype=np.float64)
    
    # cv2.triangulatePoints outputs homogeneous coordinates (4D)
    pts_4d = cv2.triangulatePoints(P1, P2, pts_left, pts_right)
    
    # Convert homogeneous (x, y, z, w) to Euclidean (x/w, y/w, z/w)
    pts_3d = pts_4d[:3] / pts_4d[3]
    return pts_3d.flatten()

def calculate_3d_centerline_length(left_centerline_pts, right_centerline_pts, P1, P2):
    """
    Computes the absolute 3D true body length L using Discrete Euclidean Summation.
    L = Sum(sqrt(dX^2 + dY^2 + dZ^2)) over the centerline trajectory.
    
    left_centerline_pts: List of (x,y) coordinates ordered from head to tail
    right_centerline_pts: List of matching (x,y) coordinates ordered from head to tail
    """
    if len(left_centerline_pts) != len(right_centerline_pts):
        raise ValueError("Must have the same number of matching points in both cameras.")
    
    if len(left_centerline_pts) < 2:
        return 0.0

    total_length_cm = 0.0
    previous_3d_pt = None
    
    for pt_l, pt_r in zip(left_centerline_pts, right_centerline_pts):
        # 1. Triangulate the discrete point to get Z(depth), X, Y in real world units (cm)
        current_3d_pt = triangulate_point(pt_l, pt_r, P1, P2)
        
        if previous_3d_pt is not None:
            # 2. Euclidean distance between the sequential 3D centerline points
            distance = np.linalg.norm(current_3d_pt - previous_3d_pt)
            total_length_cm += distance
            
        previous_3d_pt = current_3d_pt
        
    return total_length_cm