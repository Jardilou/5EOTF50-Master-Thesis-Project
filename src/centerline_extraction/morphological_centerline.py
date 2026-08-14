import cv2
import numpy as np
from scipy.interpolate import splprep, splev

def extract_centerline(binary_mask, num_points=20):
    """
    Extracts the morphological centerline (spine) of a fish using 
    Principal Component Analysis (PCA) to find the true anatomical extremes.
    """
    # 1. Get all (x, y) coordinates belonging to the fish body
    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    y_indices, x_indices = np.where(mask_uint8 > 0)
    
    # Failsafe for empty or microscopic masks
    if len(x_indices) < 10:
        return []
        
    points = np.column_stack((x_indices, y_indices))
    
    # 2. Apply Principal Component Analysis (PCA)
    # Center the data
    mean_pt = np.mean(points, axis=0)
    centered_points = points - mean_pt
    
    # Calculate Covariance Matrix and Eigen decomposition
    cov_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # The primary axis (spine direction) is the eigenvector with the largest eigenvalue
    pc1_idx = np.argmax(eigenvalues)
    primary_vector = eigenvectors[:, pc1_idx]
    
    # 3. Project all fish pixels onto the primary spine vector
    # This mathematically squashes the fish flat along its own axis
    projections = np.dot(centered_points, primary_vector)
    
    # The minimum and maximum projected values represent the absolute Head and Tail
    head_idx = np.argmin(projections)
    tail_idx = np.argmax(projections)
    
    head_pt = points[head_idx]
    tail_pt = points[tail_idx]

    # 4. Find the Center of Mass (to help guide the spline through the belly)
    M = cv2.moments(mask_uint8)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = int(mean_pt[0]), int(mean_pt[1])

    # 5. Fit a smooth parametric B-Spline through Head -> Center of Mass -> Tail
    x_coords = [head_pt[0], cx, tail_pt[0]]
    y_coords = [head_pt[1], cy, tail_pt[1]]
    
    # Ensure there are no duplicate points which would crash the spline interpolator
    unique_points = list(set(zip(x_coords, y_coords)))
    if len(unique_points) < 3:
        return [(int(head_pt[0]), int(head_pt[1])), (int(tail_pt[0]), int(tail_pt[1]))]

    tck, u = splprep([x_coords, y_coords], s=0, k=2)
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)
    
    # Return as list of discrete tuples [(x1,y1), (x2,y2)...]
    centerline_pts = [(int(x), int(y)) for x, y in zip(x_new, y_new)]
    return centerline_pts