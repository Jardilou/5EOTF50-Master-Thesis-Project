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
    # CRITICAL FIX: cv2.triangulatePoints absolutely requires 2x1 arrays.
    pts_left = np.array(pt_left, dtype=np.float64).reshape(2, 1)
    pts_right = np.array(pt_right, dtype=np.float64).reshape(2, 1)
    
    # cv2.triangulatePoints outputs homogeneous coordinates (4D)
    pts_4d = cv2.triangulatePoints(P1, P2, pts_left, pts_right)
    
    # Prevent division by zero if point triangulated to infinity
    w = pts_4d[3, 0]
    if abs(w) < 1e-6:
        print(f"[TRIANG-DEBUG] Warning: Triangulated to infinity (w={w}). Returning NaNs.")
        return np.array([np.nan, np.nan, np.nan])
        
    # Convert homogeneous (x, y, z, w) to Euclidean (x/w, y/w, z/w)
    pts_3d = (pts_4d[:3, 0] / w).flatten()
    return pts_3d

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
        return 0.0, np.array([])

    total_length_cm = 0.0
    previous_3d_pt = None
    points_3d = [] # Store points for 3D visualization
    
    for pt_l, pt_r in zip(left_centerline_pts, right_centerline_pts):
        # 1. Triangulate the discrete point to get Z(depth), X, Y in real world units (cm)
        current_3d_pt = triangulate_point(pt_l, pt_r, P1, P2)
        points_3d.append(current_3d_pt)
        
        if previous_3d_pt is not None:
            # 2. Euclidean distance between the sequential 3D centerline points
            distance = np.linalg.norm(current_3d_pt - previous_3d_pt)
            total_length_cm += distance
            
        previous_3d_pt = current_3d_pt
        
    return total_length_cm*100, np.array(points_3d)

def calculate_3d_point_cloud(left_mask, right_mask, P1, P2, density=15):
    """
    Generates a dense 3D point cloud representing the volume of the fish.
    Uses Proportional Anatomical Mapping to handle unrectified vertical camera shifts.
    """
    import numpy as np
    
    # Ensure masks are boolean arrays
    l_mask = (left_mask > 0)
    r_mask = (right_mask > 0)
    
    y_l, x_l = np.where(l_mask)
    y_r, x_r = np.where(r_mask)
    
    if len(y_l) == 0 or len(y_r) == 0:
        return np.array([])
        
    # Find the bounding boxes of the masks (Top and Bottom limits)
    min_yl, max_yl = y_l.min(), y_l.max()
    min_yr, max_yr = y_r.min(), y_r.max()
    
    raw_points_3d = []
    
    # Sample relative horizontal slices (from 5% to 95% of the fish's height)
    for ratio in np.linspace(0.05, 0.95, density):
        # Map the relative height to the actual Y pixel in both cameras
        yl = int(min_yl + ratio * (max_yl - min_yl))
        yr = int(min_yr + ratio * (max_yr - min_yr))
        
        # Get the horizontal span of the fish on this specific row
        x_l_row = x_l[y_l == yl]
        x_r_row = x_r[y_r == yr]
        
        if len(x_l_row) == 0 or len(x_r_row) == 0:
            continue
            
        min_xl, max_xl = x_l_row.min(), x_l_row.max()
        min_xr, max_xr = x_r_row.min(), x_r_row.max()
        
        # Interpolate X coordinates across the width of the fish (Left border to Right border)
        sampled_xl = np.linspace(min_xl, max_xl, density)
        sampled_xr = np.linspace(min_xr, max_xr, density)
        
        for xl, xr in zip(sampled_xl, sampled_xr):
            # Triangulate this specific interior/border point
            pt_3d = triangulate_point((xl, yl), (xr, yr), P1, P2)
            
            # FAILSAFE: Prevent division-by-zero
            if not np.any(np.isnan(pt_3d)) and not np.any(np.isinf(pt_3d)):
                raw_points_3d.append(pt_3d)
                
    if not raw_points_3d:
        return np.array([])
        
    raw_points_3d = np.array(raw_points_3d)
    
    # Statistical Outlier Rejection
    z_values = raw_points_3d[:, 2]
    median_z = np.median(z_values)
    
    # Keep points strictly within the immediate physical volume of the fish (+/- 30cm)
    valid_mask = np.abs(z_values - median_z) < 30.0
    clean_points_3d = raw_points_3d[valid_mask]
            
    return clean_points_3d