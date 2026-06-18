import cv2
import numpy as np
from scipy.interpolate import splprep, splev

def extract_centerline(binary_mask, num_points=20):
    """
    Extracts the morphological centerline (spine) of a fish from its binary mask.
    Returns a list of discrete (x, y) coordinates from head to tail.
    """
    # 1. Find the external contour of the fish mask
    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
        
    main_contour = max(contours, key=cv2.contourArea)
    
    # 2. Geometric Heuristic (Cheng et al., 2020 approximation): 
    # Find the two points on the contour that are furthest apart (Head and Tail)
    # This is a simplified but highly robust approach for underwater imagery.
    max_dist = 0
    head_pt, tail_pt = None, None
    
    # Compute convex hull to reduce point comparisons significantly
    hull = cv2.convexHull(main_contour).squeeze()
    if len(hull.shape) < 2 or len(hull) < 2: return []
    
    for i in range(len(hull)):
        for j in range(i + 1, len(hull)):
            dist = np.linalg.norm(hull[i] - hull[j])
            if dist > max_dist:
                max_dist = dist
                head_pt, tail_pt = hull[i], hull[j]
                
    if head_pt is None or tail_pt is None:
        return []

    # 3. Parametric B-Spline interpolation along the medial axis.
    # For a robust, fast approximation, we interpolate a curve between the extreme points 
    # that gravitates toward the center of mass of the mask.
    M = cv2.moments(main_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = (head_pt[0] + tail_pt[0]) // 2, (head_pt[1] + tail_pt[1]) // 2

    # Fit a smooth spline through Head -> Center of Mass -> Tail
    x_coords = [head_pt[0], cx, tail_pt[0]]
    y_coords = [head_pt[1], cy, tail_pt[1]]
    
    # Ensure there are no duplicate points which crash the spline
    if len(set(zip(x_coords, y_coords))) < 3:
        return [(head_pt[0], head_pt[1]), (tail_pt[0], tail_pt[1])]

    tck, u = splprep([x_coords, y_coords], s=0, k=2)
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)
    
    # Return as list of discrete tuples [(x1,y1), (x2,y2)...]
    centerline_pts = [(int(x), int(y)) for x, y in zip(x_new, y_new)]
    return centerline_pts