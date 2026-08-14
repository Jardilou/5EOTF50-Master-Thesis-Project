import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_fundamental_matrix(K1, K2, R, T):
    """
    Computes the Fundamental Matrix (F) from intrinsic and extrinsic camera matrices.
    """
    Tx = np.array([
        [0, -T[2, 0], T[1, 0]],
        [T[2, 0], 0, -T[0, 0]],
        [-T[1, 0], T[0, 0], 0]
    ])
    E = np.dot(Tx, R)
    K2_inv_T = np.linalg.inv(K2).T
    K1_inv = np.linalg.inv(K1)
    F = np.dot(K2_inv_T, np.dot(E, K1_inv))
    F = F / F[2, 2]
    return F

def match_boxes_epipolar(left_boxes, right_boxes, left_embeddings, right_embeddings, F, max_epipolar_dist=50.0, max_area_diff_ratio=2.5, min_box_size=35):
    """
    Pairs bounding boxes using strict Epipolar Geometry and Deep Visual Similarity (Cosine Distance).
    Filters out distant/tiny fish to preserve data accuracy and prevent false matches.
    """
    if not left_boxes or not right_boxes:
        return []

    # 1. Extract features, embeddings, and filter out tiny, distant fish
    def get_valid_features(boxes, embeddings):
        features = []
        valid_data = []
        for box, emb in zip(boxes, embeddings):
            xmin, ymin, xmax, ymax = box
            w = xmax - xmin
            h = ymax - ymin
            
            # Drop fish that are too far away (bounding box too small)
            if w < min_box_size or h < min_box_size:
                continue
                
            cx = xmin + w / 2.0
            cy = ymin + h / 2.0
            area = w * h
            aspect_ratio = w / h if h > 0 else 1.0
            
            features.append({'cx': cx, 'cy': cy, 'w': w, 'h': h, 'area': area, 'ar': aspect_ratio, 'embedding': emb})
            valid_data.append(box)
        return features, valid_data

    l_feats, valid_l_boxes = get_valid_features(left_boxes, left_embeddings)
    r_feats, valid_r_boxes = get_valid_features(right_boxes, right_embeddings)
    
    if not valid_l_boxes or not valid_r_boxes:
        return []

    # 2. Build the Cost Matrix based on Mathematical Distance and Visual Similarity
    cost_matrix = np.zeros((len(valid_l_boxes), len(valid_r_boxes)))

    for i, lf in enumerate(l_feats):
        # Format the center of the left box as a homogeneous 3D coordinate
        pt_l = np.array([lf['cx'], lf['cy'], 1.0])
        
        # Project the left point through the Fundamental Matrix
        epipolar_line_r = np.dot(F, pt_l)
        A, B, C = epipolar_line_r
        line_norm = np.sqrt(A**2 + B**2) + 1e-6

        for j, rf in enumerate(r_feats):
            # Calculate exact geometric distance
            pt_r = np.array([rf['cx'], rf['cy'], 1.0])
            epipolar_dist = abs(np.dot(epipolar_line_r, pt_r)) / line_norm
            
            area_ratio = max(lf['area'] / rf['area'], rf['area'] / lf['area'])
            ar_diff = abs(lf['ar'] - rf['ar'])
            
            # Deep Visual Similarity (Cosine Distance)
            emb_l, emb_r = lf['embedding'], rf['embedding']
            norm_l, norm_r = np.linalg.norm(emb_l), np.linalg.norm(emb_r)
            
            if norm_l > 0 and norm_r > 0:
                cosine_sim = np.dot(emb_l, emb_r) / (norm_l * norm_r)
            else:
                cosine_sim = 0.0
                
            # Convert similarity (1.0 = identical, 0.0 = completely different) to a penalty cost
            appearance_penalty = (1.0 - cosine_sim) * 100.0

            # Strict Rejection
            if epipolar_dist > max_epipolar_dist or area_ratio > max_area_diff_ratio:
                cost_matrix[i, j] = 999999.0
            else:
                # Cost combines geometric epipolar distance, shape changes, and visual appearance!
                cost_matrix[i, j] = epipolar_dist + (area_ratio * 15.0) + (ar_diff * 30.0) + appearance_penalty

    # 3. Solve the assignment problem optimally (Hungarian Algorithm)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 4. Filter out the invalid pairings
    matched_pairs = []
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < 999999.0:
            matched_pairs.append((valid_l_boxes[i], valid_r_boxes[j]))

    return matched_pairs