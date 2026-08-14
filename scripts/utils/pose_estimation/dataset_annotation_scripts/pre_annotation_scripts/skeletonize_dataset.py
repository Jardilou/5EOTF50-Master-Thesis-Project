import cv2
import numpy as np
import os
import glob

def process_deepfish_masks(mask_dir, output_labels_dir):
    """
    Converts binary segmentation masks into YOLOv8-Pose Bézier labels.
    """
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Find all image masks in the folder (DeepFish usually uses .jpg or .png)
    mask_files = glob.glob(os.path.join(mask_dir, '*.*'))
    processed_count = 0
    
    print(f"Found {len(mask_files)} masks. Starting skeletonization...")
    
    for mask_path in mask_files:
        # Load the mask in Grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
            
        h_img, w_img = mask.shape
        
        # Ensure it is purely binary (0 = background, 255 = fish)
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Find every distinct fish in the mask
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        yolo_lines = []
        
        for cnt in contours:
            # Ignore tiny specks of noise
            if cv2.contourArea(cnt) < 100: 
                continue
                
            # 1. CALCULATE THE BOUNDING BOX
            x, y, w, h = cv2.boundingRect(cnt)
            cx = (x + w / 2) / w_img
            cy = (y + h / 2) / h_img
            nw = w / w_img
            nh = h / h_img
            
            # 2. SKELETONIZE THE FISH
            # Create a blank black canvas and draw ONLY this single fish on it
            single_fish = np.zeros_like(binary)
            cv2.drawContours(single_fish, [cnt], -1, 255, thickness=cv2.FILLED)
            
            # Use OpenCV's thinning algorithm to find the exact spine
            skeleton = cv2.ximgproc.thinning(single_fish)
            
            # Get the (y, x) coordinates of every pixel that makes up the spine
            pts_y, pts_x = np.where(skeleton > 0)
            
            # If the fish is too small and has less than 4 spine pixels, skip it
            if len(pts_x) < 4: 
                continue
            
            points = np.column_stack((pts_x, pts_y))
            
            # 3. SORT THE SPINE POINTS FROM HEAD TO TAIL (Using PCA Math)
            # Find the main directional axis of the fish
            mean, eigenvectors = cv2.PCACompute(points.astype(np.float32), mean=None)
            main_axis = eigenvectors[0]
            
            # Project all points onto this main axis to figure out their order
            projections = np.dot(points - mean, main_axis)
            sorted_indices = np.argsort(projections.flatten())
            sorted_points = points[sorted_indices]
            
            # 4. EXTRACT THE 4 BÉZIER CONTROL POINTS
            # We take the 0% (Head), 33% (Mid1), 66% (Mid2), and 100% (Tail) points
            idx = [0, len(sorted_points)//3, 2*len(sorted_points)//3, len(sorted_points)-1]
            sampled_points = sorted_points[idx]
            
            # 5. FORMAT AS YOLO STRING
            # Format: class cx cy w h px1 py1 vis1 px2 py2 vis2 ...
            # Visibility is set to 2 (meaning labeled and visible)
            line = f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
            for pt in sampled_points:
                px = pt[0] / w_img
                py = pt[1] / h_img
                line += f" {px:.6f} {py:.6f} 2"
            
            yolo_lines.append(line)
            
        # Save the .txt file ONLY if we actually found fish
        if yolo_lines:
            base_name = os.path.splitext(os.path.basename(mask_path))[0]
            txt_path = os.path.join(output_labels_dir, f"{base_name}.txt")
            
            with open(txt_path, 'w') as f:
                f.write("\n".join(yolo_lines))
                
            processed_count += 1

    print(f"\nSUCCESS! Converted {processed_count} masks into YOLO Bézier labels.")

import cv2
import numpy as np
import os
import glob

def process_deepfish_masks_sl(mask_dir, output_labels_dir):
    """
    Converts binary masks into YOLOv8-Pose Bézier labels based on STANDARD LENGTH.
    Uses Distance Transform to find the caudal peduncle and ignore the tail fin.
    """
    os.makedirs(output_labels_dir, exist_ok=True)
    mask_files = glob.glob(os.path.join(mask_dir, '*.*'))
    processed_count = 0
    
    print(f"Found {len(mask_files)} masks. Starting Standard Length extraction...")
    
    for mask_path in mask_files:
        # Load mask, ensuring it's binary
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        h_img, w_img = mask.shape
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yolo_lines = []
        
        for cnt in contours:
            if cv2.contourArea(cnt) < 200: continue # Skip noise
                
            # 1. BOUNDING BOX (Still encompasses the whole fish for YOLO context)
            x, y, w, h = cv2.boundingRect(cnt)
            cx = (x + w / 2) / w_img
            cy = (y + h / 2) / h_img
            nw = w / w_img
            nh = h / h_img
            
            # Create a clean canvas for just this fish
            single_fish = np.zeros_like(binary)
            cv2.drawContours(single_fish, [cnt], -1, 255, thickness=cv2.FILLED)
            
            # =================================================================
            # NEW: DISTANCE TRANSFORM (Calculate thickness of the fish)
            # =================================================================
            # This creates an image where pixel values equal distance to the edge
            dist_img = cv2.distanceTransform(single_fish, cv2.DIST_L2, 5)
            
            # 2. SKELETONIZE
            skeleton = cv2.ximgproc.thinning(single_fish)
            pts_y, pts_x = np.where(skeleton > 0)
            if len(pts_x) < 20: continue # Fish is too tiny/glitched
            points = np.column_stack((pts_x, pts_y))
            
            # 3. SORT SPINE POINTS (Using PCA)
            mean, eigenvectors = cv2.PCACompute(points.astype(np.float32), mean=None)
            projections = np.dot(points - mean, eigenvectors[0])
            sorted_indices = np.argsort(projections.flatten())
            sorted_points = points[sorted_indices]
            
            # 4. HEAD/TAIL ORIENTATION CHECK
            # Extract the thickness at every point along the spine
            thickness_profile = [dist_img[pt[1], pt[0]] for pt in sorted_points]
            
            # The head/body (front 25%) is always thicker than the tail fin (back 25%)
            front_thickness = np.mean(thickness_profile[:len(thickness_profile)//4])
            back_thickness = np.mean(thickness_profile[-len(thickness_profile)//4:])
            
            if back_thickness > front_thickness:
                # The math accidentally sorted it Tail-to-Head. Reverse it!
                sorted_points = sorted_points[::-1]
                thickness_profile = thickness_profile[::-1]

            # =================================================================
            # NEW: FIND THE CAUDAL PEDUNCLE (Standard Length Cutoff)
            # =================================================================
            # The peduncle is usually a "pinch" point in the last 15% to 35% of the fish.
            search_start = int(len(sorted_points) * 0.65) # Start looking past the belly
            search_end = int(len(sorted_points) * 0.95)   # Ignore the absolute frayed fin tips
            
            if search_end > search_start:
                # Find the index of the MINIMUM thickness in this rear search area
                rear_profile = thickness_profile[search_start:search_end]
                local_min_idx = np.argmin(rear_profile)
                peduncle_idx = search_start + local_min_idx
            else:
                peduncle_idx = len(sorted_points) - 1 # Fallback
                
            # TRUNCATE the spine! We throw away everything after the peduncle.
            sl_spine = sorted_points[:peduncle_idx + 1]

            # 5. EXTRACT THE 4 CONTROL POINTS FROM THE NEW SHORTENED SPINE
            idx = [0, len(sl_spine)//3, 2*len(sl_spine)//3, len(sl_spine)-1]
            sampled_points = sl_spine[idx]
            
            # 6. FORMAT AS YOLO STRING
            line = f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
            for pt in sampled_points:
                px = pt[0] / w_img
                py = pt[1] / h_img
                line += f" {px:.6f} {py:.6f} 2"
            yolo_lines.append(line)
            
        if yolo_lines:
            base_name = os.path.splitext(os.path.basename(mask_path))[0]
            txt_path = os.path.join(output_labels_dir, f"{base_name}.txt")
            with open(txt_path, 'w') as f:
                f.write("\n".join(yolo_lines))
            processed_count += 1

    print(f"\nSUCCESS! Extracted {processed_count} Standard Length annotations.")

# --- USAGE ---
mask_folder = r"path\to\masks"
output_folder = r"path\to\output_labels"
process_deepfish_masks_sl(mask_folder, output_folder)


# --- USAGE ---
# Point this to where you extracted the DeepFish masks
deepfish_masks = r"DATASETS/DeepFish/Segmentation/masks/valid"

# Point this to where you want the new .txt files saved
output_yolo_labels = r"DATASETS/DeepFish/Segmentation/masks/valid_labels"

process_deepfish_masks_sl(deepfish_masks, output_yolo_labels)