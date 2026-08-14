"""
================================================================================
Script: Interactive YOLO Mask Cleaner
================================================================================

Description:
    An OpenCV-based graphical interface for reviewing and manually cleaning 
    YOLO-format instance segmentation datasets. The script renders a side-by-side 
    view of the original image and the annotated mask overlay. It provides basic 
    viewport controls (pan and zoom) to inspect specific image regions, and 
    allows users to remove incorrect polygons via mouse input. The updated 
    annotations can be saved directly back to the original text files.

    Key Functions:
    1. Dual Display: Shows the raw image alongside the annotated image to 
       facilitate visual verification.
    2. Viewport Navigation: Supports scroll-wheel zooming and right-click 
       panning for navigating large images within a fixed window size.
    3. Polygon Deletion: Left-clicking inside a rendered polygon removes 
       it from the active session.
    4. File Saving: Pressing 'S' overwrites the source .txt file with the 
       current, modified set of polygons.

Author:
    Louis Jardinet
    Universiti Teknologi Petronas

Date:
    January 2026 - June 2026

Usage:
    Mouse Interactions:
    [Left-Click]  : Delete the selected polygon
    [Right-Hold]  : Pan the camera view
    [Scroll]      : Zoom in or out

    Keyboard Shortcuts:
    [S] : Save the current annotations to the .txt file
    [D] : Load the next image
    [A] : Load the previous image
    [+] / [-] : Keyboard alternatives for zooming
    [Q] : Exit the script

Dependencies:
    - Python standard libraries: os
    - External packages: opencv-python (cv2), numpy

Inputs:
    - img_dir: Path to the directory containing the source images.
    - label_dir: Path to the directory containing the YOLO .txt label files.
================================================================================
"""

import os
import cv2
import numpy as np

# --- GLOBALS FOR UI AND CAMERA ---
current_lines = []
current_polygons = []
redraw_needed = False
img_w_global = 0

# Viewport Configuration
VIEW_W, VIEW_H = 1600, 900  # Size of the application window on your monitor
scale = 1.0
pan_x = 0
pan_y = 0
is_panning = False
start_pan_x = 0
start_pan_y = 0

def click_event(event, x, y, flags, param):
    global current_lines, current_polygons, redraw_needed, img_w_global
    global scale, pan_x, pan_y, is_panning, start_pan_x, start_pan_y

    # Translate the mouse's screen coordinate back to the absolute image coordinate
    abs_x = int(x / scale) + pan_x
    abs_y = int(y / scale) + pan_y

    # 1. DELETE LOGIC (Left Click)
    if event == cv2.EVENT_LBUTTONDOWN:
        if abs_x >= img_w_global:
            local_x = abs_x - img_w_global
            for i in range(len(current_polygons) - 1, -1, -1):
                poly = current_polygons[i]
                if cv2.pointPolygonTest(poly, (local_x, abs_y), False) >= 0:
                    print(f"[-] Deleted polygon {i}")
                    current_polygons.pop(i)
                    current_lines.pop(i)
                    redraw_needed = True
                    break

    # 2. PANNING LOGIC (Right Click & Drag)
    elif event == cv2.EVENT_RBUTTONDOWN:
        is_panning = True
        start_pan_x, start_pan_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_panning:
            dx = x - start_pan_x
            dy = y - start_pan_y
            pan_x -= int(dx / scale)
            pan_y -= int(dy / scale)
            start_pan_x, start_pan_y = x, y
            redraw_needed = True

    elif event == cv2.EVENT_RBUTTONUP:
        is_panning = False

    # 3. ZOOMING LOGIC (Mouse Wheel)
    elif event == cv2.EVENT_MOUSEWHEEL:
        zoom_factor = 1.15 if flags > 0 else 0.85
        new_scale = scale * zoom_factor
        
        # Lock max and min zoom
        new_scale = max(0.2, min(new_scale, 10.0))

        # Mathematical offset to ensure we zoom in on the exact location of the mouse cursor
        pan_x = abs_x - int(x / new_scale)
        pan_y = abs_y - int(y / new_scale)
        
        scale = new_scale
        redraw_needed = True

def clean_dataset_interactively(img_dir, label_dir):
    global current_lines, current_polygons, redraw_needed, img_w_global
    global scale, pan_x, pan_y
    
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    if not label_files:
        print("No labels found in the target directory.")
        return

    # Use AUTOSIZE because our custom camera engine controls the exact pixel output
    cv2.namedWindow("Mask Scalpel", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Mask Scalpel", click_event)

    index = 0
    total = len(label_files)

    while True:
        label_filename = label_files[index]
        img_filename = label_filename.replace('.txt', '.jpg')
        img_path = os.path.join(img_dir, img_filename)
        label_path = os.path.join(label_dir, label_filename)

        if not os.path.exists(img_path):
            index = (index + 1) % total
            continue

        base_img = cv2.imread(img_path)
        img_h, img_w = base_img.shape[:2]
        img_w_global = img_w 

        # Reset camera position for the new image
        scale = 1.0
        # Automatically fit the side-by-side image into the viewport height
        scale = VIEW_H / img_h 
        pan_x, pan_y = 0, 0

        # Load labels
        current_lines = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                current_lines = [line.strip() for line in f.readlines() if line.strip()]
            
        current_polygons = []
        for line in current_lines:
            parts = line.split()
            coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            coords[:, 0] *= img_w
            coords[:, 1] *= img_h
            current_polygons.append(np.int32(coords))

        redraw_needed = True

        while True:
            if redraw_needed:
                # 1. Render Left Image
                raw_display = base_img.copy()
                cv2.putText(raw_display, "RAW REFERENCE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # 2. Render Right Image (Annotated)
                annotated_display = base_img.copy()
                overlay = base_img.copy()
                for i, poly in enumerate(current_polygons):
                    class_id = int(current_lines[i].split()[0])
                    color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
                    cv2.fillPoly(overlay, [poly], color=color)
                    cv2.polylines(annotated_display, [poly], isClosed=True, color=(255, 255, 255), thickness=1)
                cv2.addWeighted(overlay, 0.4, annotated_display, 0.6, 0, annotated_display)
                
                cv2.putText(annotated_display, f"[{index+1}/{total}] {img_filename}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(annotated_display, "Scroll: Zoom | R-Click: Pan | L-Click: Delete", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(annotated_display, "'S': Save | 'D': Next | 'A': Prev", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # 3. Stitch together
                combined_display = np.hstack((raw_display, annotated_display))
                comb_h, comb_w = combined_display.shape[:2]

                # 4. Camera Viewport Engine
                crop_w = int(VIEW_W / scale)
                crop_h = int(VIEW_H / scale)

                # Clamp panning so we don't drag the camera off into the void
                pan_x = max(0, min(pan_x, comb_w - crop_w))
                pan_y = max(0, min(pan_y, comb_h - crop_h))

                # Extract the crop and resize to fit our computer screen
                roi = combined_display[pan_y : pan_y + crop_h, pan_x : pan_x + crop_w]
                
                try:
                    final_view = cv2.resize(roi, (VIEW_W, VIEW_H))
                    cv2.imshow("Mask Scalpel", final_view)
                except Exception as e:
                    pass # Failsafe for extreme zoom values hitting floating point limits

                redraw_needed = False

            key = cv2.waitKey(15) & 0xFF
            
            if key == ord('s'):
                with open(label_path, 'w') as f:
                    f.write("\n".join(current_lines))
                print(f"[+] Saved {label_filename}")
                redraw_needed = True
            elif key == ord('d'):
                index = (index + 1) % total
                break
            elif key == ord('a'):
                index = (index - 1) % total
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return
            # Keyboard fallbacks for zooming (if your OS blocks the mouse wheel)
            elif key == ord('+') or key == ord('='):
                scale = min(scale * 1.15, 10.0)
                redraw_needed = True
            elif key == ord('-') or key == ord('_'):
                scale = max(scale * 0.85, 0.2)
                redraw_needed = True
            

if __name__ == '__main__':
    clean_dataset_interactively(
        img_dir=r"DATASETS FOR INSTANCE SEGMENTATION\UMT_dataset\UMT images for annotation V1 CLEANED",
        label_dir=r"DATASETS FOR INSTANCE SEGMENTATION\UMT_dataset\DINO+SAM_filtered_labels"
    )