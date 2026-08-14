import os
import cv2
import math
import xml.etree.ElementTree as ET

def is_on_edge(px, py, xtl, ytl, xbr, ybr, tol=1.5):
    """
    Checks if a point (px, py) lies on the border of a bounding box.
    Uses a small tolerance (tol) in pixels to account for floating point rounding.
    """
    on_left = abs(px - xtl) <= tol
    on_right = abs(px - xbr) <= tol
    on_top = abs(py - ytl) <= tol
    on_bottom = abs(py - ybr) <= tol
    
    return on_left or on_right or on_top or on_bottom

def audit_unadjusted_annotations():
    # ==========================================
    # --- CONFIGURATION ---
    # ==========================================
    CVAT_XML_PATH = r"DATASETS\UTM_dataset\UTM Dataset V1 Refixed\annotations.xml"
    IMG_DIR = r"DATASETS\UTM_dataset\UTM images for annotation V1 CLEANED"
    OUTPUT_DEBUG_DIR = r"DATASETS\UTM_dataset\Images to be relabelled"
    OUTPUT_INDEX_FILE = r"DATASETS\UTM_dataset\relabel_indices.txt"
    
    # Tolerance in pixels. If a point is within 1.0 pixels of the bounding box edge, 
    # it is considered "unadjusted"
    EDGE_TOLERANCE = 1.0 
    # ==========================================

    os.makedirs(OUTPUT_DEBUG_DIR, exist_ok=True)
    print(f"Scanning XML for unadjusted pre-annotations: {CVAT_XML_PATH}...\n")
    
    tree = ET.parse(CVAT_XML_PATH)
    root = tree.getroot()
    
    bad_images_list = []
    total_unadjusted_fish = 0

    for image_elem in root.findall('image'):
        img_id = image_elem.attrib['id']
        img_name = image_elem.attrib['name']
        
        # Map boxes and skeletons by group_id
        groups = {}
        for box in image_elem.findall('box'):
            g_id = box.attrib.get('group_id')
            if g_id: groups.setdefault(g_id, {})['box'] = box
                
        for skeleton in image_elem.findall('skeleton'):
            g_id = skeleton.attrib.get('group_id')
            if g_id: groups.setdefault(g_id, {})['skeleton'] = skeleton
                
        if not groups:
            continue
            
        needs_relabel = False
        fish_draw_data = [] # Store data to draw later if needed
        
        for g_id, data in groups.items():
            if 'box' not in data or 'skeleton' not in data:
                continue 
                
            box = data['box']
            skeleton = data['skeleton']
            
            xtl, ytl = float(box.attrib['xtl']), float(box.attrib['ytl'])
            xbr, ybr = float(box.attrib['xbr']), float(box.attrib['ybr'])
            
            # Find Head and Tail points
            hx, hy, tx, ty = None, None, None, None
            for pt in skeleton.findall('points'):
                label = pt.attrib['label']
                px, py = float(pt.attrib['points'].split(',')[0]), float(pt.attrib['points'].split(',')[1])
                
                if label == "Head":
                    hx, hy = px, py
                elif label == "Tail":
                    tx, ty = px, py
                    
            if hx is not None and tx is not None:
                # --- THE MATHEMATICAL CHECK ---
                head_on_edge = is_on_edge(hx, hy, xtl, ytl, xbr, ybr, tol=EDGE_TOLERANCE)
                tail_on_edge = is_on_edge(tx, ty, xtl, ytl, xbr, ybr, tol=EDGE_TOLERANCE)
                
                # If BOTH head and tail are sitting on the bounding box lines:
                is_unadjusted = head_on_edge and tail_on_edge
                
                if is_unadjusted:
                    needs_relabel = True
                    total_unadjusted_fish += 1
                    
                fish_draw_data.append({
                    "bbox": (int(xtl), int(ytl), int(xbr), int(ybr)),
                    "head": (int(hx), int(hy)),
                    "tail": (int(tx), int(ty)),
                    "is_unadjusted": is_unadjusted
                })

        # --- VISUALIZATION OUTPUT ---
        # If this image has at least one unadjusted fish, draw and save it
        if needs_relabel:
            bad_images_list.append(f"[{img_id}] | {img_name}")
            
            img_path = os.path.join(IMG_DIR, img_name)
            img = cv2.imread(img_path)
            
            if img is not None:
                for fish in fish_draw_data:
                    x1, y1, x2, y2 = fish["bbox"]
                    hx, hy = fish["head"]
                    tx, ty = fish["tail"]
                    
                    # Highlight unadjusted fish in thick RED, adjusted fish in thin GREEN
                    if fish["is_unadjusted"]:
                        color = (0, 0, 255) # BGR Red
                        thickness = 3
                        cv2.putText(img, "FIX ME", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else:
                        color = (0, 255, 0) # BGR Green
                        thickness = 1
                        
                    # Draw Box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                    # Draw Head and Tail
                    cv2.circle(img, (hx, hy), 4, (0, 255, 255), -1) # Yellow dots
                    cv2.circle(img, (tx, ty), 4, (0, 255, 255), -1)
                    
                save_path = os.path.join(OUTPUT_DEBUG_DIR, img_name)
                cv2.imwrite(save_path, img)

    # --- SAVE THE TEXT FILE ---
    with open(OUTPUT_INDEX_FILE, 'w') as f:
        f.write("--------------------------------------------------\n")
        f.write("XML ID     | File Name\n")
        f.write("--------------------------------------------------\n")
        f.write("\n".join(bad_images_list))

    print("Audit Complete!")
    print(f"Total unadjusted fish found: {total_unadjusted_fish}")
    print(f"Generated index file: {OUTPUT_INDEX_FILE}")
    print(f"Check '{OUTPUT_DEBUG_DIR}' to see the highlighted individuals.")

if __name__ == '__main__':
    audit_unadjusted_annotations()