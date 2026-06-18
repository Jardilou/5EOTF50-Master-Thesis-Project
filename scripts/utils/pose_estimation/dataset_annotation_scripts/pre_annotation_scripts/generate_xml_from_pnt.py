import json
import cv2
import os
from ultralytics import YOLO

def generate_cvat_xml():
    # ==========================================
    # --- CONFIGURATION ---
    # ==========================================
    PNT_FILE = r'DATASETS\UTM_dataset\STEREO IMAGES TO ANALYZE\Images 1-190714\St3-Pantai Vietnam\St3_Pantai Vietnam.pnt'               # Your ground-truth species file
    IMG_DIR = r'DATASETS\UTM_dataset\STEREO IMAGES TO ANALYZE\Images 1-190714\St3-Pantai Vietnam\TG4-Red(R) Images st3 (P.V)' # Folder containing the wild images
    MODEL_PATH = r'ultralytics\runs\pose\BezierFusion\eamrf_training_run17\weights\best.pt'                # Your current YOLO weights
    OUTPUT_XML = r'CVAT_Annotations_St3-Pantai Vietnam_TG4-Red.xml'
    
    RATIO_THRESHOLD = 2.0                 # If Width / Height > 2.0, use 4-point skeleton
    
    # ⚠️ CRITICAL: These must exactly match the point names you defined in CVAT!
    POINTS_4 = ["Head", "Mid1", "Mid2", "Tail"] 
    POINTS_2 = ["Head", "Tail"]
    # ==========================================

    print("🚀 Loading YOLO model and .pnt data...")
    model = YOLO(MODEL_PATH)
    
    with open(PNT_FILE, 'r') as f:
        pnt_data = json.load(f)

    # Initialize CVAT XML 1.1 Structure
    xml_lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<annotations>',
        '  <version>1.1</version>'
    ]

    img_id = 0
    for img_name, species_dict in pnt_data["points"].items():
        img_path = os.path.join(IMG_DIR, img_name)
        
        # Get image dimensions (required by CVAT XML)
        if not os.path.exists(img_path):
            print(f"⚠️ Skipping {img_name} - Image not found.")
            continue
            
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        xml_lines.append(f'  <image id="{img_id}" name="{img_name}" width="{w}" height="{h}">')
        
        # Run YOLO on the wild image
        results = model.predict(source=img_path, conf=0.25, verbose=False)
        yolo_boxes = results[0].boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
        
        # Process every ground-truth point in the .pnt file
        for species_name, points_list in species_dict.items():
            for pt in points_list:
                px, py = pt["x"], pt["y"]
                
                # Check if this point falls inside ANY YOLO box
                matched_box = None
                for box in yolo_boxes:
                    x1, y1, x2, y2 = box
                    if x1 <= px <= x2 and y1 <= py <= y2:
                        matched_box = box
                        break
                
                # If YOLO found it, use YOLO's box. If YOLO missed it, make a Seed Box!
                if matched_box is not None:
                    x1, y1, x2, y2 = matched_box
                else:
                    # YOLO completely missed the wild fish. Create a 100x100 box around the point.
                    x1, y1 = px - 50, py - 50
                    x2, y2 = px + 50, py + 50
                
                box_width = x2 - x1
                box_height = y2 - y1
                ratio = box_width / (box_height + 1e-6) # Prevent division by zero
                
                # Determine strategy based on ratio
                if ratio > RATIO_THRESHOLD:
                    label = "fish-curve"
                    pt_names = POINTS_4
                    # Space 4 points evenly across the horizontal center of the box
                    y_center = y1 + (box_height / 2)
                    coords = [
                        (x1, y_center), 
                        (x1 + box_width * 0.33, y_center), 
                        (x1 + box_width * 0.66, y_center), 
                        (x2, y_center)
                    ]
                else:
                    label = "stiff-fish-curve"
                    pt_names = POINTS_2
                    # Space 2 points (head to tail)
                    y_center = y1 + (box_height / 2)
                    coords = [
                        (x1, y_center), 
                        (x2, y_center)
                    ]
                
                # Write the CVAT Skeleton XML block
                xml_lines.append(f'    <skeleton label="{label}" z_order="0">')
                # Add the species name as an attribute!
                xml_lines.append(f'      <attribute name="species">{species_name}</attribute>')
                
                for pt_name, (cx, cy) in zip(pt_names, coords):
                    xml_lines.append(f'      <points label="{pt_name}" points="{cx:.2f},{cy:.2f}" outside="0" occluded="0"/>')
                
                xml_lines.append('    </skeleton>')
                
        xml_lines.append('  </image>')
        img_id += 1

    xml_lines.append('</annotations>')

    # Save to file
    with open(OUTPUT_XML, 'w', encoding='utf-8') as f:
        f.write('\n'.join(xml_lines))
        
    print(f"\n✅ Success! Generated {OUTPUT_XML} ready for CVAT.")

if __name__ == '__main__':
    generate_cvat_xml()