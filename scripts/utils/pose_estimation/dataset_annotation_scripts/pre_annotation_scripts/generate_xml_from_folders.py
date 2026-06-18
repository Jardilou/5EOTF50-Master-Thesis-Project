import json
import cv2
import os
import shutil
from ultralytics import YOLO

def process_utm_datasets():
    # ==========================================
    # --- 1. CONFIGURATION ---
    # ==========================================
    # Just paste the absolute paths to your .pnt files here.
    # The script assumes the raw images are in the exact same folder as the .pnt file!
    PNT_FILES = [
        r"DATASETS\UTM_dataset\STEREO IMAGES TO ANALYZE\Images 1-190714\St1-Pantai Pasir Cina\TG4-Red(R) Images st1\St1_Pantai Pasir Cina 1.pnt",
        r"DATASETS\UTM_dataset\STEREO IMAGES TO ANALYZE\Images 1-190714\St2-Pantai Pasir Cina\TG4-Black(L) Images st2 (P.C.2)\St2_Pantai Pasir Cina 2.pnt",
        r"DATASETS\UTM_dataset\STEREO IMAGES TO ANALYZE\Images 1-190714\St3-Pantai Vietnam\TG4-Red(R) Images st3 (P.V)\St3_Pantai Vietnam.pnt",
        r"DATASETS\UTM_dataset\STEREO IMAGES TO ANALYZE\Images 2-190715\St4-Karang Tengah\TG4 Black(L) images st4 (K.T)\St4_Karang Tengah.pnt",
        r"DATASETS\UTM_dataset\STEREO IMAGES TO ANALYZE\Images 2-190715\St6-Batu Menangis\TG4 Red(R) images st6 (B.M)\St6_Batu Menangis.pnt",
        r"DATASETS\UTM_dataset\STEREO IMAGES TO ANALYZE\Images 3-190716\St7-Teluk Air\TG4 Black(L) images st7 (T.A)\St7_Teluk Air.pnt",
        r"DATASETS\UTM_dataset\STEREO IMAGES TO ANALYZE\Images 3-190716\St8-Vietnamese Jetty (R)\TG4 Black(L) images st8 (V.J)\St8_Vietnamese Jetty.pnt",
        r"DATASETS\UTM_dataset\STEREO IMAGES TO ANALYZE\Images 4-190717\St9-Geluk West\TG4 Red(R) images st9 (P.G)\St9_Geluk West.pnt",
        r"DATASETS\UTM_dataset\STEREO IMAGES TO ANALYZE\Images 4-190717\St10-Batu Payung\TG4 Red(R) images st10 (B.P)\St10_Batu Payung.pnt",
        r"DATASETS\UTM_dataset\STEREO IMAGES TO ANALYZE\Images 5-190722\St11-Christmas Garden\TG4 Black(L) images st11 (C.G)\St11_Christmas Garden.pnt",
        r"DATASETS\UTM_dataset\STEREO IMAGES TO ANALYZE\Images 5-190722\St12-Dinding Laut\TG4 Black(L) images st12 (D.L)\St12_Dinding Laut.pnt",
        r"DATASETS\UTM_dataset\STEREO IMAGES TO ANALYZE\Images 6-190723\St13-Geluk East\TG4 Black(L) images st13\St13_Geluk East.pnt",
        r"DATASETS\UTM_dataset\STEREO IMAGES TO ANALYZE\Images 6-190723\St14-Tengkorak West\TG4 Black(L) Images st14\St14_Tengkorak West.pnt",
        r"DATASETS\UTM_dataset\STEREO IMAGES TO ANALYZE\Images 7-190724\St15-Pulau Karah\TG4 Red(R) Images St15\St15_Pulau Karah.pnt",
        r"DATASETS\UTM_dataset\STEREO IMAGES TO ANALYZE\Images 7-190724\St16-Tengkorak East\TG4 Black(L) Images St16\St16_Tengkorak East.pnt"
    ]
    
    OUTPUT_IMG_DIR = "UTM images for annotation V1"
    OUTPUT_XML = "UTM_annotations.xml"
    MODEL_PATH = r"ultralytics\runs\pose\BezierFusion\eamrf_training_run17\weights\best.pt" # Point this to your trained YOLO weights
    RATIO_THRESHOLD = 2.0
    
    # MUST match the specific point names you created in CVAT!
    POINTS_4 = ["Head", "Mid1", "Mid2", "Tail"] 
    POINTS_2 = ["Head", "Tail"]
    # ==========================================

    print("🚀 Initializing Pipeline and loading YOLO model...")
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)
    
    # Start the Master CVAT XML file
    xml_lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<annotations>',
        '  <version>1.1</version>'
    ]
    
    global_img_id = 0
    global_group_id = 1 # NEW: This links the box and the skeleton together!
    copied_images_count = 0
    
    # Loop through every .pnt file in your list
    for pnt_file in PNT_FILES:
        print(f"\n📂 Scanning dataset: {os.path.basename(pnt_file)}...")
        
        if not os.path.exists(pnt_file):
            print(f"⚠️ Warning: {pnt_file} not found. Skipping.")
            continue
            
        # Automatically grab the folder where the .pnt file lives
        img_dir = os.path.dirname(pnt_file) 
            
        with open(pnt_file, 'r') as f:
            pnt_data = json.load(f)
            
        # Loop through every image referenced in the .pnt file
        for img_name, species_dict in pnt_data.get("points", {}).items():
            
            # 1. FILTER: If the dictionary is empty, skip this image entirely!
            if not species_dict:
                continue
                
            src_img_path = os.path.join(img_dir, img_name)
            dst_img_path = os.path.join(OUTPUT_IMG_DIR, img_name)
            
            if not os.path.exists(src_img_path):
                print(f"⚠️ Image missing from disk: {src_img_path}")
                continue
            
            # 2. CONSOLIDATE: Copy the annotated image to the new UTM folder
            shutil.copy(src_img_path, dst_img_path)
            copied_images_count += 1
            
            # 3. ANALYZE: Read dimensions for CVAT XML
            img = cv2.imread(dst_img_path)
            h, w, _ = img.shape
            
            xml_lines.append(f'  <image id="{global_img_id}" name="{img_name}" width="{w}" height="{h}">')
            
            # 4. PREDICT: Run YOLO on the image
            results = model.predict(source=dst_img_path, conf=0.25, verbose=False)
            yolo_boxes = results[0].boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
            
            # 5. MATCH & DRAW: Process the ground-truth species points
            for species_name, points_list in species_dict.items():
                for pt in points_list:
                    px, py = pt["x"], pt["y"]
                    
                    # Check if the wild fish point falls inside a YOLO box
                    matched_box = None
                    for box in yolo_boxes:
                        x1, y1, x2, y2 = box
                        if x1 <= px <= x2 and y1 <= py <= y2:
                            matched_box = box
                            break
                            
                    # Use YOLO box if found, otherwise generate a 100x100 Seed Box
                    if matched_box is not None:
                        x1, y1, x2, y2 = matched_box
                    else:
                        x1, y1 = px - 50, py - 50
                        x2, y2 = px + 50, py + 50
                        
                    box_width = x2 - x1
                    box_height = y2 - y1
                    ratio = box_width / (box_height + 1e-6)

                    # --- NEW: Output an independent, resizable Bounding Box ---
                    xml_lines.append(f'    <box label="fish-box" xtl="{x1:.2f}" ytl="{y1:.2f}" xbr="{x2:.2f}" ybr="{y2:.2f}" z_order="0" group_id="{global_group_id}">')
                    xml_lines.append(f'      <attribute name="species">{species_name}</attribute>')
                    xml_lines.append('    </box>')
                    
                    # Route to correct Skeleton Strategy based on Length/Height ratio
                    if ratio > RATIO_THRESHOLD:
                        label = "fish-curve"
                        pt_names = POINTS_4
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
                        y_center = y1 + (box_height / 2)
                        coords = [
                            (x1, y_center), 
                            (x2, y_center)
                        ]
                        
                    # Build the XML block for this specific fish with GROUP_ID
                    xml_lines.append(f'    <skeleton label="{label}" z_order="0" group_id="{global_group_id}">')
                    xml_lines.append(f'      <attribute name="species">{species_name}</attribute>')
                    for pt_name, (cx, cy) in zip(pt_names, coords):
                        xml_lines.append(f'      <points label="{pt_name}" points="{cx:.2f},{cy:.2f}" outside="0" occluded="0"/>')
                    xml_lines.append('    </skeleton>')

                    global_group_id += 1 # Increment link ID for the next fish
                    
            xml_lines.append('  </image>')
            global_img_id += 1 # Increment CVAT's global image counter
            
    xml_lines.append('</annotations>')
    
    # Save the Master XML File
    with open(OUTPUT_XML, 'w', encoding='utf-8') as f:
        f.write('\n'.join(xml_lines))
        
    print(f"\n✅ UTM Pipeline Complete!")
    print(f"📸 Total annotated images consolidated: {copied_images_count}")
    print(f"📂 Images safely copied to: {OUTPUT_IMG_DIR}")
    print(f"📄 Master CVAT XML saved as: {OUTPUT_XML}")

if __name__ == '__main__':
    process_utm_datasets()