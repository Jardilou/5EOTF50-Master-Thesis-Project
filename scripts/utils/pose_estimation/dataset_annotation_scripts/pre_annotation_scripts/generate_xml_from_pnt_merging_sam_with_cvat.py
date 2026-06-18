import json
import cv2
import os
import xml.etree.ElementTree as ET
from ultralytics import SAM

def merge_sam_with_cvat():
    # ==========================================
    # --- 1. CONFIGURATION ---
    # ==========================================
    # ⚠️ YOUR 5 MANUAL FRAMES GO HERE ⚠️
    # The script will protect these and overwrite everything else!
    PROTECTED_FRAMES = [
        "St9_75-100m(R).MOV-14.jpg",
        "St9_75-100m(R).MOV-2.jpg", 
        "St9_75-100m(R).MOV-21.jpg",
        "St9_75-100m(R).MOV-34.jpg",
        "St9_75-100m(R).MOV-48.jpg",
        "St9_75-100m(R).MOV-5.jpg"
    ]

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
    
    CVAT_EXPORT_XML = r"TRAINTEST STEPS\OTHER STEPS FOR DATA PREPARATION\UTM_5_Frames_annotations.xml" 
    OUTPUT_XML = "Merged_Annotations.xml"
    
    MODEL_PATH = "sam_b.pt" 
    RATIO_THRESHOLD = 2.0
    POINTS_4 = ["Head", "Mid1", "Mid2", "Tail"] 
    POINTS_2 = ["Head", "Tail"]
    # ==========================================

    print("Initializing Pipeline and loading SAM Foundation Model...")
    model = SAM(MODEL_PATH) 
    
    # 1. Map all images and their paths from the .pnt files
    img_name_to_path = {}
    img_name_to_pnt_data = {}
    
    for pnt_file in PNT_FILES:
        if not os.path.exists(pnt_file): continue
        img_dir = os.path.dirname(pnt_file) 
        
        with open(pnt_file, 'r') as f:
            pnt_data = json.load(f)
            
        for img_name, species_dict in pnt_data.get("points", {}).items():
            if species_dict: 
                img_name_to_pnt_data[img_name] = species_dict
                img_name_to_path[img_name] = os.path.join(img_dir, img_name)

    # 2. Parse the existing CVAT XML safely
    print(f"Parsing existing CVAT XML: {CVAT_EXPORT_XML}...")
    tree = ET.parse(CVAT_EXPORT_XML)
    root = tree.getroot()
    
    # Find the highest existing group_id so we don't break your manual links
    max_group_id = 0
    for elem in root.iter():
        if 'group_id' in elem.attrib:
            max_group_id = max(max_group_id, int(elem.attrib['group_id']))
            
    global_group_id = max_group_id + 1
    
    manual_kept = 0
    sam_injected = 0
    annotations_wiped = 0
    
    # 3. Iterate over the <image> tags
    for image_elem in root.findall('image'):
        img_name = image_elem.attrib['name']
        
        # --- NEW PROTECTION LOGIC ---
        if img_name in PROTECTED_FRAMES:
            # You perfected this frame. Leave it completely untouched.
            manual_kept += 1
            continue
        else:
            # This is an unadjusted frame. Wipe the old, rough annotations out of the XML.
            elements_to_remove = [child for child in image_elem if child.tag in ['box', 'skeleton', 'polygon', 'points']]
            for elem in elements_to_remove:
                image_elem.remove(elem)
            if elements_to_remove:
                annotations_wiped += 1

        # Now that the slate is clean, generate the new SAM annotations
        if img_name in img_name_to_pnt_data:
            src_img_path = img_name_to_path[img_name]
            if not os.path.exists(src_img_path): continue
            
            species_dict = img_name_to_pnt_data[img_name]
            
            for species_name, points_list in species_dict.items():
                for pt in points_list:
                    px, py = pt["x"], pt["y"]
                    
                    # Run SAM using the exact pixel coordinate
                    results = model.predict(source=src_img_path, points=[[px, py]], labels=[1], verbose=False)
                    
                    if len(results[0].boxes) > 0:
                        x1, y1, x2, y2 = results[0].boxes.xyxy[0].cpu().numpy()
                    else:
                        x1, y1 = px - 50, py - 50
                        x2, y2 = px + 50, py + 50
                        
                    box_width = x2 - x1
                    box_height = y2 - y1
                    ratio = box_width / (box_height + 1e-6)

                    # Build the <box> Element
                    box_elem = ET.SubElement(image_elem, 'box')
                    box_elem.set('label', 'fish-box')
                    box_elem.set('xtl', f"{x1:.2f}")
                    box_elem.set('ytl', f"{y1:.2f}")
                    box_elem.set('xbr', f"{x2:.2f}")
                    box_elem.set('ybr', f"{y2:.2f}")
                    box_elem.set('z_order', '0')
                    box_elem.set('group_id', str(global_group_id))
                    
                    attr_elem = ET.SubElement(box_elem, 'attribute')
                    attr_elem.set('name', 'species')
                    attr_elem.text = species_name
                    
                    # Calculate Skeleton Strategy
                    if ratio > RATIO_THRESHOLD:
                        label = "fish-curve"
                        pt_names = POINTS_4
                        y_center = y1 + (box_height / 2)
                        coords = [(x1, y_center), (x1 + box_width * 0.33, y_center), (x1 + box_width * 0.66, y_center), (x2, y_center)]
                    else:
                        label = "stiff-fish-curve"
                        pt_names = POINTS_2
                        y_center = y1 + (box_height / 2)
                        coords = [(x1, y_center), (x2, y_center)]
                        
                    # Build the <skeleton> Element
                    skel_elem = ET.SubElement(image_elem, 'skeleton')
                    skel_elem.set('label', label)
                    skel_elem.set('z_order', '0')
                    skel_elem.set('group_id', str(global_group_id))
                    
                    skel_attr = ET.SubElement(skel_elem, 'attribute')
                    skel_attr.set('name', 'species')
                    skel_attr.text = species_name
                    
                    for pt_name, (cx, cy) in zip(pt_names, coords):
                        pt_elem = ET.SubElement(skel_elem, 'points')
                        pt_elem.set('label', pt_name)
                        pt_elem.set('points', f"{cx:.2f},{cy:.2f}")
                        pt_elem.set('outside', '0')
                        pt_elem.set('occluded', '0')

                    global_group_id += 1 
                    sam_injected += 1

    # Save the injected tree
    tree.write(OUTPUT_XML, encoding='utf-8', xml_declaration=True)
        
    print(f"\nSmart Injection Complete!")
    print(f"Protected manual frames: {manual_kept}")
    print(f"Old annotations wiped from unadjusted frames: {annotations_wiped}")
    print(f"SAM Foundation Model injections added: {sam_injected}")
    print(f"Upload this file back to CVAT: {OUTPUT_XML}")

if __name__ == '__main__':
    merge_sam_with_cvat()