import xml.etree.ElementTree as ET
import json
import os

def convert_cvat_xml_to_json(xml_path, json_output_path, image_dir):
    if not os.path.exists(xml_path):
        print(f"Error: Could not find XML file at {xml_path}")
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    left_images = {}
    right_images = {}
    
    for image_tag in root.findall('image'):
        filename = image_tag.get('name')
        
        # --- THE FIX: We append the full path so cv2.imread never fails ---
        full_image_path = os.path.join(image_dir, filename)
        
        # Normal Left/Right assignments (No Swapping!)
        if "Left_Image_" in filename:
            is_left = True
            base_name = filename.replace("Left_Image_", "")
        elif "Right_Image_" in filename:
            is_left = False
            base_name = filename.replace("Right_Image_", "")
        else:
            continue
            
        points_dict = {}
        for pt_tag in image_tag.findall('.//points'):
            label = pt_tag.get('label')   
            coords = pt_tag.get('points') 
            if not label or not coords or not label.startswith('R'): continue
                
            x_str, y_str = coords.split(',')
            points_dict[label] = [float(x_str), float(y_str)]
            
        if len(points_dict) > 0:
            if is_left:
                left_images[base_name] = {"path": full_image_path, "points": points_dict}
            else:
                right_images[base_name] = {"path": full_image_path, "points": points_dict}

    # Pair them up into the JSON structure
    common_bases = sorted(list(set(left_images.keys()) & set(right_images.keys())))
    json_list = []
    
    for base in common_bases:
        pair_data = {
            "left": left_images[base]["path"],
            "right": right_images[base]["path"],
            "left_points": left_images[base]["points"],
            "right_points": right_images[base]["points"]
        }
        json_list.append(pair_data)
        
    with open(json_output_path, 'w') as f:
        json.dump(json_list, f, indent=4)
        
    print(f"SUCCESS: Converted {len(json_list)} stereo pairs with FULL paths attached.")

# --- EXECUTE ---
XML_FILE = r"XML ANNOTATIONS\final-calibration-dataset-annotations.xml" 
JSON_FILE = "calibration_annotations.json"


IMAGE_DIR = r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\DATASETS\DATASET FOR STEREO CALIBRATION\Calibration images\Stereo Video Calibration STEREO-DOVs\All_Original_Calibration_Images"

convert_cvat_xml_to_json(XML_FILE, JSON_FILE, IMAGE_DIR)