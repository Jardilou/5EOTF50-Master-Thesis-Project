import xml.etree.ElementTree as ET
import json
import os

def convert_cvat_xml_to_json(xml_path, json_output_path):
    """
    Parses the CVAT XML calibration file and exports a JSON list 
    formatted specifically for the stereo_fish_3d.py pipeline.
    """
    if not os.path.exists(xml_path):
        print(f"Error: Could not find XML file at {xml_path}")
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    left_images = {}
    right_images = {}
    
    # 1. Parse all images and points
    for image_tag in root.findall('image'):
        filename = image_tag.get('name')
        
        # Determine if left or right and extract the common base name
        if "Left_Image_" in filename:
            is_left = True
            base_name = filename.replace("Left_Image_", "")
        elif "Right_Image_" in filename:
            is_left = False
            base_name = filename.replace("Right_Image_", "")
        else:
            continue
            
        points_dict = {}
        
        # Extract all <points> tags (CVAT format)
        for pt_tag in image_tag.findall('.//points'):
            label = pt_tag.get('label')   # e.g., "R4C2"
            coords = pt_tag.get('points') # e.g., "877.59,754.27"
            
            # Ensure it's a grid point
            if not label or not coords or not label.startswith('R'):
                continue
                
            x_str, y_str = coords.split(',')
            points_dict[label] = [float(x_str), float(y_str)]
            
        # Store in temp dictionaries if it contains points
        if len(points_dict) > 0:
            if is_left:
                left_images[base_name] = {"path": filename, "points": points_dict}
            else:
                right_images[base_name] = {"path": filename, "points": points_dict}

    # 2. Pair them up into the JSON structure
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
        
    # 3. Export to JSON
    with open(json_output_path, 'w') as f:
        json.dump(json_list, f, indent=4)
        
    print(f"SUCCESS: Converted {len(json_list)} stereo pairs.")
    print(f"Saved to: {json_output_path}")

# --- EXECUTE CONVERSION ---
# Make sure to point this to your actual XML file
xml_file = r"XML ANNOTATIONS\final-calibration-dataset-annotations.xml" 
json_file = "calibration_annotations.json"

convert_cvat_xml_to_json(xml_file, json_file)