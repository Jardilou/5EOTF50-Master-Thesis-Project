import xml.etree.ElementTree as ET
import os
import cv2

def process_cvat_xml(input_xml, output_xml, images_folder, visual_check_folder):
    print(f"Loading '{input_xml}'...")
    
    try:
        tree = ET.parse(input_xml)
        root = tree.getroot()
    except Exception as e:
        print(f"Failed to load XML: {e}")
        return

    # 1. Group images by their prefix (e.g., "Left_Image_Quadrat 2.5m.MOV-1")
    prefix_map = {}
    for image in root.findall('image'):
        name = image.get('name', '')
        
        if 'solo_channel' in name:
            prefix = name.split('-solo_channel')[0]
            if prefix not in prefix_map:
                prefix_map[prefix] = {}
            prefix_map[prefix]['solo'] = image
            
        elif 'unchanged' in name:
            prefix = name.split('-unchanged')[0]
            if prefix not in prefix_map:
                prefix_map[prefix] = {}
            prefix_map[prefix]['unchanged'] = image

    # Ensure output folder for visual checks exists
    if images_folder and visual_check_folder:
        os.makedirs(visual_check_folder, exist_ok=True)

    # 2. Process matched pairs
    for prefix, nodes in prefix_map.items():
        solo_img = nodes.get('solo')
        unchanged_img = nodes.get('unchanged')
        
        if solo_img is not None and unchanged_img is not None:
            # Transfer all annotations (children nodes) from solo to unchanged
            for child in list(solo_img):
                unchanged_img.append(child)
            
            # Remove the trailing underscore before the extension 
            # e.g., "unchanged_.jpg" -> "unchanged.jpg"
            old_name = unchanged_img.get('name')
            new_name = old_name.replace('unchanged_.', 'unchanged.')
            unchanged_img.set('name', new_name)
            
            # Remove the solo_channel image element entirely
            root.remove(solo_img)

            # --- VISUALIZATION BLOCK ---
            if images_folder and visual_check_folder:
                # The physical file might still be named with or without the '_'
                img_path_old = os.path.join(images_folder, old_name)
                img_path_new = os.path.join(images_folder, new_name)
                target_path = img_path_new if os.path.exists(img_path_new) else img_path_old
                
                if os.path.exists(target_path):
                    img_data = cv2.imread(target_path)
                    if img_data is not None:
                        # Find all skeleton points and draw them
                        for skel in unchanged_img.findall('skeleton'):
                            for pt in skel.findall('points'):
                                coords = pt.get('points')
                                if coords:
                                    x, y = map(float, coords.split(','))
                                    label = pt.get('label', '')
                                    
                                    # Draw red point
                                    cv2.circle(img_data, (int(x), int(y)), 4, (0, 0, 255), -1)
                                    # Draw green label text
                                    cv2.putText(img_data, label, (int(x) + 5, int(y) - 5), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        
                        save_path = os.path.join(visual_check_folder, f"CHECK_{new_name}")
                        cv2.imwrite(save_path, img_data)
                    else:
                        print(f"Could not read image: {target_path}")
                else:
                    print(f"Warning: Image not found for visualization -> {target_path}")

    # 3. Re-index IDs and Update Meta-Data
    # CVAT expects sequential IDs and accurate frame counts for jobs
    all_images = root.findall('image')
    img_count = len(all_images)
    
    for idx, img in enumerate(all_images):
        img.set('id', str(idx))
        
    for tag in ['.//meta/job/size', './/meta/job/stop_frame', './/meta/job/segments/segment/stop']:
        elem = root.find(tag)
        if elem is not None:
            # Stop frames are 0-indexed (Total - 1), Size is 1-indexed (Total)
            elem.text = str(img_count - 1) if 'stop' in tag else str(img_count)

    # 4. Save Final XML
    tree.write(output_xml, encoding='utf-8', xml_declaration=True)
    print("--- Processing Complete ---")
    print(f"Images remaining: {img_count}")
    print(f"Saved new XML to: {output_xml}")
    if visual_check_folder:
        print(f"Saved visual checks to: {visual_check_folder}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Put your actual file paths here
    INPUT_XML = r"XML ANNOTATIONS\Intermediary-dataset-calibration-images.xml"
    OUTPUT_XML = r"XML ANNOTATIONS\final-calibration-dataset-annotations.xml"
    
    # Folder where your raw/unchanged images are currently located
    IMAGES_FOLDER = r"DATASETS\DATASET FOR STEREO CALIBRATION\Calibration images\Stereo Video Calibration STEREO-DOVs\Original+Processed_Calibration_Images" 
    
    # Folder where you want the script to spit out the visualized checks
    VISUAL_CHECK_FOLDER = r"DATASETS\DATASET FOR STEREO CALIBRATION\Calibration images\Stereo Video Calibration STEREO-DOVs\visual_checks" 
    
    process_cvat_xml(INPUT_XML, OUTPUT_XML, IMAGES_FOLDER, VISUAL_CHECK_FOLDER)