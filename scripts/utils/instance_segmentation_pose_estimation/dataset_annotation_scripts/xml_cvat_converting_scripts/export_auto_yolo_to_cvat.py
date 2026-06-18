import os
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
import copy

def prettify(elem):
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def load_original_data(xml_path):
    if not os.path.exists(xml_path):
        return {}
        
    tree = ET.parse(xml_path)
    root = tree.getroot()
    original_data = {}
    
    for image in root.findall('image'):
        raw_basename = os.path.splitext(os.path.basename(image.attrib['name']))[0]
        img_data = {'boxes': [], 'metadata': []}
        
        for element in image:
            if element.tag == 'box':
                xtl, ytl = float(element.attrib['xtl']), float(element.attrib['ytl'])
                xbr, ybr = float(element.attrib['xbr']), float(element.attrib['ybr'])
                cx, cy = (xtl + xbr) / 2, (ytl + ybr) / 2
                
                box_label = element.attrib.get('label', 'measurable_fish')
                
                img_data['boxes'].append({
                    'cx': cx, 'cy': cy, 'xtl': xtl, 'ytl': ytl, 'xbr': xbr, 'ybr': ybr,
                    'label': box_label, 'matched': False
                })
            
            if element.tag in ['skeleton', 'box', 'points', 'polygon']:
                species = None
                for attr in element.findall('attribute'):
                    if attr.attrib.get('name').lower() == 'species':
                        species = attr.text
                        break
                
                if species:
                    if element.tag == 'skeleton':
                        all_x, all_y = [], []
                        for nested_pt in element.findall('points'):
                            pts = nested_pt.attrib['points'].split(';')
                            for p in pts:
                                all_x.append(float(p.split(',')[0]))
                                all_y.append(float(p.split(',')[1]))
                        if not all_x: continue 
                        cx, cy = sum(all_x) / len(all_x), sum(all_y) / len(all_y)
                    elif element.tag == 'box':
                        cx = (float(element.attrib['xtl']) + float(element.attrib['xbr'])) / 2
                        cy = (float(element.attrib['ytl']) + float(element.attrib['ybr'])) / 2
                    else: 
                        pts = element.attrib['points'].split(';')
                        x_coords, y_coords = [float(p.split(',')[0]) for p in pts], [float(p.split(',')[1]) for p in pts]
                        cx, cy = sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)
                        
                    img_data['metadata'].append({'cx': cx, 'cy': cy, 'species': species})
                    
        original_data[raw_basename] = img_data
    return original_data

def load_progress_elements(xml_path, protected_frames):
    if not os.path.exists(xml_path):
        return {}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    progress_elements = {}
    for image in root.findall('image'):
        img_name = os.path.basename(image.attrib['name'])
        if img_name in protected_frames:
            progress_elements[img_name] = image
    return progress_elements

def generate_cvat_xml(images_dir, labels_dir, output_xml_path, original_xml_path, progress_xml_path, protected_frames):
    print("--- INITIATING TRI-FUSION (WITH LABEL TRANSLATION) ---")
    
    orig_data = load_original_data(original_xml_path)
    progress_elements = load_progress_elements(progress_xml_path, protected_frames)
    
    annotations = ET.Element('annotations')
    ET.SubElement(annotations, 'version').text = '1.1'
    
    meta = ET.SubElement(annotations, 'meta')
    task = ET.SubElement(meta, 'task')
    labels = ET.SubElement(task, 'labels')
    
    # 1. Config for Measurable Fish
    fish_label = ET.SubElement(labels, 'label')
    ET.SubElement(fish_label, 'name').text = 'measurable_fish' 
    attr_m = ET.SubElement(fish_label, 'attributes')
    sp_m = ET.SubElement(attr_m, 'attribute')
    ET.SubElement(sp_m, 'name').text = 'species'
    ET.SubElement(sp_m, 'mutable').text = 'False'
    ET.SubElement(sp_m, 'input_type').text = 'text' 
    ET.SubElement(sp_m, 'default_value').text = 'unknown'
    ET.SubElement(sp_m, 'values').text = 'unknown'

    # 2. Config for Unmeasurable Fish
    unmeas_label = ET.SubElement(labels, 'label')
    ET.SubElement(unmeas_label, 'name').text = 'unmeasurable_fish'
    attr_u = ET.SubElement(unmeas_label, 'attributes')
    sp_u = ET.SubElement(attr_u, 'attribute')
    ET.SubElement(sp_u, 'name').text = 'species'
    ET.SubElement(sp_u, 'mutable').text = 'False'
    ET.SubElement(sp_u, 'input_type').text = 'text' 
    ET.SubElement(sp_u, 'default_value').text = 'unknown'
    ET.SubElement(sp_u, 'values').text = 'unknown'

    for label_name in ['Head', 'Tail']:
        lbl = ET.SubElement(labels, 'label')
        ET.SubElement(lbl, 'name').text = label_name

    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    metrics = {'processed': 0, 'protected': 0, 'boxes_rescued': 0, 'species_rescued': 0, 'unmeasurable_injected': 0}

    for img_id, img_name in enumerate(image_files):
        if img_name in protected_frames:
            if img_name in progress_elements:
                safe_elem = copy.deepcopy(progress_elements[img_name])
                safe_elem.attrib['id'] = str(img_id)
                annotations.append(safe_elem)
                metrics['protected'] += 1
                continue 

        base_img_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(images_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        img_h, img_w = img.shape[:2]

        image_elem = ET.SubElement(annotations, 'image', id=str(img_id), name=img_name, width=str(img_w), height=str(img_h))

        txt_name = base_img_name + '.txt'
        txt_path = os.path.join(labels_dir, txt_name)
        
        lines = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()

        current_max_group_id = 0

        # PASS 1: YOLO Lines (Measurable Fish)
        for obj_id, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 11: continue

            class_id, cx, cy, w, h, hx, hy, hvis, tx, ty, tvis = map(float, parts[:11])

            abs_cx, abs_cy = cx * img_w, cy * img_h
            abs_w, abs_h = w * img_w, h * img_h
            final_xtl, final_ytl = abs_cx - (abs_w / 2), abs_cy - (abs_h / 2)
            final_xbr, final_ybr = abs_cx + (abs_w / 2), abs_cy + (abs_h / 2)
            
            assigned_species = "unknown"
            
            # DEFAULT to the new standard format
            final_label = "measurable_fish"

            if base_img_name in orig_data:
                best_dist_sp = float('inf')
                for orig_meta in orig_data[base_img_name]['metadata']:
                    dist = ((orig_meta['cx'] - abs_cx)**2 + (orig_meta['cy'] - abs_cy)**2)**0.5
                    if dist < best_dist_sp:
                        best_dist_sp = dist
                        assigned_species = orig_meta['species']
                
                if best_dist_sp > 150: assigned_species = "unknown"
                else: metrics['species_rescued'] += 1

                best_dist_box = float('inf')
                best_box = None
                for orig_box in orig_data[base_img_name]['boxes']:
                    dist = ((orig_box['cx'] - abs_cx)**2 + (orig_box['cy'] - abs_cy)**2)**0.5
                    if dist < best_dist_box:
                        best_dist_box = dist
                        best_box = orig_box
                
                if best_box and best_dist_box < 150:
                    final_xtl, final_ytl = best_box['xtl'], best_box['ytl']
                    final_xbr, final_ybr = best_box['xbr'], best_box['ybr']
                    best_box['matched'] = True
                    # Notice we DO NOT inherit best_box['label'] anymore. We force "measurable_fish".
                    metrics['boxes_rescued'] += 1

            current_max_group_id = obj_id + 1
            group_id = str(current_max_group_id)

            box_elem = ET.SubElement(image_elem, 'box', label=final_label, 
                          xtl=f"{final_xtl:.2f}", ytl=f"{final_ytl:.2f}", 
                          xbr=f"{final_xbr:.2f}", ybr=f"{final_ybr:.2f}", 
                          group_id=group_id)
            
            attr_elem = ET.SubElement(box_elem, 'attribute', name='species')
            attr_elem.text = assigned_species

            if hvis > 0:
                ET.SubElement(image_elem, 'points', label='Head', points=f"{hx * img_w:.2f},{hy * img_h:.2f}", group_id=group_id)
            if tvis > 0:
                ET.SubElement(image_elem, 'points', label='Tail', points=f"{tx * img_w:.2f},{ty * img_h:.2f}", group_id=group_id)

        # PASS 2: LEFTOVER SWEEP (Unmeasurable Fish & Abandoned fish-box)
        if base_img_name in orig_data:
            for orig_box in orig_data[base_img_name]['boxes']:
                # If it wasn't matched AND it is either an explicitly unmeasurable fish or a legacy fish-box
                if not orig_box['matched'] and orig_box['label'] in ['unmeasurable_fish', 'fish-box']:
                    current_max_group_id += 1
                    group_id = str(current_max_group_id)
                    
                    assigned_species = "unknown"
                    best_dist_sp = float('inf')
                    for orig_meta in orig_data[base_img_name]['metadata']:
                        dist = ((orig_meta['cx'] - orig_box['cx'])**2 + (orig_meta['cy'] - orig_box['cy'])**2)**0.5
                        if dist < best_dist_sp:
                            best_dist_sp = dist
                            assigned_species = orig_meta['species']
                    if best_dist_sp > 150: assigned_species = "unknown"
                    
                    # Force all leftovers into the unmeasurable_fish category
                    leftover_box = ET.SubElement(image_elem, 'box', label='unmeasurable_fish', 
                                  xtl=f"{orig_box['xtl']:.2f}", ytl=f"{orig_box['ytl']:.2f}", 
                                  xbr=f"{orig_box['xbr']:.2f}", ybr=f"{orig_box['ybr']:.2f}", 
                                  group_id=group_id)
                    
                    attr_elem = ET.SubElement(leftover_box, 'attribute', name='species')
                    attr_elem.text = assigned_species
                    
                    orig_box['matched'] = True
                    metrics['unmeasurable_injected'] += 1

        metrics['processed'] += 1

    with open(output_xml_path, 'w', encoding='utf-8') as f:
        f.write(prettify(annotations))
        
    print(f"\n[+] Success! Processed {metrics['processed']} auto-images.")
    print(f"    - Protected Frames Copied: {metrics['protected']}")
    print(f"    - Manual Boxes Rescued (Measurable): {metrics['boxes_rescued']}")
    print(f"    - Unmeasurable/Legacy Boxes Injected: {metrics['unmeasurable_injected']}")
    print(f"    - Species Tags Rescued: {metrics['species_rescued']}")
    print(f"[+] Final CVAT XML saved to: {output_xml_path}")



if __name__ == '__main__':
    img_directory = r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\UMT Dataset\Head+Tail UMT Annotations\images"
    labels_directory = r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\UMT Dataset\Head+Tail UMT Annotations\labels"
    output_xml = r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\UMT Dataset\Head+Tail UMT Annotations\VS_to_CVAT_Annotations_DS_V1.xml"
    
    # 1. The VERY FIRST XML (with your original manual boxes and species)
    original_cvat_xml = r"XML ANNOTATIONS\UMT Dataset V1 Refixed\annotations_FIXED.xml" 
    
    # 2. The NEW XML you just exported (with your 3 protected frames)
    progress_cvat_xml = r"INSTANCE SEGMENTATION + POSE ESTIMATION\DATASETS\UMT Dataset\3FirstFrames\annotations.xml" 
    
    
    # 3. The filenames of the already annotated frames
    PROTECTED_FRAMES = [
        "P7230001.MOV-33.jpg",
        "P7230001.MOV-38.jpg",
        "P7230002.MOV-14.jpg"
    ]
    
    generate_cvat_xml(images_dir=img_directory, labels_dir=labels_directory, 
                      output_xml_path=output_xml, 
                      original_xml_path=original_cvat_xml,
                      progress_xml_path=progress_cvat_xml,
                      protected_frames=PROTECTED_FRAMES)



