
import os
import xml.etree.ElementTree as ET

def point_in_box(p, box):
    return (box['xtl'] <= p['x'] <= box['xbr']) and (box['ytl'] <= p['y'] <= box['ybr'])

def opposite_halves(p1, p2, box):
    if not p1 or not p2: return True
    cx = (box['xtl'] + box['xbr']) / 2.0
    cy = (box['ytl'] + box['ybr']) / 2.0
    v1_x, v1_y = p1['x'] - cx, p1['y'] - cy
    v2_x, v2_y = p2['x'] - cx, p2['y'] - cy
    return ((v1_x * v2_x) + (v1_y * v2_y)) < 0

def get_fish_type(label_name):
    """Returns a tuple: (Class_ID, Is_Measurable_Boolean)"""
    label_lower = label_name.lower()
    if "unmeas" in label_lower or "unmes" in label_lower:
        return (0, False) # Class 0, but DO NOT use keypoints
    elif "meas" in label_lower:
        return (0, True)  # Class 0, AND use keypoints
    else:
        return (-1, False) # Ignore
    


def extract_species_attribute(element):
    """Digs into the CVAT XML to find the custom 'species' attribute."""
    for attr in element.findall('attribute'):
        if attr.attrib.get('name', '').lower() == 'species':
            return attr.text
    return "unknown"

def convert_cvat_to_yolo_pose(xml_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    KPT_ORDER = ['Head', 'Tail', 'Tail_2']
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    total_measurable = 0
    total_unmeasurable = 0
    
    for image in root.findall('image'):
        # Safely strip any CVAT path prefixes
        raw_name = os.path.basename(image.attrib['name'])
        img_w = float(image.attrib['width'])
        img_h = float(image.attrib['height'])
        
        txt_filename = os.path.splitext(raw_name)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)
        
        boxes = []
        points = []
        
        # 1. EXTRACT ALL SHAPES (Both <box> and <polygon>)
        # Some annotators use polygons for unmeasurable fish. This catches them both.
        for shape in image:
            if shape.tag not in ['box', 'polygon']:
                continue
                
            label = shape.attrib.get('label', '')
            class_id, is_measurable = get_fish_type(label)
            if class_id == -1: continue
            
            # Get the exact bounds depending on the shape type
            if shape.tag == 'box':
                xtl, ytl = float(shape.attrib['xtl']), float(shape.attrib['ytl'])
                xbr, ybr = float(shape.attrib['xbr']), float(shape.attrib['ybr'])
            elif shape.tag == 'polygon':
                # Convert polygon boundary into a standard bounding box
                pts_str = shape.attrib.get('points', '').replace(';', ',')
                coords = list(map(float, pts_str.split(',')))
                xs = coords[0::2]
                ys = coords[1::2]
                xtl, ytl, xbr, ybr = min(xs), min(ys), max(xs), max(ys)
            
            area = (xbr - xtl) * (ybr - ytl)
            species_name = extract_species_attribute(shape)
            
            boxes.append({
                'class_id': class_id,
                'is_measurable': is_measurable, # NEW: Track if it gets keypoints
                'species': species_name,
                'xtl': xtl, 'ytl': ytl, 'xbr': xbr, 'ybr': ybr,
                'area': area,
                'kpts': {'Head': None, 'Tail': None, 'Tail_2': None}
            })
            
        # 2. Extract Keypoints safely (handles CVAT semicolon bugs)
        for pt in image.findall('points'):
            label = pt.attrib.get('label', '')
            if label not in KPT_ORDER: continue
            
            cleaned_string = pt.attrib.get('points', '').replace(';', ',')
            raw_coords = cleaned_string.split(',')
            x, y = float(raw_coords[0]), float(raw_coords[1])
            
            points.append({'label': label, 'x': x, 'y': y, 'used': False})
            
        # 3. Sort boxes by area (Smallest first to solve overlap theft)
        boxes.sort(key=lambda b: b['area'])
        
        # 4. Assignment Loop (Keypoint mapping)
        for box in boxes:
            # We don't bother looking for keypoints if it's an unmeasurable fish (Class 1)
            if box['is_measurable'] == 0:
                for kpt_label in KPT_ORDER:
                    candidates = [p for p in points if p['label'] == kpt_label and not p['used'] and point_in_box(p, box)]
                    
                    if len(candidates) == 1:
                        box['kpts'][kpt_label] = candidates[0]
                        candidates[0]['used'] = True
                    elif len(candidates) > 1:
                        # Conflict! Apply the Opposite Halves Heuristic
                        anchor = box['kpts']['Tail'] if kpt_label == 'Head' else box['kpts']['Head']
                        best_cand = candidates[0]
                        if anchor:
                            for cand in candidates:
                                if opposite_halves(cand, anchor, box):
                                    best_cand = cand
                                    break
                        box['kpts'][kpt_label] = best_cand
                        best_cand['used'] = True

        # 5. Write to YOLO format
        with open(txt_path, 'w') as f:
            for box in boxes:
                c_id = box['class_id']
                if c_id == 0: total_measurable += 1
                else: total_unmeasurable += 1
                
                # Normalize Bounding Box
                cx = ((box['xtl'] + box['xbr']) / 2.0) / img_w
                cy = ((box['ytl'] + box['ybr']) / 2.0) / img_h
                w = (box['xbr'] - box['xtl']) / img_w
                h = (box['ybr'] - box['ytl']) / img_h
                
                line = f"{c_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                
                # Append Keypoints (Only populate them for measurable fish)
                for kpt_label in KPT_ORDER:
                    # Only write real keypoints if it is a measurable fish
                    kpt = box['kpts'][kpt_label] if box['is_measurable'] else None
                    if kpt:
                        kx, ky = kpt['x'] / img_w, kpt['y'] / img_h
                        line += f" {kx:.6f} {ky:.6f} 2"
                    else:
                        line += " 0.000000 0.000000 0"
                
                # APPEND THE SAFELY COMMENTED SPECIES ATTRIBUTE
                line += f" # species: {box['species']}"
                
                f.write(line + "\n")
                
    print("\n" + "="*50)
    print(" CVAT TO YOLO CONVERSION COMPLETE")
    print("="*50)
    print(f" -> Measurable Fish Converted:   {total_measurable}")
    print(f" -> Unmeasurable Fish Converted: {total_unmeasurable}")
    print("="*50 + "\n")
