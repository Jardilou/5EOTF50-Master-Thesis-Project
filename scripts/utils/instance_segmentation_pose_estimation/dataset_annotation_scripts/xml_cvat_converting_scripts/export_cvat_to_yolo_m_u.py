
import os
import csv
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
        return (0, False) # Class 0 (Unified), but DO NOT use keypoints
    elif "meas" in label_lower:
        return (0, True)  # Class 0 (Unified), AND use keypoints
    else:
        return (-1, False)

def extract_species_attribute(element):
    for attr in element.findall('attribute'):
        if attr.attrib.get('name', '').lower() == 'species':
            # Clean out any nasty 0xa0 non-breaking spaces that break YOLO
            clean_text = attr.text.replace('\xa0', ' ').strip() if attr.text else "unknown"
            return clean_text
    return "unknown"

def convert_cvat_to_yolo_pose(xml_path, output_dir, project_root):
    os.makedirs(output_dir, exist_ok=True)
    KPT_ORDER = ['Head', 'Tail', 'Tail_2']
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    total_measurable = 0
    total_unmeasurable = 0
    
    # Open our Metadata Ledger
    csv_path = os.path.join(project_root, "species_metadata.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image_Name', 'Class_ID', 'Is_Measurable', 'Species', 'BBox_CX', 'BBox_CY'])
        
        for image in root.findall('image'):
            raw_name = os.path.basename(image.attrib['name'])
            img_w = float(image.attrib['width'])
            img_h = float(image.attrib['height'])
            
            txt_filename = os.path.splitext(raw_name)[0] + ".txt"
            txt_path = os.path.join(output_dir, txt_filename)
            
            boxes = []
            points = []
            
            for shape in image:
                if shape.tag not in ['box', 'polygon']: continue
                    
                label = shape.attrib.get('label', '')
                class_id, is_measurable = get_fish_type(label)
                if class_id == -1: continue
                
                if shape.tag == 'box':
                    xtl, ytl = float(shape.attrib['xtl']), float(shape.attrib['ytl'])
                    xbr, ybr = float(shape.attrib['xbr']), float(shape.attrib['ybr'])
                elif shape.tag == 'polygon':
                    pts_str = shape.attrib.get('points', '').replace(';', ',')
                    coords = list(map(float, pts_str.split(',')))
                    xs, ys = coords[0::2], coords[1::2]
                    xtl, ytl, xbr, ybr = min(xs), min(ys), max(xs), max(ys)
                
                area = (xbr - xtl) * (ybr - ytl)
                species_name = extract_species_attribute(shape)
                
                boxes.append({
                    'class_id': class_id,
                    'is_measurable': is_measurable,
                    'species': species_name,
                    'xtl': xtl, 'ytl': ytl, 'xbr': xbr, 'ybr': ybr,
                    'area': area,
                    'kpts': {'Head': None, 'Tail': None, 'Tail_2': None}
                })
                
            for pt in image.findall('points'):
                label = pt.attrib.get('label', '')
                if label not in KPT_ORDER: continue
                cleaned_string = pt.attrib.get('points', '').replace(';', ',')
                raw_coords = cleaned_string.split(',')
                points.append({'label': label, 'x': float(raw_coords[0]), 'y': float(raw_coords[1]), 'used': False})
                
            boxes.sort(key=lambda b: b['area'])
            
            for box in boxes:
                if box['is_measurable']:
                    for kpt_label in KPT_ORDER:
                        candidates = [p for p in points if p['label'] == kpt_label and not p['used'] and point_in_box(p, box)]
                        if len(candidates) == 1:
                            box['kpts'][kpt_label] = candidates[0]
                            candidates[0]['used'] = True
                        elif len(candidates) > 1:
                            anchor = box['kpts']['Tail'] if kpt_label == 'Head' else box['kpts']['Head']
                            best_cand = candidates[0]
                            if anchor:
                                for cand in candidates:
                                    if opposite_halves(cand, anchor, box):
                                        best_cand = cand
                                        break
                            box['kpts'][kpt_label] = best_cand
                            best_cand['used'] = True

            with open(txt_path, 'w', encoding='utf-8') as f:
                for box in boxes:
                    if box['is_measurable']: total_measurable += 1
                    else: total_unmeasurable += 1
                    
                    cx = ((box['xtl'] + box['xbr']) / 2.0) / img_w
                    cy = ((box['ytl'] + box['ybr']) / 2.0) / img_h
                    w = (box['xbr'] - box['xtl']) / img_w
                    h = (box['ybr'] - box['ytl']) / img_h
                    
                    # 1. WRITE PURE MATHEMATICS FOR YOLO
                    line = f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                    
                    for kpt_label in KPT_ORDER:
                        kpt = box['kpts'][kpt_label] if box['is_measurable'] else None
                        if kpt:
                            line += f" {kpt['x'] / img_w:.6f} {kpt['y'] / img_h:.6f} 2"
                        else:
                            line += " 0.000000 0.000000 0"
                    
                    f.write(line + "\n")
                    
                    # 2. WRITE METADATA TO CSV FOR YOU
                    csv_writer.writerow([raw_name, 0, box['is_measurable'], box['species'], f"{cx:.6f}", f"{cy:.6f}"])
                    
    print("\n" + "="*50)
    print(" CVAT TO YOLO CONVERSION COMPLETE (Numpy Compliant)")
    print("="*50)
    print(f" -> Measurable Fish Converted:   {total_measurable}")
    print(f" -> Unmeasurable Fish Converted: {total_unmeasurable}")
    print(f" -> Species Ledger saved to:     {csv_path}")
    print("="*50 + "\n")


if __name__ == '__main__':
    # Add the path to your exported CVAT XML and where you want the YOLO txt files
    CVAT_XML_PATH = r"XML ANNOTATIONS\UMT Head + Tails UMT Annotation V2_2.xml"
    OUTPUT_LABELS_DIR = r"DATASETS\DATASETS FOR INSTANCE SEGMENTATION + POSE ESTIMATION\UMT Dataset\CURRENT\Head + Tails UMT Annotation V2\LABELS"
    PROJECT_ROOT = r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE" 
    convert_cvat_to_yolo_pose(CVAT_XML_PATH, OUTPUT_LABELS_DIR, PROJECT_ROOT)