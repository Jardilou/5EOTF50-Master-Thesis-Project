import xml.etree.ElementTree as ET

def auto_group_cvat_annotations(input_xml, output_xml):
    print(f"🚀 Scanning {input_xml} to mathematically group boxes and skeletons...\n")
    
    try:
        tree = ET.parse(input_xml)
        root = tree.getroot()
    except Exception as e:
        print(f"❌ Error loading XML: {e}")
        return

    group_counter = 10000 # Start at a high number to avoid any CVAT defaults
    matched_count = 0
    unmatched_skeletons = 0

    # Iterate through every image in the XML
    for image in root.findall('image'):
        img_name = image.attrib['name']
        
        boxes = image.findall('box')
        skeletons = image.findall('skeleton')
        
        # We will iterate through skeletons and try to find their parent box
        for skel in skeletons:
            pts = skel.findall('points')
            if not pts: 
                continue
            
            # 1. Calculate the Centroid (middle X, Y) of the skeleton
            x_coords = []
            y_coords = []
            for pt in pts:
                coords = pt.attrib['points'].split(',')
                x_coords.append(float(coords[0]))
                y_coords.append(float(coords[1]))
            
            centroid_x = sum(x_coords) / len(x_coords)
            centroid_y = sum(y_coords) / len(y_coords)

            best_box = None
            min_area = float('inf')

            # 2. Find the smallest bounding box that contains this centroid
            for box in boxes:
                xtl = float(box.attrib['xtl'])
                ytl = float(box.attrib['ytl'])
                xbr = float(box.attrib['xbr'])
                ybr = float(box.attrib['ybr'])

                # Check if the skeleton's center point is inside this box
                if xtl <= centroid_x <= xbr and ytl <= centroid_y <= ybr:
                    area = (xbr - xtl) * (ybr - ytl)
                    # If multiple boxes overlap, pick the tightest fitting one
                    if area < min_area:
                        min_area = area
                        best_box = box

            # 3. If a match is found, permanently group them!
            if best_box is not None:
                shared_id = str(group_counter)
                
                # Overwrite or create the group_id attribute
                skel.set('group_id', shared_id)
                best_box.set('group_id', shared_id)
                
                group_counter += 1
                matched_count += 1
                
                # Remove the box from the available pool so two skeletons don't claim it
                boxes.remove(best_box) 
            else:
                unmatched_skeletons += 1
                print(f"⚠️ Warning: Could not find a bounding box for a skeleton in {img_name}")

    # Save the repaired XML
    tree.write(output_xml, encoding='utf-8', xml_declaration=True)
    
    print("-" * 50)
    print("Auto-Grouping Complete!")
    print(f"Successfully paired and grouped {matched_count} fish.")
    if unmatched_skeletons > 0:
        print(f"Failed to find boxes for {unmatched_skeletons} skeletons.")
    print(f"📄 Saved clean data to: {output_xml}")

if __name__ == '__main__':
    # Input: Your broken XML. Output: The repaired XML.
    auto_group_cvat_annotations(
        input_xml=r"DATASETS\UTM_dataset\UTM Dataset V1 Refixed\annotations.xml", 
        output_xml=r"DATASETS\UTM_dataset\UTM Dataset V1 Refixed\annotations_FIXED.xml"
    )