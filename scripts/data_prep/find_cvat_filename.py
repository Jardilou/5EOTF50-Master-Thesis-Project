import xml.etree.ElementTree as ET

def scan_xml_indices(xml_path):
    print(f"🔍 Scanning XML: {xml_path}\n")
    print("-" * 50)
    print(f"{'XML ID'.ljust(10)} | {'File Name'}")
    print("-" * 50)
    
    # Parse the CVAT XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    count = 0
    
    # Find every <image> tag and extract its attributes
    for image_elem in root.findall('image'):
        img_id = image_elem.attrib.get('id')
        img_name = image_elem.attrib.get('name')
        
        # Print the mapping
        print(f"[{img_id}]".ljust(10) + f" | {img_name}")
        count += 1
        
    print("-" * 50)
    print(f"Total images found in XML: {count}")

if __name__ == '__main__':
    # Point this to your exported CVAT XML file
    scan_xml_indices(r"DATASETS\UTM_dataset\UTM Dataset V1 Refixed\annotations_FIXED.xml")