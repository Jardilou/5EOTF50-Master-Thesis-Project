
import xml.etree.ElementTree as ET
import argparse

def filter_cvat_annotations(input_xml, output_xml, keyword="solo_channel"):
    print(f"Loading '{input_xml}'...")
    
    # Parse the XML file
    try:
        tree = ET.parse(input_xml)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"Error: The file {input_xml} was not found.")
        return
    except ET.ParseError:
        print(f"Error: The file {input_xml} is not a valid XML file.")
        return

    # Keep track of counts for reporting
    initial_count = 0
    removed_count = 0

    # Iterate through all <image> tags in the CVAT XML
    # We use findall to create a list so we can safely modify the root tree
    for image in root.findall('image'):
        initial_count += 1
        filename = image.get('name', '')
        
        # If the keyword is not in the filename, remove the entire <image> node
        if keyword not in filename:
            root.remove(image)
            removed_count += 1

    # Save the modified XML tree to the output file
    tree.write(output_xml, encoding='utf-8', xml_declaration=True)
    
    # Print summary
    kept_count = initial_count - removed_count
    print("--- Filtering Complete ---")
    print(f"Total images originally: {initial_count}")
    print(f"Images removed:          {removed_count}")
    print(f"Images kept:             {kept_count}")
    print(f"Saved filtered data to:  '{output_xml}'")

if __name__ == "__main__":
    # You can change these variables manually or use the command line arguments
    INPUT_FILE = r"Calibration images\Intermediary-dataset-calibration-images\annotations.xml"          # Replace with your input file name
    OUTPUT_FILE = r"Calibration images\Intermediary-dataset-calibration-images\filtered_annotations.xml" # Replace with your desired output file name
    TARGET_KEYWORD = "solo_channel"
    
    filter_cvat_annotations(INPUT_FILE, OUTPUT_FILE, TARGET_KEYWORD)