import os
import cv2

def crop_images_from_yolo_masks(img_dir, label_dir, output_dir):
    # Determine the parent folder name (XX)
    # If img_dir is "path/to/images/train", folder_name will be "train"
    folder_name = os.path.basename(os.path.normpath(img_dir))
    
    
    output_dir = os.path.join(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving cropped images to: {output_dir}")

    # Common image extensions to look for
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')
    
    # Grab all image files
    img_files = [f for f in os.listdir(img_dir) if f.endswith(valid_extensions)]
    
    if not img_files:
        print(f"No images found in {img_dir}")
        return

    for img_filename in img_files:
        base_name = os.path.splitext(img_filename)[0]
        label_filename = base_name + '.txt'
        
        img_path = os.path.join(img_dir, img_filename)
        label_path = os.path.join(label_dir, label_filename)
        
        # Skip if there is no corresponding label file
        if not os.path.exists(label_path):
            continue
            
        # Load the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
            
        img_h, img_w = img.shape[:2]
        
        # Read the YOLO label file
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        # Loop through every object mask in the text file
        for index, line in enumerate(lines):
            parts = line.strip().split()
            
            # Ensure it's a valid segmentation format (class_id + at least 3 x,y pairs)
            if len(parts) < 7:
                continue
                
            # Extract coordinates
            coords = [float(val) for val in parts[1:]]
            
            # Separate x and y coordinates
            x_coords = coords[0::2]
            y_coords = coords[1::2]
            
            # Find the min and max coordinates (the bounding box of the polygon)
            x_min = int(min(x_coords) * img_w)
            x_max = int(max(x_coords) * img_w)
            y_min = int(min(y_coords) * img_h)
            y_max = int(max(y_coords) * img_h)
            
            # Constrain coordinates to image dimensions (safety check)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_w, x_max)
            y_max = min(img_h, y_max)
            
            # Crop the image using numpy slicing [y:y+h, x:x+w]
            cropped_img = img[y_min:y_max, x_min:x_max]
            
            # Ensure the crop is valid (not empty)
            if cropped_img.size == 0:
                print(f"Warning: Empty crop for {img_filename}, object {index}")
                continue
                
            # Save the cropped image
            # Naming format: originalname_crop_0.jpg
            ext = os.path.splitext(img_filename)[1]
            output_filename = f"{base_name}_crop_{index}{ext}"
            output_filepath = os.path.join(output_dir, output_filename)
            
            cv2.imwrite(output_filepath, cropped_img)
            
    print("Cropping process completed.")

if __name__ == '__main__':
    # Replace these with your actual directories
    img_directory = r"data\processed\fish4knowledge\segmentation\images\val"
    labels_directory = r"data\processed\fish4knowledge\segmentation\labels\val"
    output_directory = r"data\processed\fish4knowledge\segmentation\cropped-images\val"
    
    crop_images_from_yolo_masks(img_dir=img_directory, label_dir=labels_directory, output_dir=output_directory)