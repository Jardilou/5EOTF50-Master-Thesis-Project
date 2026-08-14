from ultralytics import YOLO
import cv2
import numpy as np

# 1. Load your newly trained model
model = YOLO("Fish_IS_Comparison/yolo_seg_run_1/weights/best.pt")

# 2. Run inference on a test image (or a folder of images)
image_path = "path/to/test/image.jpg"
results = model(image_path)

# 3. Extract and visualize the results
for result in results:
    # Save the visualization (image with overlaid mask and bounding box)
    result.save(filename="inference_result.jpg")
    
    # Extract bounding boxes and masks for your mathematical comparison with SAM 2
    if result.masks is not None:
        # Get the binary mask array (shape: [num_instances, height, width])
        masks_data = result.masks.data.cpu().numpy()
        
        # Get the bounding boxes
        boxes_data = result.boxes.xyxy.cpu().numpy() 
        
        print(f"Found {len(masks_data)} fish in the image.")
        
        # Example of how to access a specific mask
        for i, mask in enumerate(masks_data):
            # The mask is normalized between 0 and 1. Convert to 0-255 for OpenCV saving/viewing
            binary_mask = (mask * 255).astype(np.uint8)
            
            # Resize the mask back to the original image dimensions if necessary
            original_shape = result.orig_shape
            binary_mask_resized = cv2.resize(binary_mask, (original_shape[1], original_shape[0]))
            
            # Save or process the isolated mask
            cv2.imwrite(f"extracted_mask_{i}.png", binary_mask_resized)
            
    else:
        print("No fish detected in this image.")