import cv2
import numpy as np
import os

def sharpen_image(image, amount=1.5, threshold=0):
    """
    Enhances edges using an Unsharp Mask.
    amount: Strength of the sharpening (usually 1.0 to 2.0).
    threshold: Minimum brightness change to be sharpened (prevents noise).
    """
    # Use a Gaussian Blur to create the "unsharp" version
    blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
    
    # Calculate the sharpened image: original + (original - blurred) * amount
    sharpened = float(amount + 1) * image - float(amount) * blurred
    
    # Clip values to stay within [0, 255] and convert back to uint8
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    if threshold > 0:
        low_contrast_mask = np.abs(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
        
    return sharpened

def enhance_contrast(image):
    # CLAHE for contrast
    if len(image.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        return sharpen_image(enhanced) # Apply sharpening after contrast
    
    # Color: LAB space for contrast
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Apply sharpening specifically to the L (lightness) channel
    cl_sharp = sharpen_image(cl)
    
    enhanced_lab = cv2.merge((cl_sharp, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

def process_folder(input_folder, output_folder, channel_to_show='R'):
    # 1. Create Output Directory
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    # 2. Get list of all files in the folder
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]

    if not files:
        print(f"No images found in {input_folder}")
        return

    print(f"Found {len(files)} images. Starting processing...")

    for filename in files:
        file_path = os.path.join(input_folder, filename)
        
        # Load image
        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        channel_idx = {'R': 0, 'G': 1, 'B': 2}.get(channel_to_show.upper(), 0)
        
        # Color Isolation
        isolated_img = np.zeros_like(img_rgb)
        isolated_img[:, :, channel_idx] = img_rgb[:, :, channel_idx]
        
        # Intensity (Grayscale)
        intensity_img = img_rgb[:, :, channel_idx]

        # Enhance
        isolated_enhanced = enhance_contrast(isolated_img)
        intensity_enhanced = enhance_contrast(intensity_img)

        # Naming (original_name_solo_channel_X.jpg)
        base_name = os.path.splitext(filename)[0]
        suffix = f"_solo_channel_{channel_to_show.upper()}"
        
        # Save both versions
        # cv2.imwrite(os.path.join(output_folder, f"{base_name}{suffix}_color.jpg"), 
        #             cv2.cvtColor(isolated_enhanced, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_folder, f"{base_name}{suffix}_gray.jpg"), 
                    intensity_enhanced)

    print("Done! All images processed.")


# --- EXECUTION ---
input_path = r'Calibration images\Stereo Video Calibration STEREO-DOVs\All_Calibration_Images'
output_dir = r'Calibration images\Stereo Video Calibration STEREO-DOVs\Processed_Calibration_Images_Channel_R'

process_folder(input_path, output_dir, 'R')