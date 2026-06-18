import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_single_channel(image_path, channel_to_show='R'):
    # 1. Read the image
    # OpenCV loads images in BGR format by default
    img_bgr = cv2.imread(image_path)
    
    if img_bgr is None:
        print("Error: Could not load image. Check the path.")
        return

    # Convert BGR to standard RGB format
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # ---------------------------------------------------------
    # Method A: Color Isolation (Keeps the image looking R, G, or B)
    # ---------------------------------------------------------
    isolated_img = img_rgb.copy()
    
    # In RGB: 0 is Red, 1 is Green, 2 is Blue
    if channel_to_show.upper() == 'R':
        isolated_img[:, :, 1] = 0  # Zero out Green
        isolated_img[:, :, 2] = 0  # Zero out Blue
        channel_idx = 0
    elif channel_to_show.upper() == 'G':
        isolated_img[:, :, 0] = 0  # Zero out Red
        isolated_img[:, :, 2] = 0  # Zero out Blue
        channel_idx = 1
    elif channel_to_show.upper() == 'B':
        isolated_img[:, :, 0] = 0  # Zero out Red
        isolated_img[:, :, 1] = 0  # Zero out Green
        channel_idx = 2
    else:
        print("Invalid channel. Choose 'R', 'G', or 'B'.")
        return

    # ---------------------------------------------------------
    # Method B: Grayscale Intensity (Shows pure channel data)
    # ---------------------------------------------------------
    intensity_img = img_rgb[:, :, channel_idx]

    # Plotting both methods side-by-side
    plt.figure(figsize=(10, 5))

    # Plot Method A
    plt.subplot(1, 2, 1)
    plt.imshow(isolated_img)
    plt.title(f'Isolated {channel_to_show.upper()} Channel (Color)')
    plt.axis('off')

    # Plot Method B
    plt.subplot(1, 2, 2)
    plt.imshow(intensity_img, cmap='gray')
    plt.title(f'{channel_to_show.upper()} Channel Intensity (Grayscale)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


      # Change 'G' to 'R' or 'B' as needed
display_single_channel(r'STEREO VIDEO LENGTH RETRIEVAL\UMT Calibration Images\Quadrat 5m.MOV-7_Left.jpg', 'R')