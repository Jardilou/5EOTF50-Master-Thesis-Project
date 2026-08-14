import os
import cv2
import numpy as np

def calculate_absolute_pose_metrics(dataset_pairs):
    """
    Reads YOLO Pose format annotations and corresponding images to calculate
    absolute pixel dimensions, area, and aspect ratio.

    dataset_pairs: A list of tuples containing (img_dir, label_dir)
    """
    abs_widths, abs_heights, pixel_areas, aspect_ratios = [], [], [], []

    print(f"Scanning {len(dataset_pairs)} directory pairs...")

    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')

    for img_dir, label_dir in dataset_pairs:
        if not os.path.exists(label_dir) or not os.path.exists(img_dir):
            print(f"Warning: Missing directory in pair ({img_dir}, {label_dir}). Skipping.")
            continue

        label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

        for label_filename in label_files:
            base_name = os.path.splitext(label_filename)[0]
            label_path = os.path.join(label_dir, label_filename)

            # Find the corresponding image to get its pixel dimensions
            img_path = None
            for ext in valid_extensions:
                temp_path = os.path.join(img_dir, base_name + ext)
                if os.path.exists(temp_path):
                    img_path = temp_path
                    break

            if not img_path:
                continue

            # Load image just to read its shape (height, width)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_h, img_w = img.shape[:2]

            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()

                # Check for exactly 17 elements (Pose with 4 keypoints)
                if len(parts) == 17:
                    try:
                        norm_w = float(parts[3])
                        norm_h = float(parts[4])

                        # Convert normalized dimensions to absolute pixels
                        abs_w = norm_w * img_w
                        abs_h = norm_h * img_h

                        # Calculate total pixel area
                        pixel_area = abs_w * abs_h

                        abs_widths.append(abs_w)
                        abs_heights.append(abs_h)
                        pixel_areas.append(pixel_area)

                        # Calculate Aspect Ratio (Width / Height)
                        if abs_h > 0:
                            aspect_ratio = abs_w / abs_h
                            aspect_ratios.append(aspect_ratio)

                    except ValueError:
                        continue

    if not abs_widths:
        print("No valid 4-keypoint YOLO Pose annotations found with matching images.")
        return

    # Calculate means
    mean_abs_width = np.mean(abs_widths)
    mean_abs_height = np.mean(abs_heights)
    mean_pixel_area = np.mean(pixel_areas)
    mean_aspect_ratio = np.mean(aspect_ratios)
    total_boxes = len(abs_widths)

    print("--- ABSOLUTE PIXEL METRICS ---")
    print(f"Total targets analyzed: {total_boxes}")
    print(f"Mean Width (pixels):  {mean_abs_width:.2f} px")
    print(f"Mean Height (pixels): {mean_abs_height:.2f} px")
    print(f"Mean Area (pixels):   {mean_pixel_area:.2f} px squared")
    print(f"Mean Scale (pixels)(square root of area):   {np.sqrt(mean_pixel_area):.2f} px")
    print(f"Mean Aspect Ratio:    {mean_aspect_ratio:.2f} (W/H)")
    print("------------------------------")



if __name__ == '__main__':
    # Define pairs of (Image_Directory, Label_Directory)
    # dataset_directories = [
    #     (r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\hainan-bezierfusion\Dataset_1\detection+pose_estimation\train_test\images\val", r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\hainan-bezierfusion\Dataset_1\detection+pose_estimation\train_test\labels\val"),
    #     (r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\hainan-bezierfusion\Dataset_2\detection+pose_estimation\train_test\images\val",   r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\hainan-bezierfusion\Dataset_2\detection+pose_estimation\train_test\labels\val"),
    #     (r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\UMT\detection+4_keypoints_pose_estimation\images\val",  r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\UMT\detection+4_keypoints_pose_estimation\labels\val"),
    #     (r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\deepfish\detection+4_keypoints_pose_estimation\images\val",  r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\deepfish\detection+4_keypoints_pose_estimation\labels\val")
    # ]


    dataset_directories = [
            # (r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\hainan-bezierfusion\Dataset_1\detection+pose_estimation\train_test\images\train", r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\hainan-bezierfusion\Dataset_1\detection+pose_estimation\train_test\labels\train"),
            # (r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\hainan-bezierfusion\Dataset_2\detection+pose_estimation\train_test\images\train",   r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\hainan-bezierfusion\Dataset_2\detection+pose_estimation\train_test\labels\train"),
            # (r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\hainan-bezierfusion\Dataset_1\detection+pose_estimation\train_test\images\val", r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\hainan-bezierfusion\Dataset_1\detection+pose_estimation\train_test\labels\val"),
            # (r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\hainan-bezierfusion\Dataset_2\detection+pose_estimation\train_test\images\val",   r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\hainan-bezierfusion\Dataset_2\detection+pose_estimation\train_test\labels\val"),



            (r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\UMT\detection+4_keypoints_pose_estimation\images\val",  r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\UMT\detection+4_keypoints_pose_estimation\labels\val"),
            (r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\UMT\detection+4_keypoints_pose_estimation\images\train",  r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\data\processed\UMT\detection+4_keypoints_pose_estimation\labels\train"),
        ]


    calculate_absolute_pose_metrics(dataset_pairs=dataset_directories)
