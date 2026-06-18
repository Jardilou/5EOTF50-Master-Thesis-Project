

import os
import shutil

# --- Configuration ---
# 1. The folder containing the images you want to retrieve (the "list" of x.jpg)
reference_folder = "dataset1/dataset1/results_from_research_paper/Correct-Images"

# 2. The folders where the master images and labels are currently stored
source_images = "dataset1/dataset1/rgb"
source_labels = "dataset1/dataset1/labels"

# 3. Where you want the retrieved files to be copied to
dest_correct_images = "dataset1/dataset1/correct_images"
dest_correct_labels = "dataset1/dataset1/correct_labels"

# 4. Destination folders for INCORRECT data
dest_incorrect_images = "dataset1/dataset1/incorrect_images"
dest_incorrect_labels = "dataset1/dataset1/incorrect_labels"
# ---------------------

# Create all 4 destination folders safely
for folder in [dest_correct_images, dest_correct_labels, dest_incorrect_images, dest_incorrect_labels]:
    os.makedirs(folder, exist_ok=True)

# Step 1: Create a fast lookup list (set) of CORRECT base names
correct_bases = set()
for filename in os.listdir(reference_folder):
    if filename.endswith(".jpg"):
        correct_bases.add(os.path.splitext(filename)[0])

correct_count = 0
incorrect_count = 0
missing_labels = 0

# Step 2: Go through ALL images in the master source folder
for filename in os.listdir(source_images):
    if filename.endswith(".jpg"):
        base_name = os.path.splitext(filename)[0]
        
        src_img_path = os.path.join(source_images, filename)
        src_lbl_path = os.path.join(source_labels, f"{base_name}.txt")
        
        # Check if the label exists in the source folder
        label_exists = os.path.exists(src_lbl_path)
        if not label_exists:
            missing_labels += 1

        # Step 3: Route to either correct or incorrect folders
        if base_name in correct_bases:
            # It's in the reference list -> Copy to CORRECT folders
            shutil.copy2(src_img_path, os.path.join(dest_correct_images, filename))
            if label_exists:
                shutil.copy2(src_lbl_path, os.path.join(dest_correct_labels, f"{base_name}.txt"))
            correct_count += 1
            
        else:
            # It's NOT in the reference list -> Copy to INCORRECT folders
            shutil.copy2(src_img_path, os.path.join(dest_incorrect_images, filename))
            if label_exists:
                shutil.copy2(src_lbl_path, os.path.join(dest_incorrect_labels, f"{base_name}.txt"))
            incorrect_count += 1

print("-" * 30)
print("Sorting complete!")
print(f"Correct images sorted: {correct_count}")
print(f"Incorrect images sorted: {incorrect_count}")
if missing_labels > 0:
    print(f"Note: {missing_labels} images were missing their .txt labels in the master folder.")