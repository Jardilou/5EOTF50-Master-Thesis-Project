# from ultralytics import YOLO
# import os

# def generate_pseudo_labels(model_path, images_dir, output_dir):
#     if not os.path.exists(model_path):
#         print("❌ Model not found! Check your path.")
#         return

#     print("🤖 Loading model...")
#     model = YOLO(model_path)
    
#     print(f"🚀 Starting AI annotation on {images_dir}...")
    
#     # Run prediction and tell YOLO to generate text files
#     results = model.predict(
#         source=images_dir,
#         conf=0.4,            # Only save labels if the AI is at least 40% sure
#         save_txt=True,       # THIS is the magic command that creates the .txt files
#         save_conf=False,     # Keep to False: CVAT doesn't want confidence scores in the file
#         project=output_dir,
#         name="deepfish_pseudo_labels",
#         exist_ok=True
#     )
    
#     labels_path = os.path.join(output_dir, "deepfish_pseudo_labels", "labels")
#     print(f"\n✅ Done! The AI has generated your text files here:")
#     print(f"📂 {labels_path}")

# # --- EXECUTION ---
# # USE YOUR ABSOLUTE PATHS HERE
# generate_pseudo_labels(
#     model_path=r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\ultralytics\runs\pose\BezierFusion\eamrf_training_run14\weights\last.pt",
#     images_dir=r"C:\path\to\DeepFish\images\train", 
#     output_dir=r"C:\path\to\where\you\want\the\results"
# )

import os
import shutil
from ultralytics import YOLO

def filter_high_confidence_images(model_path, source_folders, curated_img_dir, conf_threshold=0.85):
    """
    Scans multiple folders, runs rapid inference, and copies ONLY images 
    where a fish is detected with a confidence higher than the threshold.
    """
    os.makedirs(curated_img_dir, exist_ok=True)
    model = YOLO(model_path)
    
    total_copied = 0
    
    print(f"Scanning folders for fish with > {conf_threshold*100}% confidence...")
    
    for folder_idx, folder in enumerate(source_folders):
        if not os.path.exists(folder):
            print(f"⚠️ Skipping {folder} (Not found)")
            continue
            
        print(f"  -> Scanning folder: {folder}")
        
        # stream=True is crucial for massive datasets. It processes images one by one 
        # instead of loading them all into RAM.
        results = model.predict(source=folder, conf=conf_threshold, stream=True, verbose=False)
        
        for result in results:
            # If the model found at least 1 bounding box above our threshold...
            if len(result.boxes) > 0:
                original_img_path = result.path
                filename = os.path.basename(original_img_path)
                
                # To prevent images from different folders from overwriting each other if 
                # they have the same name (e.g., "img_01.jpg"), we add a folder prefix.
                safe_filename = f"folder{folder_idx}_{filename}"
                destination_path = os.path.join(curated_img_dir, safe_filename)
                
                # Copy the image to the new curated folder
                shutil.copy(original_img_path, destination_path)
                total_copied += 1
                
    print(f"Filtering Complete! Found {total_copied} high-confidence images.")
    return curated_img_dir

def generate_pseudo_labels(model_path, images_dir, output_dir):
    """
    Takes the curated folder of high-confidence images and generates YOLO .txt labels for CVAT.
    """
    print(f"\nStarting Pseudo-labeling on the curated images...")
    model = YOLO(model_path)
    
    # We lower the conf here slightly just to ensure we draw the points on the fish 
    # we already know are there, but the folder is already filtered!
    model.predict(
        source=images_dir,
        conf=0.4,            
        save_txt=True,       
        save_conf=False,     
        project=output_dir,
        name="curated_pseudo_labels",
        exist_ok=True
    )
    
    labels_path = os.path.join(output_dir, "curated_pseudo_labels", "labels")
    print(f"Pseudo-labeling Done! Your CVAT-ready text files are here:")
    print(f"{labels_path}")

# ==========================================
# --- EXECUTION PIPELINE ---
# ==========================================
def main():
    MODEL_WEIGHTS = r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\ultralytics\runs\pose\BezierFusion\eamrf_training_run14\weights\last.pt"
    
    # 1. List all the raw, unannotated folders you want to dig through
    RAW_DATA_FOLDERS = [
        r"C:/Users/Work Mode Big Dog/OneDrive - ECAM/Bureau/ERASMUS/PROJECT/CODE/DATASETS/DeepFish/Classification/7117/valid",
        r"C:/Users/Work Mode Big Dog/OneDrive - ECAM/Bureau/ERASMUS/PROJECT/CODE/DATASETS/DeepFish/Classification/7268/valid",
        r"C:/Users/Work Mode Big Dog/OneDrive - ECAM/Bureau/ERASMUS/PROJECT/CODE/DATASETS/DeepFish/Classification/7393/valid",
        r"C:/Users/Work Mode Big Dog/OneDrive - ECAM/Bureau/ERASMUS/PROJECT/CODE/DATASETS/DeepFish/Classification/7398/valid",
        r"C:/Users/Work Mode Big Dog/OneDrive - ECAM/Bureau/ERASMUS/PROJECT/CODE/DATASETS/DeepFish/Classification/7426/valid",
        r"C:/Users/Work Mode Big Dog/OneDrive - ECAM/Bureau/ERASMUS/PROJECT/CODE/DATASETS/DeepFish/Classification/7434/valid",
        r"C:/Users/Work Mode Big Dog/OneDrive - ECAM/Bureau/ERASMUS/PROJECT/CODE/DATASETS/DeepFish/Classification/7463/valid",
        r"C:/Users/Work Mode Big Dog/OneDrive - ECAM/Bureau/ERASMUS/PROJECT/CODE/DATASETS/DeepFish/Classification/7482/valid",
        r"C:/Users/Work Mode Big Dog/OneDrive - ECAM/Bureau/ERASMUS/PROJECT/CODE/DATASETS/DeepFish/Classification/7490/valid",
        r"C:/Users/Work Mode Big Dog/OneDrive - ECAM/Bureau/ERASMUS/PROJECT/CODE/DATASETS/DeepFish/Classification/7585/valid",
        r"C:/Users/Work Mode Big Dog/OneDrive - ECAM/Bureau/ERASMUS/PROJECT/CODE/DATASETS/DeepFish/Classification/7623/valid",
        r"C:/Users/Work Mode Big Dog/OneDrive - ECAM/Bureau/ERASMUS/PROJECT/CODE/DATASETS/DeepFish/Classification/9852/valid",
        r"C:/Users/Work Mode Big Dog/OneDrive - ECAM/Bureau/ERASMUS/PROJECT/CODE/DATASETS/DeepFish/Classification/9862/valid",
        r"C:/Users/Work Mode Big Dog/OneDrive - ECAM/Bureau/ERASMUS/PROJECT/CODE/DATASETS/DeepFish/Classification/9866/valid",
        r"C:/Users/Work Mode Big Dog/OneDrive - ECAM/Bureau/ERASMUS/PROJECT/CODE/DATASETS/DeepFish/Classification/9870/valid",
        r"C:/Users/Work Mode Big Dog/OneDrive - ECAM/Bureau/ERASMUS/PROJECT/CODE/DATASETS/DeepFish/Classification/9892/valid",
        r"C:/Users/Work Mode Big Dog/OneDrive - ECAM/Bureau/ERASMUS/PROJECT/CODE/DATASETS/DeepFish/Classification/9894/valid",
        r"C:/Users/Work Mode Big Dog/OneDrive - ECAM/Bureau/ERASMUS/PROJECT/CODE/DATASETS/DeepFish/Classification/9898/valid",
        r"C:/Users/Work Mode Big Dog/OneDrive - ECAM/Bureau/ERASMUS/PROJECT/CODE/DATASETS/DeepFish/Classification/9907/valid",
        r"C:/Users/Work Mode Big Dog/OneDrive - ECAM/Bureau/ERASMUS/PROJECT/CODE/DATASETS/DeepFish/Classification/9908/valid"
    ]
    
    # 2. Where should the filtered images go?
    CURATED_IMAGES_FOLDER = "DEEPFISH TO BE ANNOTATED"
    
    # 3. Where should the final .txt pseudo-labels go?
    FINAL_LABELS_FOLDER = "DEEPFISH Output PseudoLabels"
    
    # --- Run Step 1: Filter ---
    # We set the threshold to 0.85 (85% confidence). You can raise or lower this!
    filter_high_confidence_images(
        model_path=MODEL_WEIGHTS, 
        source_folders=RAW_DATA_FOLDERS, 
        curated_img_dir=CURATED_IMAGES_FOLDER, 
        conf_threshold=0.40 
    )
    
    # --- NEW: Safety Check ---
    # Check how many images actually made it into the curated folder
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    copied_files = [f for f in os.listdir(CURATED_IMAGES_FOLDER) if f.lower().endswith(valid_extensions)]
    
    if len(copied_files) == 0:
        print("ERROR: The curated folder is empty! Try lowering the conf_threshold even more.")
        return # Stops the script gracefully before YOLO crashes
    
    print(f"Found {len(copied_files)} images in the curated folder. Moving to Step 2...")

    # --- Run Step 2: Annotate ---
    generate_pseudo_labels(
        model_path=MODEL_WEIGHTS, 
        images_dir=CURATED_IMAGES_FOLDER, 
        output_dir=FINAL_LABELS_FOLDER
    )

if __name__ == '__main__':
    main()