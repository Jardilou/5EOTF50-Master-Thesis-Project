import os

def reset_classes_to_zero(labels_dir):
    """
    Forces all YOLO polygon labels to belong to class 0 (Single-class model).
    """
    for root, _, files in os.walk(labels_dir):
        for file in files:
            if file.endswith('.txt'):
                filepath = os.path.join(root, file)
                
                # Read the current lines
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                
                # Overwrite the file with class 0
                with open(filepath, 'w') as f:
                    for line in lines:
                        parts = line.strip().split()
                        if not parts: continue
                        
                        # Replace the first element (class ID) with '0'
                        parts[0] = '0'
                        f.write(" ".join(parts) + "\n")

    print(f"All labels in {labels_dir} have been reset to Class 0!")

# ==========================================
# Run the Cleanup
# ==========================================
# Point this to the root of your split YOLO dataset labels
TRAIN_LABELS = r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\DATASETS\DATASETS FOR INSTANCE SEGMENTATION\Fish4knowledge\split_dataset\labels\train"
VAL_LABELS = r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\DATASETS\DATASETS FOR INSTANCE SEGMENTATION\Fish4knowledge\split_dataset\labels\val"

reset_classes_to_zero(TRAIN_LABELS)
reset_classes_to_zero(VAL_LABELS)