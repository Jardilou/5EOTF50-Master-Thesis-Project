import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_dino_embeddings(model, transform, image_rgb, device):
    """Passes a single cropped image through DINO to get the 384-d vector."""
    img_tensor = transform(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding.cpu().numpy()

def evaluate_target_species_linear(original_train_dir, custom_images_dir, custom_labels_dir):
    # The exact 5 targets we want to evaluate
    TARGET_SPECIES = [
        "Dascyllus_reticulatus", 
        "Hemigymnus_melapterus", 
        "Pomacentrus_moluccensis", 
        "Abudefduf_vaigiensis", 
        "Scaridae"
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading DINOv2 on {device}...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    dinov2.eval()

    # --- THE FIX: Split Transforms ---
    # 1. Transform for ImageFolder (Already PIL format)
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Transform for OpenCV Crops (Numpy format)
    inference_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # ---------------------------------

    # 1. Extract Embeddings from the Original DINO Train set
    print("Extracting foundation embeddings from Training set...")
    train_dataset = ImageFolder(root=original_train_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    
    X_train, y_train = [], []
    with torch.no_grad():
        for images, targets in tqdm(train_loader, desc="Extracting Features"):
            features = dinov2(images.to(device))
            X_train.append(features.cpu().numpy())
            y_train.append(targets.numpy())
            
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    database_classes = train_dataset.classes

    # 2. Train the Linear Classifier 
    print("\nTraining Balanced Linear Classifier...")
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(X_train, y_train)
    print("Classifier trained successfully!")

    # 3. Run Inference ONLY on Targets in the Custom Dataset
    print("\nRunning Inference on Custom Bounding Boxes...")
    y_true, y_pred = [], []
    correct_images, incorrect_images = [], [] 

    for label_file in tqdm(os.listdir(custom_labels_dir)):
        if not label_file.endswith('.txt'): continue
        
        img_path = os.path.join(custom_images_dir, label_file.replace('.txt', '.jpg'))
        if not os.path.exists(img_path): continue
            
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        with open(os.path.join(custom_labels_dir, label_file), 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            true_species = parts[0]
            
            # STRICT FILTER: Only evaluate if the label is one of our 5 targets
            if true_species not in TARGET_SPECIES: 
                continue
            
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            
            crop = img_rgb[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0: continue
                
            # Use the inference_transform for the OpenCV crop
            emb = get_dino_embeddings(dinov2, inference_transform, crop, device)
            pred_idx = clf.predict(emb)[0]
            pred_species = database_classes[pred_idx]
            
            y_true.append(true_species)
            y_pred.append(pred_species)
            
            vis_data = (crop, true_species, pred_species)
            if true_species == pred_species and len(correct_images) < 9:
                correct_images.append(vis_data)
            elif true_species != pred_species and len(incorrect_images) < 9:
                incorrect_images.append(vis_data)

    # 4. Generate Target-Specific Metrics
    print("\n" + "="*50)
    print("TARGET SPECIES CLASSIFICATION REPORT (LINEAR PROBE)")
    print("="*50)
    print(classification_report(y_true, y_pred, zero_division=0))

    all_present_classes = sorted(list(set(y_true + y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=all_present_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=all_present_classes, yticklabels=all_present_classes)
    plt.title('DINO Linear Probe Confusion Matrix (Targeted)')
    plt.xlabel('Predicted Species')
    plt.ylabel('Actual Species')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('targeted_confusion_matrix.png')
    print("\nSaved 'targeted_confusion_matrix.png'")

    # 5. Generate Visuals
    def plot_grid(images_list, title, filename):
        if not images_list: return
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle(title, fontsize=16)
        for i, ax in enumerate(axes.flat):
            if i < len(images_list):
                crop, t_spec, p_spec = images_list[i]
                ax.imshow(crop)
                ax.set_title(f"T: {t_spec[:10]}..\nP: {p_spec[:10]}..", color='green' if t_spec==p_spec else 'red')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Saved '{filename}'")

    plot_grid(correct_images, "Target Correct Predictions", "targeted_correct.png")
    plot_grid(incorrect_images, "Target Hallucinations/Errors", "targeted_incorrect.png")


# ==========================================
# Execute the Pipeline
# ==========================================
if __name__ == '__main__':
    # 1. Point to your DINO Train folder (to teach the linear probe)
    DINO_TRAIN_DIR = r"DATASETS\DATASETS FOR SPECIES CLASSIFICATION\Fish_Species_Classifier_Data_DINO_All_Samples_F4N+UMT\train" 
    
    # 2. Point to the Custom DeepFish images
    CUSTOM_IMAGES = r"DATASETS\DATASETS FOR SPECIES CLASSIFICATION\UMT For Validation\Images"
    
    # 3. Point to the labels generated by Script 1 (with string names)
    CUSTOM_LABELS = r"DATASETS\DATASETS FOR SPECIES CLASSIFICATION\UMT For Validation\updated-labels-with-species"
    
    evaluate_target_species_linear(DINO_TRAIN_DIR, CUSTOM_IMAGES, CUSTOM_LABELS)
