import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix


def get_dino_embeddings(model, transform, image_rgb, device):
    img_tensor = transform(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(img_tensor).cpu().numpy()

def run_dino_analytics(train_dir, custom_images_dir, custom_labels_dir):
    # Ensure these EXACT strings match your YOLO text files
    TARGET_SPECIES = [
        "Dascyllus_reticulatus", "Hemigymnus_melapterus", 
        "Pomacentrus_moluccensis", "Abudefduf_vaigiensis", "Scaridae"
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    dinov2.eval()

    train_transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    inference_transform = T.Compose([T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 1. EXTRACT ALL TRAINING FEATURES ONCE
    print("Extracting ALL training features...")
    dataset = ImageFolder(root=train_dir, transform=train_transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    X_train, y_train = [], []
    with torch.no_grad():
        for imgs, tgts in tqdm(loader):
            X_train.append(dinov2(imgs.to(device)).cpu().numpy())
            y_train.append(tgts.numpy())
            
    X_train, y_train = np.vstack(X_train), np.concatenate(y_train)
    all_classes = np.array(dataset.classes)

    # 2. EXTRACT ALL INFERENCE (CUSTOM) FEATURES ONCE
    print("\nExtracting Inference features...")
    X_test, y_test_true = [], []
    
    for label_file in tqdm(os.listdir(custom_labels_dir)):
        if not label_file.endswith('.txt'): continue
        img_path = os.path.join(custom_images_dir, label_file.replace('.txt', '.jpg'))
        if not os.path.exists(img_path): continue
            
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        with open(os.path.join(custom_labels_dir, label_file), 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                true_species = parts[0]
                
                if true_species not in TARGET_SPECIES: continue
                
                cx, cy, bw, bh = map(float, parts[1:5])
                crop = img_rgb[max(0, int((cy - bh/2)*h)):min(h, int((cy + bh/2)*h)), max(0, int((cx - bw/2)*w)):min(w, int((cx + bw/2)*w))]
                if crop.size > 0:
                    X_test.append(get_dino_embeddings(dinov2, inference_transform, crop, device)[0])
                    y_test_true.append(true_species)

    X_test = np.array(X_test)

    # 3. LOOP: Train and Evaluate from 5 to 23 classes (Tracking PER SPECIES)
    print("\nRunning Class-Scaling Analytics...")
    
    # Dictionary to hold the history of each species for each metric
    metrics_history = {
        'f1': {sp: [] for sp in TARGET_SPECIES},
        'recall': {sp: [] for sp in TARGET_SPECIES},
        'precision': {sp: [] for sp in TARGET_SPECIES}
    }
    
    num_classes_range = list(range(5, len(all_classes) + 1))
    final_23_preds = []

    for k in num_classes_range:
        allowed_classes = all_classes[:k]
        allowed_indices = [i for i, c in enumerate(all_classes) if c in allowed_classes]
        
        mask = np.isin(y_train, allowed_indices)
        X_train_k = X_train[mask]
        y_train_k = y_train[mask]
        
        clf = LogisticRegression(class_weight='balanced', max_iter=1000)
        clf.fit(X_train_k, y_train_k)
        
        preds_k_indices = clf.predict(X_test)
        preds_k_strings = [all_classes[idx] for idx in preds_k_indices]
        
        if k == 23: final_23_preds = preds_k_strings
            
        # Using average=None returns an array of scores in the exact order of TARGET_SPECIES
        f1_vals = f1_score(y_test_true, preds_k_strings, labels=TARGET_SPECIES, average=None, zero_division=0)
        rec_vals = recall_score(y_test_true, preds_k_strings, labels=TARGET_SPECIES, average=None, zero_division=0)
        prec_vals = precision_score(y_test_true, preds_k_strings, labels=TARGET_SPECIES, average=None, zero_division=0)
        
        # Save the scores into our history dictionary
        for i, sp in enumerate(TARGET_SPECIES):
            metrics_history['f1'][sp].append(f1_vals[i])
            metrics_history['recall'][sp].append(rec_vals[i])
            metrics_history['precision'][sp].append(prec_vals[i])

    # 4. PLOT PER-SPECIES GRAPHS
    print("Generating Per-Species Analytics Graphs...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle('DINO Linear Probe: Metric Decay by Target Species (Adding Distractors)', fontsize=16)
    
    metrics = ['f1', 'recall', 'precision']
    titles = ['F1-Score', 'Recall', 'Precision']
    
    for ax_idx, metric_name in enumerate(metrics):
        ax = axes[ax_idx]
        for sp in TARGET_SPECIES:
            ax.plot(num_classes_range, metrics_history[metric_name][sp], label=sp, marker='o', markersize=4)
            
        ax.set_title(titles[ax_idx])
        ax.set_xlabel('Total Training Classes')
        ax.set_xticks(num_classes_range)
        ax.grid(True, alpha=0.4)
        
        # Only put the Y-axis label and Legend on the first chart to keep it clean
        if ax_idx == 0:
            ax.set_ylabel('Score (0.0 to 1.0)')
            ax.legend(loc='lower left', fontsize='small')

    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig('species_performance_curves.png')
    print("Saved 'species_performance_curves.png'")

    # 5. SLICED CONFUSION MATRIX (23-Class Model)
    # We want ALL predicted classes on the X-axis, but only 5 true classes on the Y-axis
    all_predicted_labels = sorted(list(set(final_23_preds + TARGET_SPECIES)))
    
    cm = confusion_matrix(y_test_true, final_23_preds, labels=all_predicted_labels)
    cm_df = pd.DataFrame(cm, index=all_predicted_labels, columns=all_predicted_labels)
    
    # Slice the dataframe to only keep the 5 target rows
    cm_df_sliced = cm_df.loc[TARGET_SPECIES, :]
    # Drop columns that were never predicted to save space
    cm_df_sliced = cm_df_sliced.loc[:, (cm_df_sliced != 0).any(axis=0)] 

    plt.figure(figsize=(14, 6))
    sns.heatmap(cm_df_sliced, annot=True, fmt='d', cmap='Blues')
    plt.title('Sliced Confusion Matrix (True Targets vs. All Predictions)')
    plt.xlabel('Predicted Species')
    plt.ylabel('Actual Target Species')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('sliced_confusion_matrix.png')
    print("Saved 'sliced_confusion_matrix.png'")
    
if __name__ == '__main__':
# 1. Point to your DINO Train folder (to teach the linear probe)
    DINO_TRAIN_DIR = r"DATASETS\DATASETS FOR SPECIES CLASSIFICATION\Fish4knowledge\Fish_Species_Classifier_Data_DINO\train" 
    
    # 2. Point to the Custom DeepFish images
    CUSTOM_IMAGES = r"DATASETS\DATASETS FOR SPECIES CLASSIFICATION\UMT For Validation\Images"
    
    # 3. Point to the labels generated by Script 1 (with string names)
    CUSTOM_LABELS = r"DATASETS\DATASETS FOR SPECIES CLASSIFICATION\UMT For Validation\updated-labels-with-species"
    
    # Execute
    run_dino_analytics(DINO_TRAIN_DIR, CUSTOM_IMAGES, CUSTOM_LABELS )