import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm

def extract_embeddings(model, dataloader, device):
    """Passes images through DINOv2 to get their mathematical embeddings."""
    embeddings = []
    labels = []
    
    with torch.no_grad(): # No training, just analyzing
        for images, targets in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)
            # DINOv2 outputs a rich 384-dimensional vector for each image
            features = model(images) 
            
            embeddings.append(features.cpu().numpy())
            labels.append(targets.numpy())
            
    return np.vstack(embeddings), np.concatenate(labels)

def run_dino_pipeline(data_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load the Pre-trained DINOv2 Foundation Model (Vision Transformer Small)
    print("Downloading/Loading DINOv2...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    dinov2.eval() # Set to evaluation mode

    # 2. Prepare the Images (DINOv2 expects 224x224 images with specific normalization)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load your Train and Val folders
    train_dataset = ImageFolder(root=f"{data_dir}/train", transform=transform)
    val_dataset = ImageFolder(root=f"{data_dir}/val", transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    class_names = train_dataset.classes

    # 3. Extract Embeddings
    print("\nAnalyzing Training Data...")
    X_train, y_train = extract_embeddings(dinov2, train_loader, device)
    
    print("\nAnalyzing Validation Data...")
    X_val, y_val = extract_embeddings(dinov2, val_loader, device)

    # 4. Train a simple K-Nearest Neighbors Classifier (Takes seconds)
    print("\nFitting KNN Classifier...")
    # Look at the 5 closest mathematical matches to determine the species
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine') 
    knn.fit(X_train, y_train)

    # 5. Evaluate Accuracy
    print("\nEvaluating Model...")
    predictions = knn.predict(X_val)
    
    accuracy = accuracy_score(y_val, predictions)
    print(f"\nOverall Validation Accuracy: {accuracy * 100:.2f}%\n")
    
    # Print a detailed breakdown per species
    print(classification_report(y_val, predictions, target_names=class_names))

if __name__ == '__main__':
    # Point this to your Train/Val root folder
    DATA_DIRECTORY = r"DATASETS\DATASETS FOR SPECIES CLASSIFICATION\Fish4knowledge\Fish_Species_Classifier_Data_DINO" 
    run_dino_pipeline(DATA_DIRECTORY)