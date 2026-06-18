from ultralytics import YOLO
import os

def run_phase_2_only():
    print(" Starting Phase 2: Fine-tuning and generalization with SGD...")
    
    # 1. Point directly to the absolute path from your console output
    phase1_weights = r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\ultralytics\runs\pose\BezierFusion\Phase1_AdamW\weights\last.pt"
    
    if not os.path.exists(phase1_weights):
        print(f" Error: Still can't find the file at {phase1_weights}. Double check the spelling!")
        return
        
    # 2. Load the weights AdamW just finished making
    model_phase2 = YOLO(phase1_weights)
    
    # 3. Run the final 50 epochs with SGD
    model_phase2.train(
        data='fish_bezier.yaml', 
        epochs=50,               # The remaining 10%
        batch=5,                 
        imgsz=640,
        optimizer='SGD',         # Exploring flatter, generalized minima
        lr0=1e-4,                # CRITICAL: Slashed learning rate
        weight_decay=0.0005,     
        cos_lr=True,             
        lrf=0.01,
        
        # Keep all augmentations and loss weights identical
        scale=0.5, translate=0.1, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        box=7.5, dfl=1.5, pose=12.0, cls=3.0,
        
        # Force it to save right next to Phase 1
        project=r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\ultralytics\runs\pose\BezierFusion",
        name='Phase2_SGD',
        val=False
    )
    
    print("\n Phase 2 Complete! Your final, fine-tuned weights are ready.")

if __name__ == '__main__':
    run_phase_2_only()