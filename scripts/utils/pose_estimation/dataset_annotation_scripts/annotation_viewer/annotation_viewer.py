import cv2
import numpy as np
import os

def process_and_save_all(img_dir, label_dir, output_dir):
    # 1. Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Get list of all image files
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(img_files)} images. Starting processing...")

    for filename in img_files:
        img_path = os.path.join(img_dir, filename)
        label_path = os.path.join(label_dir, filename.rsplit('.', 1)[0] + ".txt")

        # Skip if label doesn't exist
        if not os.path.exists(label_path):
            print(f"Warning: Label not found for {filename}, skipping.")
            continue

        img = cv2.imread(img_path)
        if img is None: continue
        h, w, _ = img.shape

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = list(map(float, line.strip().split()))
            if len(data) < 5: continue

            # --- Draw Bounding Box ---
            cx, cy, bw, bh = data[1:5]
            x1, y1 = int((cx - bw/2) * w), int((cy - bh/2) * h)
            x2, y2 = int((cx + bw/2) * w), int((cy + bh/2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

            # --- Extract Keypoints (Step by 3 to skip the "2") ---
            pts = []
            raw_points = data[5:]
            for i in range(0, len(raw_points), 3):
                if i + 1 < len(raw_points):
                    px, py = int(raw_points[i] * w), int(raw_points[i+1] * h)
                    pts.append((px, py))

            # --- Draw Bézier Curve (P1 to P4) ---
            if len(pts) >= 4:
                p1, p2, p3, p4 = pts[:4]
                curve_points = []
                for t in np.linspace(0, 1, 30):
                    # Cubic Bezier Formula
                    bx = (1-t)**3*p1[0] + 3*(1-t)**2*t*p2[0] + 3*(1-t)*t**2*p3[0] + t**3*p4[0]
                    by = (1-t)**3*p1[1] + 3*(1-t)**2*t*p2[1] + 3*(1-t)*t**2*p3[1] + t**3*p4[1]
                    curve_points.append([int(bx), int(by)])
                
                pts_array = np.array(curve_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts_array], False, (255, 255, 0), 2)

            # --- Draw Keypoints ---
            for i, (px, py) in enumerate(pts):
                color = (0, 255 - (i*60), i*60)
                cv2.circle(img, (px, py), 6, color, -1)
                cv2.putText(img, f"P{i+1}", (px+5, py-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,255,255), 3)

        # 2. Save the result
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, img)
        
    print(f"Done! All results saved in: {output_dir}")

# # --- EXECUTION ---
# process_and_save_all(
#     img_dir="DATASETS\DeepFish\Segmentation\images\valid", 
#     label_dir="DATASETS\DeepFish\Segmentation\masks\valid_labels", 
#     output_dir="Bezierfusion Dataset 2 Reannotated V2 results"
# )
# process_and_save_all(
#     img_dir="dataset1/dataset1/rgb", 
#     label_dir="dataset1/dataset1/labels", 
#     output_dir="dataset1/dataset1/results"
# )

# process_and_save_all(
#     img_dir="valid", 
#     label_dir="valid_labels_V2", 
#     output_dir="Deepfish Annotated  results V2"
# )

# process_and_save_all(
#     img_dir="DEEPFISH TO BE ANNOTATED", 
#     label_dir="DEEPFISH Output PseudoLabels\curated_pseudo_labels\labels", 
#     output_dir="Deepfish Annotated  results V3"
# )

# process_and_save_all(
#     img_dir="rgb", 
#     label_dir="yolo_labels_overlap_safe_V4", 
#     output_dir="Bezierfusion Dataset 2 Reannotated V4 results"
# )

# process_and_save_all(
#     img_dir="DATASETS\DeepFish\First_batch_training", 
#     label_dir="yolo_labels_overlap_safe_deepfish_V1", 
#     output_dir="Deepfish Dataset First Batch results"
# )

process_and_save_all(
    img_dir=r"DATASETS\DATASETS FOR POSE ESTIMATION\Bezierfusion Dataset 2 Reannotated V4\rgb", 
    label_dir=r"DATASETS\DATASETS FOR POSE ESTIMATION\Bezierfusion Dataset 2 Reannotated V4\yolo_labels_overlap_safe_V4", 
    output_dir=r"DATASETS\DATASETS FOR POSE ESTIMATION\Bezierfusion Dataset 2 Reannotated V4\Annotations_Check"
)