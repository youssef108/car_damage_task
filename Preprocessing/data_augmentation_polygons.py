import os
import cv2
import random
import albumentations as A
from tqdm import tqdm

# Define paths
train_images_dir = r"C:\Slynder_Task\Car_paint_damage_detection.v10-car-damage-dataset.yolov11\train\images"
train_labels_dir = r"C:\Slynder_Task\Car_paint_damage_detection.v10-car-damage-dataset.yolov11\train\labels"
aug_images_dir = r"C:\Slynder_Task\Car_paint_damage_detection.v10-car-damage-dataset_augmented.yolov11\train_aug\images"
aug_labels_dir = r"C:\Slynder_Task\Car_paint_damage_detection.v10-car-damage-dataset_augmented.yolov11\train_aug\labels"

os.makedirs(aug_images_dir, exist_ok=True)
os.makedirs(aug_labels_dir, exist_ok=True)

# Define augmentation pipelines that handle keypoints (polygons)
augmentation_pipeline1 = A.Compose(
    [
        A.HorizontalFlip(p=1.0),
        A.RandomBrightnessContrast(p=0.5),
    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
)

augmentation_pipeline2 = A.Compose(
    [
        A.Rotate(limit=15, p=1.0),
        A.Equalize(p=0.5),
    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
)

# New third pipeline
augmentation_pipeline3 = A.Compose(
    [
        A.VerticalFlip(p=1.0),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
)

# Include the new pipeline in the list
augmentation_pipelines = [augmentation_pipeline1, augmentation_pipeline2, augmentation_pipeline3]

def read_yolo_polygons(label_file):
    polygons, labels = [], []
    if not os.path.exists(label_file):
        return polygons, labels

    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            # At least: class_id + (x,y)*3 = 1 + 6 = 7 values
            if len(parts) >= 7:
                try:
                    label = int(parts[0])
                    polygon_coords = list(map(float, parts[1:]))
                    polygons.append(polygon_coords)
                    labels.append(label)
                except ValueError:
                    # Malformed line, skip it
                    continue
    return polygons, labels

def save_yolo_polygons(label_file, polygons, labels):
    """
    Saves polygons in YOLO polygon format:
    class_id x1 y1 x2 y2 ... xn yn
    Ensures the directory exists before writing.
    """
    os.makedirs(os.path.dirname(label_file), exist_ok=True)
    with open(label_file, "w") as f:
        for polygon, label in zip(polygons, labels):
            polygon_str = " ".join(map(str, polygon))
            f.write(f"{label} {polygon_str}\n")

def polygons_to_keypoints(polygons):
    keypoints = []
    for polygon in polygons:
        pts = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
        keypoints.append(pts)
    return keypoints

def keypoints_to_polygons(keypoints_list):
    polygons = []
    for pts in keypoints_list:
        polygon = [coord for point in pts for coord in point]
        polygons.append(polygon)
    return polygons

for image_file in tqdm(os.listdir(train_images_dir), desc="Augmenting images"):
    image_path = os.path.join(train_images_dir, image_file)
    label_path = os.path.join(train_labels_dir, image_file.replace(".jpg", ".txt"))

    image = cv2.imread(image_path)
    if image is None:
        continue

    height, width = image.shape[:2]

    polygons, class_labels = read_yolo_polygons(label_path)
    if not polygons:
        continue

    # Convert YOLO normalized coords to pixel coords before augmentation
    polygons_keypoints = polygons_to_keypoints(polygons)
    pixel_polygons_keypoints = [
        [(x * width, y * height) for (x, y) in poly]
        for poly in polygons_keypoints
    ]

    # Save original (already normalized) labels and image
    original_image_path = os.path.join(aug_images_dir, image_file)
    original_label_path = os.path.join(aug_labels_dir, image_file.replace(".jpg", ".txt"))
    try:
        save_yolo_polygons(original_label_path, polygons, class_labels)
        cv2.imwrite(original_image_path, image)
    except Exception as e:
        print(f"Error saving original labels or image for {image_file}: {e}")
        continue

    # Generate two augmented versions per image
    for i in range(3):
        # Flatten pixel keypoints for Albumentations
        all_keypoints_flat = [p for poly in pixel_polygons_keypoints for p in poly]

        # Select a random augmentation pipeline (now we have 3)
        aug_pipeline = random.choice(augmentation_pipelines)
        augmented = aug_pipeline(image=image, keypoints=all_keypoints_flat)

        aug_image = augmented['image']
        aug_keypoints_flat = augmented['keypoints']

        # Re-group augmented keypoints
        aug_polygons_list = []
        offset = 0
        for poly in pixel_polygons_keypoints:
            poly_length = len(poly)
            aug_poly_points = aug_keypoints_flat[offset:offset+poly_length]
            offset += poly_length
            aug_polygons_list.append(aug_poly_points)

        # Convert pixel coords back to normalized YOLO coords and clamp them
        normalized_aug_polygons_list = []
        for aug_poly in aug_polygons_list:
            normalized_poly = []
            for (x_px, y_px) in aug_poly:
                nx = x_px / width
                ny = y_px / height
                # Clamp to [0, 1]
                nx = min(max(nx, 0.0), 1.0)
                ny = min(max(ny, 0.0), 1.0)
                normalized_poly.append((nx, ny))
            normalized_aug_polygons_list.append(normalized_poly)

        # Convert back to YOLO polygon format
        aug_polygons = keypoints_to_polygons(normalized_aug_polygons_list)

        # Sanitize filename
        safe_image_file = image_file.replace('/', '_').replace('\\', '_')
        aug_image_path = os.path.join(aug_images_dir, f"aug_{i}_{safe_image_file}")
        aug_label_path = os.path.join(aug_labels_dir, f"aug_{i}_{safe_image_file.replace('.jpg', '.txt')}")

        # Save labels first (normalized), then image
        try:
            save_yolo_polygons(aug_label_path, aug_polygons, class_labels)
            cv2.imwrite(aug_image_path, aug_image)
        except Exception as e:
            print(f"Error saving augmented labels or image for {image_file}: {e}")
            continue
