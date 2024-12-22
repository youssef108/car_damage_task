import os
import cv2
import random
import albumentations as A
from tqdm import tqdm

# Define paths
train_images_dir = "C:\Slynder_Task\Car_damage_dataset_split/train\images"
train_labels_dir = "C:\Slynder_Task\Car_damage_dataset_split/train\labels"
aug_images_dir = "C:\Slynder_Task\Car_damage_dataset_split_Augmented/train_aug\images"
aug_labels_dir = "C:\Slynder_Task\Car_damage_dataset_split_Augmented/train_aug\labels"


# Create directories for augmented data
os.makedirs(aug_images_dir, exist_ok=True)
os.makedirs(aug_labels_dir, exist_ok=True)

augmentation_pipeline1 = A.Compose([
    A.HorizontalFlip(p=1.0),
    A.RandomBrightnessContrast(p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], check_each_transform=True,clip=True))

augmentation_pipeline2 = A.Compose([
    A.Rotate(limit=15, p=1.0),
    A.Equalize(p=0.5)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], check_each_transform=True, clip=True))

augmentation_pipelines = [augmentation_pipeline1, augmentation_pipeline2]

# Function to read YOLO labels
def read_yolo_labels(label_file):
    bboxes, labels = [], []
    with open(label_file, "r") as f:
        for line in f:
            # Split the line into components and ensure it's in YOLO format
            components = line.strip().split()
            if len(components) >= 5:  # Ensure enough data for YOLO format
                try:
                    label = int(components[0])  # Class ID
                    x_center, y_center, width, height = map(float, components[1:5])
                    bboxes.append([x_center, y_center, width, height])
                    labels.append(label)
                except ValueError:
                    continue  # Skip malformed lines
    return bboxes, labels

# Function to save YOLO labels
def save_yolo_labels(label_file, bboxes, labels):
    with open(label_file, "w") as f:
        for bbox, label in zip(bboxes, labels):
            f.write(f"{label} {' '.join(map(str, bbox))}\n")

# Perform augmentation
for image_file in tqdm(os.listdir(train_images_dir), desc="Augmenting images"):
    image_path = os.path.join(train_images_dir, image_file)
    label_path = os.path.join(train_labels_dir, image_file.replace(".jpg", ".txt"))

    # Read image and labels
    image = cv2.imread(image_path)
    bboxes, class_labels = read_yolo_labels(label_path)

    # Skip files with no valid bounding boxes
    if not bboxes:
        continue

    # Save the original image and labels
    original_image_path = os.path.join(aug_images_dir, image_file)
    original_label_path = os.path.join(aug_labels_dir, image_file.replace(".jpg", ".txt"))
    cv2.imwrite(original_image_path, image)
    save_yolo_labels(original_label_path, bboxes, class_labels)

    # Generate two augmented versions
    for i in range(2):  # Two augmentations per image
        # Randomly select an augmentation pipeline
        aug_pipeline = random.choice(augmentation_pipelines)
        
        # Apply augmentation
        augmented = aug_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_image = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_labels = augmented['class_labels']

        # Save augmented image and labels
        aug_image_path = os.path.join(aug_images_dir, f"aug_{i}_{image_file}")
        aug_label_path = os.path.join(aug_labels_dir, f"aug_{i}_{image_file.replace('.jpg', '.txt')}")
        cv2.imwrite(aug_image_path, aug_image)
        save_yolo_labels(aug_label_path, aug_bboxes, aug_labels)
