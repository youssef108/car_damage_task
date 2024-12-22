import os
import shutil
import random
from collections import defaultdict
from tqdm import tqdm

# Paths
data_dir = 'C:\Slynder_Task\Car_damage_datset/train'  # Directory containing images and labels
output_dir = 'C:\Slynder_Task\Car_damage_dataset_split'  # Directory for the split dataset
image_ext = ['.jpg', '.png', '.jpeg']
val_ratio = 0.2
test_ratio = 0.1

# Ensure output directories exist
subdirs = ['train', 'val', 'test']
for subdir in subdirs:
    os.makedirs(os.path.join(output_dir, subdir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, subdir, 'labels'), exist_ok=True)

def group_images_by_class(labels_dir, images_dir):
    """
    Groups images by the classes found in their labels.
    If an image contains multiple classes, it will appear in multiple class lists.
    """
    class_to_images = defaultdict(list)
    label_files = [lf for lf in os.listdir(labels_dir) if lf.endswith('.txt')]
    for label_file in tqdm(label_files, desc="Grouping images by class"):
        label_path = os.path.join(labels_dir, label_file)
        image_file_root = os.path.splitext(label_file)[0]

        # Find corresponding image by checking supported extensions
        image_file = None
        for ext in image_ext:
            candidate = f"{image_file_root}{ext}"
            if os.path.exists(os.path.join(images_dir, candidate)):
                image_file = candidate
                break
        if image_file is None:
            continue  # No corresponding image found
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            class_ids = [int(line.strip().split()[0]) for line in lines if line.strip()]
        
        # Add this image to each class it contains
        for cid in set(class_ids):
            class_to_images[cid].append((image_file, label_file))
    return class_to_images

images_dir = os.path.join(data_dir, 'images')
labels_dir = os.path.join(data_dir, 'labels')

class_to_images = group_images_by_class(labels_dir, images_dir)

splits = {'train': [], 'val': [], 'test': []}

# Keep track of which images have already been assigned to prevent leakage
assigned_images = set()

# Split data by class
for class_id, files in tqdm(class_to_images.items(), desc="Splitting dataset by class"):
    random.shuffle(files)
    total_files = len(files)
    val_count = int(total_files * val_ratio)
    test_count = int(total_files * test_ratio)
    train_count = total_files - val_count - test_count

    # Define the split ranges
    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]

    # Assign images to splits if not already assigned
    for (image_file, label_file) in train_files:
        if image_file not in assigned_images:
            splits['train'].append((image_file, label_file))
            assigned_images.add(image_file)
    for (image_file, label_file) in val_files:
        if image_file not in assigned_images:
            splits['val'].append((image_file, label_file))
            assigned_images.add(image_file)
    for (image_file, label_file) in test_files:
        if image_file not in assigned_images:
            splits['test'].append((image_file, label_file))
            assigned_images.add(image_file)

# Shuffle the splits themselves
for split in splits:
    random.shuffle(splits[split])

# Validate no data leakage
all_files = set()
for split, files in splits.items():
    for image_file, _ in files:
        if image_file in all_files:
            raise ValueError(f"Data leakage detected: {image_file} exists in multiple splits!")
        all_files.add(image_file)

# Copy files to respective directories
for split, files in splits.items():
    for image_file, label_file in tqdm(files, desc=f"Copying files to {split} folder"):
        image_src = os.path.join(images_dir, image_file)
        label_src = os.path.join(labels_dir, label_file)
        
        image_dst = os.path.join(output_dir, split, 'images', image_file)
        label_dst = os.path.join(output_dir, split, 'labels', label_file)
        
        shutil.copy(image_src, image_dst)
        shutil.copy(label_src, label_dst)

print(f"Splitting complete with no data leakage! Results saved in '{output_dir}'.")
