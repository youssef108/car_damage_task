import os
import cv2

# Paths to images and labels directories
images_dir = 'C:\Slynder_Task\Car_damage_dataset_split/train\images'
labels_dir = 'C:\Slynder_Task\Car_damage_dataset_split/train\labels'

# Get a list of all image files in the images directory
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

valid_images_data = []  # To store tuples of (image_path, bounding_boxes)

for img_file in image_files:
    img_path = os.path.join(images_dir, img_file)
    label_file_name = os.path.splitext(img_file)[0] + '.txt'
    label_path = os.path.join(labels_dir, label_file_name)

    if not os.path.exists(img_path):
        continue
    
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    height, width = img.shape[:2]
    bounding_boxes = []

    # Check label file for bounding boxes
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Only consider detection format lines with exactly 5 values
                if len(parts) == 5:
                    cls_id, x_center, y_center, w, h = parts
                    cls_id = int(cls_id)
                    x_center = float(x_center)
                    y_center = float(y_center)
                    w = float(w)
                    h = float(h)

                    # Convert normalized coordinates to pixel values
                    x_center_pixel = x_center * width
                    y_center_pixel = y_center * height
                    w_pixel = w * width
                    h_pixel = h * height

                    x_min = int(x_center_pixel - w_pixel / 2)
                    y_min = int(y_center_pixel - h_pixel / 2)
                    x_max = int(x_center_pixel + w_pixel / 2)
                    y_max = int(y_center_pixel + h_pixel / 2)
                    
                    bounding_boxes.append((cls_id, x_min, y_min, x_max, y_max))

    # If this image has at least one bounding box, store it
    if bounding_boxes:
        valid_images_data.append((img_path, bounding_boxes))

# Print the number of valid images (with at least one bounding box)
print(f"Number of valid images with bounding boxes: {len(valid_images_data)}")

# Now visualize only those images with bounding boxes
for img_path, bounding_boxes in valid_images_data:
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    for (cls_id, x_min, y_min, x_max, y_max) in bounding_boxes:
        color = (0, 255, 0)  # Green bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Put class id text above the box
        class_text = str(cls_id)
        (text_width, text_height), baseline = cv2.getTextSize(
            class_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(
            img,
            (x_min, y_min - text_height - baseline),
            (x_min + text_width, y_min),
            color,
            thickness=cv2.FILLED
        )
        cv2.putText(
            img,
            class_text,
            (x_min, y_min - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),  # Black text inside green background
            1,
            cv2.LINE_AA
        )

    cv2.imshow("Image with Bounding Boxes and Class IDs", img)
    key = cv2.waitKey(0)
    if key == 27:  # Press ESC to close early
        break

cv2.destroyAllWindows()
