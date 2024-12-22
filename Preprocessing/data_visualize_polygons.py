import os
import cv2
import numpy as np

images_dir = r"C:\Slynder_Task\Car_paint_damage_detection.v10-car-damage-dataset_augmented.yolov11\train_aug\images"
labels_dir = r"C:\Slynder_Task\Car_paint_damage_detection.v10-car-damage-dataset_augmented.yolov11\train_aug\labels"



# Get a list of all image files in the images directory
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

valid_images_data = []  # To store tuples of (image_path, polygon_data)

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
    polygons_data = []  # Will store tuples of (cls_id, polygon_points)

    # Check label file for polygons
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Polygon line: class_id + at least 3 points => at least 1 + (3*2) = 7 values
                if len(parts) >= 7:
                    try:
                        cls_id = int(parts[0])
                        coords = list(map(float, parts[1:]))

                        # Convert normalized coords to pixel coords
                        polygon_points = []
                        for i in range(0, len(coords), 2):
                            x_norm = coords[i]
                            y_norm = coords[i+1]
                            x_px = int(x_norm * width)
                            y_px = int(y_norm * height)
                            polygon_points.append((x_px, y_px))

                        polygons_data.append((cls_id, polygon_points))
                    except ValueError:
                        # Skip malformed lines
                        continue

    # If this image has at least one polygon, store it
    if polygons_data:
        valid_images_data.append((img_path, polygons_data))

# Print the number of valid images (with at least one polygon)
print(f"Number of valid images with polygons: {len(valid_images_data)}")

# Now visualize only those images with polygons
for img_path, polygons_data in valid_images_data:
    img = cv2.imread(img_path)
    if img is None:
        continue

    for (cls_id, polygon_points) in polygons_data:
        color = (0, 255, 0)  # Green polygon

        # Convert polygon_points to a numpy array for polylines
        pts = np.array(polygon_points, dtype=np.int32).reshape(-1, 1, 2)

        # If you want the polygon as is, use it directly
        # If you want the convex hull (optional), uncomment next line:
        # pts = cv2.convexHull(pts)

        # Draw the polygon
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

        # Put class id text near the first polygon vertex
        x_min, y_min = polygon_points[0]
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
            (0, 0, 0),  # Black text on green background
            1,
            cv2.LINE_AA
        )

    cv2.imshow("Image with Polygons and Class IDs", img)
    key = cv2.waitKey(0)
    if key == 27:  # Press ESC to close early
        break

cv2.destroyAllWindows()
