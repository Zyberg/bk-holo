import cv2
import numpy as np
import csv
import os

def load_metrics(metrics_file):
    z_values = []
    focus_metrics = []
    filenames = []
    with open(metrics_file, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            z_values.append(float(row[0]))
            focus_metrics.append(float(row[1]))
            filenames.append(row[2])
    return z_values, focus_metrics, filenames

def update_display(val, images_dir, filenames, focus_metrics):
    image_path = os.path.join(images_dir, filenames[val])
    image = np.load(image_path)
    cv2.imshow('Focused Image', image)
    metric = focus_metrics[val]
    cv2.displayOverlay('Focused Image', f"Focus Metric: {metric:.2f}", 1000)

# Path to the directory containing precomputed images and focus metrics
images_dir = "precomputed_focus_images"
metrics_file = os.path.join(images_dir, "focus_metrics.csv")

# Load z values, focus metrics, and filenames
z_values, focus_metrics, filenames = load_metrics(metrics_file)

# Set up the display window with a trackbar
cv2.namedWindow('Focused Image', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Image Index', 'Focused Image', 0, len(z_values) - 1, lambda val: update_display(val, images_dir, filenames, focus_metrics))

# Initialize display with the first image
update_display(0, images_dir, filenames, focus_metrics)

# Display loop, exit with 'q'
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
