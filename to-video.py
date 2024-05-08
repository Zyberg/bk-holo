import cv2
import glob
import os
import re

direction = 'right'

# Directory containing the images
image_dir = f'/home/zyberg/bin/bakalauras/src/results/temporary/reconstructed-intensity/{direction}/'

# Output video file name
output_video = f'twin-{direction}.mp4'

# Get a list of image files sorted by distance
# image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')), key=lambda x: float(re.search(r'(\d+\.\d+)$', os.path.splitext(os.path.basename(x)).group(1))[0]))
image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')), key=lambda x: float(re.search(r'(\d+\.\d+)$', os.path.splitext(os.path.basename(x))[0]).group(1)))

# Define video properties
frame_width = 1920
frame_height = 1440
fps = 2  # Frames per second (adjust as needed)
duration_per_image = 1  # Duration to display each image in seconds

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# Font settings for distance text
font_scale = 2  # Increased font size
font_thickness = 3
font_face = cv2.FONT_HERSHEY_SIMPLEX
text_color = (0, 0, 255)  # Red color (BGR format)

# Iterate through each image and add to video
for image_file in image_files:
    # Read image
    image = cv2.imread(image_file)
    
    # Extract distance from filename
    distance = os.path.splitext(os.path.basename(image_file))[0]
    
    # Add distance text to image
    cv2.putText(image, f'Distance: {distance}', (10, 40), font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # Resize image if needed
    if image.shape[1] != frame_width or image.shape[0] != frame_height:
        image = cv2.resize(image, (frame_width, frame_height))

    # Write image to video for specified duration
    for _ in range(int(fps * duration_per_image)):
        video_writer.write(image)

# Release video writer
video_writer.release()

print(f'Video saved as {output_video}')
