import cv2
import glob
import os
import re
import numpy as np


def create_video_from_npy(directory, output_video_path, frame_width, frame_height, fps=1):
    # Prepare the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # List all .npy files
    files = sorted(glob.glob(os.path.join(directory, '*.npy')))
    
    for file in files:
        # Load the complex field from the file
        complex_field = np.load(file)
        
        # Compute the intensity image
        intensity_image = np.abs(complex_field)**2
        
        # Normalize the intensity image to fit in 8-bit range
        normalized_image = cv2.normalize(intensity_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        normalized_image = normalized_image.astype(np.uint8)

        # Resize image to desired dimensions
        resized_image = cv2.resize(normalized_image, (frame_width, frame_height))
        
        # Convert grayscale to BGR for video
        video_frame = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
        
        # Write to video
        video.write(video_frame)
    
    # Release the video writer
    video.release()
    print(f'Video saved to {output_video_path}')

# Example usage
directory = '/home/zyberg/bin/bakalauras/old_backup/reconstructed-intensity/'
output_video_path = 'output_video.mp4'
frame_width = 1920
frame_height = 1440
create_video_from_npy(directory, output_video_path, frame_width, frame_height)
