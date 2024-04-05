import numpy as np
from PIL import Image as ImagePIL
from scipy import ndimage

class TwinImageCropper:
    def __init__(self, image):
        self.image = image

    def find_twin_image_cropping_region(self):
        threshold_value = self.__get_threshold_value()
        print(f'Calculated threshold: {threshold_value}',)
        # Apply thresholding to create a binary mask
        binary_mask = self.image.array > threshold_value
        
        # Find bounding boxes of connected components (twin images)
        twin_image_bounding_boxes = self.find_connected_component_bounding_boxes(binary_mask)
        
        # Compute the cropping region around the twin images
        cropping_region = self.compute_cropping_region(twin_image_bounding_boxes)

        return cropping_region

    def find_connected_component_bounding_boxes(self, image_array, min_size=1000):
        labeled_array, num_features = ndimage.label(image_array)

        # Perform connected component analysis
        bounding_boxes = ndimage.find_objects(labeled_array)
        component_areas = ndimage.sum(image_array, labeled_array, range(1, num_features+1))

        # Filter out small components based on the minimum size threshold
        filtered_bounding_boxes = [bbox for bbox, area in zip(bounding_boxes, component_areas) if area >= min_size]

        print(filtered_bounding_boxes)
        return filtered_bounding_boxes

    def compute_cropping_region(self, bounding_boxes, padding=10):
        # Compute the cropping region around the twin images
        min_x, min_y, max_x, max_y = bounding_boxes
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(self.image.array.shape[1], max_x + padding)
        max_y = min(self.image.array.shape[0], max_y + padding)

        return min_x, min_y, max_x, max_y

    def __get_threshold_value(self):
        # Compute the histogram of the image
        hist, _ = np.histogram(self.image.array.ravel(), bins=256, range=[0, 256])
        
        # Compute the cumulative distribution function (CDF) of the histogram
        cdf = hist.cumsum()
        
        # Normalize the CDF
        cdf_normalized = cdf / cdf.max()
        
        # Find the intensity value corresponding to the threshold
        threshold_value = np.argmax(cdf_normalized > 0.5)
        
        return threshold_value