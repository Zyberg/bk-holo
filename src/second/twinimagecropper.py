import numpy as np
from PIL import Image as ImagePIL
from scipy import ndimage
from skimage.filters import threshold_otsu, try_all_threshold
from skimage.feature import peak_local_max

class TwinImageCropper:
    def __init__(self, image):
        self.image = image

    def find_intensity_spots(self):
        image = np.log(1 + np.abs(self.image.get_shifted()))
        image = image - np.min(image)

        masked_image, selected_coordinates = self.find_bright_spots(image)
        
        sorted_coordinates = sorted(selected_coordinates, key=lambda x: x[0])
        radius = self.calculate_radius(sorted_coordinates)

        return sorted_coordinates, radius


    def calculate_radius(self, coords):
        distances = [np.linalg.norm(coords[i] - coords[j]) for i in range(3) for j in range(i+1, 3)]

        radius = min(distances) / 2

        return radius

    def find_bright_spots(self, image):
        # Compute the Otsu threshold
        thresh = threshold_otsu(image)
        binary_map = image > thresh

        # Define the region around the center based on the binary map
        center_y, center_x = np.array(image.shape) // 2
        radius = image.shape[0] // 2  # Adjust the radius as needed
        y, x = np.ogrid[-center_y:image.shape[0] - center_y, -center_x:image.shape[1] - center_x]
        mask = x**2 + y**2 <= radius**2
        region_of_interest = mask & binary_map 

        # Apply Otsu's method within the region of interest
        threshold_region = threshold_otsu(image[region_of_interest])

        # Create binary map for the region of interest using the second threshold
        binary_map_region = np.zeros_like(image, dtype=bool)
        binary_map_region[region_of_interest] = image[region_of_interest] > threshold_region

        # Apply binary map on the original image
        masked_image = image.copy()
        masked_image[~binary_map_region] = 0  # Set non-binary regions to 0

        # Find local maxima in the masked image
        coordinates = peak_local_max(masked_image, min_distance=5)

        # Sort coordinates by intensity value
        sorted_coordinates = coordinates[np.argsort(masked_image[coordinates[:, 0], coordinates[:, 1]])[::-1]]

        # Select the three highest peaks that are sufficiently separated
        num_peaks = 3
        selected_coordinates = [sorted_coordinates[0]]
        for coord in sorted_coordinates[1:]:
            if all(np.linalg.norm(coord - selected_coord) > 10 for selected_coord in selected_coordinates):
                selected_coordinates.append(coord)
            if len(selected_coordinates) == num_peaks:
                break

        return masked_image, selected_coordinates


    

        # fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
        # ax = axes.ravel()
        # ax[0] = plt.subplot(1, 3, 1)
        # ax[1] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

        # ax[0].imshow(image, cmap=plt.cm.gray)
        # ax[0].set_title('Original')
        # ax[0].axis('off')

        # ax[1].imshow(masked_image, cmap=plt.cm.gray)
        # selected_coordinates = np.array(selected_coordinates)
        # plt.scatter(selected_coordinates[:, 1], selected_coordinates[:, 0], color='red', s=20)
        # ax[1].set_title('Thresholded')
        # ax[1].axis('off')



        # for coord in selected_coordinates:
        #     circle = Circle((coord[1], coord[0]), radius, color='red', fill=False)
        #     ax[1].add_patch(circle)