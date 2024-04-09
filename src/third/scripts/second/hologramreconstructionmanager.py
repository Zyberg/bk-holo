import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

MASK_SHAPE_OPTIONS = ('Square', 'Circle')

class HologramReconstructionManager:
    def __init__(self, hologram_image, mask_position, mask_width, mask_shape_index):
        self.hologram_image = hologram_image

        self.mask_size = self.hologram_image.array.shape[0]

        self.mask_position = mask_position
        self.mask_width = mask_width
        self.mask_shape_index = mask_shape_index
        self.mask_shape = MASK_SHAPE_OPTIONS[mask_shape_index]

    def reconstruct(self):
        object_fft = self.hologram_image.get_fft()

        self.shifted_interference_pattern = fftshift(object_fft)

        mask = self.__make_mask()
        
        reconstructed_field_fourier_space = crop_and_center_image(self.shifted_interference_pattern, mask)

        return reconstructed_field_fourier_space, mask

    def __make_mask(self):
        mask = None

        if self.mask_shape == MASK_SHAPE_OPTIONS[0]:
            mask = make_square_mask(self.mask_size, self.mask_width, self.mask_position)
        elif self.mask_shape == MASK_SHAPE_OPTIONS[1]:
            mask = make_circle_mask(self.mask_size, self.mask_width, self.mask_position)

        return mask



def make_square_mask(size, mask_size, position):
    center_x, center_y = position
    mask = np.zeros((size, size))
    mask[center_y:center_y+mask_size, center_x:center_x+mask_size] = 1
    return mask

def make_circle_mask(size, mask_size, position):
    center_x, center_y = position
    y, x = np.ogrid[:size, :size]
    mask = np.zeros((size, size), dtype=bool)
    mask = (x - center_x)**2 + (y - center_y)**2 <= (mask_size)**2
    return mask

def make_first_quadrant_mask(size):
    mask = np.zeros((size, size))
    mask[:size//2, :size//2] = 1
    return mask

def make_true_mask(size):
    mask = np.ones((size, size))
    return mask


def crop_image_with_mask(image, mask):
    """
    Crop the image to include only the region specified by the mask 

    Parameters:
        image (ndarray): Input image.
        mask (ndarray): Mask specifying the region of interest.

    Returns:
        ndarray: Cropped image containing only the region specified by the mask.
    """
    masked_image = image * mask

    # Find the bounding box of the mask
    rows, cols = np.nonzero(mask)
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    
    # Crop the image according to the bounding box
    cropped_image = masked_image[min_row:max_row+1, min_col:max_col+1]
    
    
    return cropped_image

def crop_and_center_image(image, mask):
    """
    Crop the image to include only the region specified by the mask, and center it on a black image with the same dimensions as the initial image.

    Parameters:
        image (ndarray): Input image.
        mask (ndarray): Mask specifying the region of interest.

    Returns:
        ndarray: Cropped and centered image on a black background.
    """
    # Obtain the cropped image based on the mask
    cropped_image = crop_image_with_mask(image, mask)

    # Create a black image with the same dimensions as the initial image
    black_image = np.zeros_like(image, dtype=image.dtype)

    # Calculate the position to place the cropped image in the black image to center it
    start_row = (black_image.shape[0] - cropped_image.shape[0]) // 2
    start_col = (black_image.shape[1] - cropped_image.shape[1]) // 2

    # Place the cropped image onto the black image at the calculated position
    black_image[start_row:start_row+cropped_image.shape[0], 
                start_col:start_col+cropped_image.shape[1]] = cropped_image

    return black_image
