import numpy as np

def make_circle_mask(size, mask_size, position):
    center_x, center_y = position
    y, x = np.ogrid[:size, :size]
    mask = np.zeros((size, size), dtype=bool)
    mask = (x - center_x)**2 + (y - center_y)**2 <= (mask_size)**2
    return mask