from PIL import Image as ImagePIL
import numpy as np
import matplotlib.pyplot as plt
import os
from pyfftw.interfaces.numpy_fft import fft2, ifft2, fftshift, ifftshift

class Image:
    def __init__(self, path):
        script_dir = os.path.dirname(__file__)
        self.file_name = None
        self.original = None
        self.padded = None

        self.gray = None
        self.array = None

        self.fft = None
        self.fft_shifted = None

        # Load image
        self._load_image(path)

    def _load_image(self, file_path):
        if file_path.endswith('.tif'):
            self.original = ImagePIL.open(file_path)
            # Always pad an image to square before doing anything
            self.get_padded_to_square()
            self.file_name = os.path.basename(file_path)
            self.gray = self.original.convert('L')
            self.array = np.array(self.padded)

    def get_fft(self):
        if self.fft is None:
            self.fft = fft2(self.array)

        return self.fft


    def get_shifted(self):
        if self.fft_shifted is None:
            self.fft_shifted = fftshift(self.get_fft())

        return self.fft_shifted

    def get_padded_to_square(self):
        if self.original is None:
            raise Exception("Original image is not loaded. There is nothing to pad.")

        if self.padded is None:
            image = np.array(self.original.convert('L'))
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            
            # Find the maximum dimension
            max_dim = max(image.shape)
            
            pad_width = ((0, max_dim - image.shape[0]), (0, max_dim - image.shape[1]))
            
            self.padded = np.pad(image, pad_width, mode='constant')
        
        return self.padded