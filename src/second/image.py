from PIL import Image as ImagePIL
import numpy as np
import matplotlib.pyplot as plt
import os

class Image:
    def __init__(self, path):
        script_dir = os.path.dirname(__file__)
        self.file_name = None
        self.original = None

        self.gray = None
        self.array = None

        self.fft = None
        self.fft_shifted = None

        # Load image
        self._load_image(path)

    def _load_image(self, file_path):
        if file_path.endswith('.tif'):
            self.original = ImagePIL.open(file_path)
            self.file_name = os.path.basename(file_path)
            self.gray = self.original.convert('L')
            self.array = np.array(self.gray)

    def get_fft(self):
        if self.fft is None:
            self.fft = np.fft.fft2(self.array)

        return self.fft


    def get_shifted(self):
        if self.fft_shifted is None:
            self.fft_shifted = np.fft.fftshift(self.get_fft())

        return self.fft_shifted