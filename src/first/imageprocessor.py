from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

class ImageProcessor:
    def __init__(self, data_directory):
        script_dir = os.path.dirname(__file__)
        self.directory = os.path.join(script_dir, data_directory)
        self.images = []
        self.file_names = []
        self.load_images()

    def load_images(self):
        for filename in os.listdir(self.directory):
            if filename.endswith('.tif'):
                file_path = os.path.join(self.directory, filename)
                image = Image.open(file_path)
                self.images.append(image)
                self.file_names.append(filename)

    def display_image_options(self):
        print("Select an image:")
        for i, filename in enumerate(self.file_names):
            print(f"{i+1}. {filename}")
        print(f"{len(self.file_names) + 1}. Display FFT of all images")

    def perform_fft(self, selection=None):
        if selection is None:
            print("FFT of all photos")
            fig, axes = plt.subplots(3, len(self.images), figsize=(9, 3 * len(self.images)))
            for i, image in enumerate(self.images):
                self._display_image_with_fft(image, axes[:, i], self.file_names[i])
        else:
            selected_image = self.images[selection - 1]
            print(f"FFT of {self.file_names[selection - 1]}")

            fig, axes = plt.subplots(3, 1, figsize=(9, 3))
            self._display_image_with_fft(selected_image, axes, self.file_names[selection - 1])

        plt.tight_layout()
        plt.show()

    def _display_image_with_fft(self, image, axes, title):
        self._display_reconstructed(image, axes[0], title)
        self._display_fft(image, axes[1], title)
        self._display_original(image, axes[2], title)

    def _display_reconstructed(self, image, ax, title):
        image_gray = image.convert('L')
        image_array = np.array(image_gray)
        # Pad the image with zeros to make dimensions compatible
        padded_image_array = np.pad(image_array, ((0, image_array.shape[0]), (0, image_array.shape[1])))

        fft_result = np.fft.fft2(padded_image_array)
        # Create a mesh grid for Fourier domain manipulations
        rows, cols = fft_result.shape
        row_mid, col_mid = rows // 2, cols // 2
        row_freq = np.fft.fftfreq(rows)
        col_freq = np.fft.fftfreq(cols)
        row_mesh, col_mesh = np.meshgrid(row_freq, col_freq)

        # Define the reference wave phase 
        reference_phase = np.exp(1j * (2 * np.pi * (row_mesh + col_mesh )))
        # reference_phase = np.exp(1j * (2 * np.pi * (row_mesh * row_mid + col_mesh * col_mid)))

        # Subtract the reference wave in Fourier space 
        fft_subtracted = fft_result # * reference_phase 


        # fft_shifted = np.fft.fftshift(fft_result)
        # reconstructed_image = np.fft.ifft2(np.fft.ifftshift(fft_shifted)).real

        reconstructed_wavefront = np.fft.ifft2(fft_subtracted)
        reconstructed_intensity = np.abs(reconstructed_wavefront)

        ax.imshow(reconstructed_intensity)
        ax.axis('off')
        ax.set_title(f"Reconstructed {title}")

    def _display_fft(self, image, ax, title):
        image_gray = image.convert('L')
        image_array = np.array(image_gray)
        fft_result = np.fft.fft2(image_array)
        fft_shifted = np.fft.fftshift(fft_result)
        magnitude_spectrum = np.abs(fft_shifted)
        ax.imshow(np.log(magnitude_spectrum + 1), cmap='gray')
        ax.set_title(f"FFT of {title}")
        ax.axis('off')

    def _display_original(self, image, ax, title):
        ax.imshow(image, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
