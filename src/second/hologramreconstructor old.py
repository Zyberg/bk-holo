import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

class HologramReconstructor:
    def __init__(self, reference_hologram, object_hologram):
        self.reference_hologram = reference_hologram
        self.object_hologram = object_hologram

    def extract_twin_images(self):
        # Compute the Fourier transforms of the reference and object holograms
        reference_fft = fft2(self.reference_hologram)
        object_fft = fft2(self.object_hologram)

        # Compute the ratio of the Fourier transforms to obtain the interference pattern
        interference_pattern = object_fft / reference_fft

        # Shift the zero-frequency (DC) component to the center of the Fourier domain
        shifted_interference_pattern = fftshift(interference_pattern)

        # Identify the central peak (DC component) in the Fourier domain
        center_x, center_y = np.unravel_index(np.argmax(np.abs(shifted_interference_pattern)),
                                               shifted_interference_pattern.shape)

        # Compute the distance from each pixel to the central peak
        distances = np.sqrt((np.arange(shifted_interference_pattern.shape[0]) - center_x) ** 2 +
                            (np.arange(shifted_interference_pattern.shape[1]) - center_y) ** 2)

        # Identify the twin images based on symmetry properties
        twin_image1 = np.zeros_like(shifted_interference_pattern)
        twin_image2 = np.zeros_like(shifted_interference_pattern)

        for i in range(shifted_interference_pattern.shape[0]):
            for j in range(shifted_interference_pattern.shape[1]):
                if distances[i] == distances[center_x]:
                    twin_image1[i, j] = shifted_interference_pattern[i, j]
                    twin_image2[-i, -j] = shifted_interference_pattern[-i, -j]

        return twin_image1, twin_image2

    def reconstruct_phase_and_intensity(self, twin_image):
        # Shift the twin image back to the original position
        shifted_twin_image = fftshift(twin_image)

        # Create a mask selecting the first quadrant of the shifted twin image
        mask = np.zeros_like(shifted_twin_image)
        mask[:mask.shape[0] // 2, :mask.shape[1] // 2] = 1

        # Apply the mask to the shifted twin image
        masked_twin_image = shifted_twin_image * mask

        # Take the inverse Fourier transform of the masked twin image to obtain the reconstructed phase and intensity
        reconstructed_phase = np.angle(ifft2(masked_twin_image))
        reconstructed_intensity = np.abs(ifft2(fftshift(self.reference_hologram) * masked_twin_image))

        return reconstructed_phase, reconstructed_intensity

    def plot(self):
        # Extract twin images from the holograms
        twin_image1, twin_image2 = self.extract_twin_images()

        # Choose one twin image (e.g., the one with the highest magnitude)
        chosen_twin_image = twin_image1 if np.sum(np.abs(twin_image1)) > np.sum(np.abs(twin_image2)) else twin_image2

        # Reconstruct phase and intensity from the chosen twin image
        reconstructed_phase, reconstructed_intensity = self.reconstruct_phase_and_intensity(chosen_twin_image)

        # Plot the reconstructed phase and intensity images
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(reconstructed_phase, cmap='gray')
        axes[0].set_title('Reconstructed Phase (First Quadrant)')
        axes[0].set_xlabel('Columns')
        axes[0].set_ylabel('Rows')
        axes[0].set_aspect('equal')
        axes[0].grid(False)
        # axes[0].colorbar = plt.colorbar(orientation='horizontal', ax=axes[0])
        axes[1].imshow(reconstructed_intensity, cmap='gray')
        axes[1].set_title('Reconstructed Intensity (First Quadrant)')
        axes[1].set_xlabel('Columns')
        axes[1].set_ylabel('Rows')
        axes[1].set_aspect('equal')
        axes[1].grid(False)
        # axes[1].colorbar = plt.colorbar(orientation='horizontal', ax=axes[1])
        plt.tight_layout()
        plt.show()