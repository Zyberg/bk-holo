import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image

def load_hologram(filepath):
    # Load hologram image
    hologram = Image.open(filepath).convert('L')  # Convert to grayscale
    hologram = np.array(hologram)
    return hologram

def reconstruct_object(hologram):
    # Perform Fourier transform
    hologram_ft = fftshift(fft2(hologram))

    # Spatial filtering (e.g., remove bright spots in diagonal)
    # Example: Zero out frequencies in a circular region
    rows, cols = hologram_ft.shape
    center_row, center_col = rows // 2, cols // 2
    radius = 100  # Adjust this parameter based on your hologram
    mask = np.ones((rows, cols))
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - center_col)**2 + (y - center_row)**2 <= radius**2
    mask[mask_area] = 0
    filtered_ft = hologram_ft * mask

    # Perform inverse Fourier transform
    reconstructed_object = ifft2(ifftshift(filtered_ft))

    # Separate reconstructed complex field into amplitude and phase components
    amplitude = np.abs(reconstructed_object)
    phase = np.angle(reconstructed_object)

    return amplitude, phase

# Example usage
if __name__ == "__main__":
    # Load hologram
    # hologram_filepath = "./data/21057145-2024-03-05-175457.tif"
    hologram_filepath = "./data/21057145-2024-03-05-175752.tif"
    hologram = load_hologram(hologram_filepath)

    # Reconstruct object
    reconstructed_amplitude, reconstructed_phase = reconstruct_object(hologram)

    # Display reconstructed amplitude and phase
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(reconstructed_amplitude, cmap='gray')
    plt.title('Reconstructed Amplitude')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_phase, cmap='hsv')
    plt.title('Reconstructed Phase')
    plt.axis('off')

    plt.show()

