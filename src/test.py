import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift
from PIL import Image
import os
import csv

def load_hologram(file_path):
    hologram = Image.open(file_path)
    hologram = np.array(hologram.convert('L'), dtype=np.float32)

    max_dim = max(hologram.shape)
    
    pad_width = ((0, max_dim - hologram.shape[0]), (0, max_dim - hologram.shape[1]))
    
    hologram = np.pad(hologram, pad_width, mode='constant')

    return hologram

def make_circle_mask(size, mask_size, position):
    center_x, center_y = position
    y, x = np.ogrid[:size, :size]
    mask_area = (x - center_x)**2 + (y - center_y)**2 <= mask_size**2
    return mask_area.astype(np.float32)

def fresnel_propagation(wave, z, wavelength, dx):
    k = 2 * np.pi / wavelength
    nx, ny = wave.shape
    Lx, Ly = nx * dx, ny * dx
    x = np.linspace(-Lx/2, Lx/2, nx)
    y = np.linspace(-Ly/2, Ly/2, ny)
    X, Y = np.meshgrid(x, y)
    H = np.exp(1j * k * z) * np.exp(1j * k / (2 * (z + 1e-10)) * (X**2 + Y**2))
    Uz = ifft2(fft2(wave) * fftshift(H))
    return Uz


def fresnel_propagation_with_lens(wave, z, wavelength, dx, f):
    k = 2 * np.pi / wavelength
    nx, ny = wave.shape
    Lx, Ly = nx * dx, ny * dx
    x = np.linspace(-Lx/2, Lx/2, nx)
    y = np.linspace(-Ly/2, Ly/2, ny)
    X, Y = np.meshgrid(x, y)

    # Calculate the Fresnel diffraction pattern
    H_fresnel = np.exp(1j * k * z) * np.exp(1j * k / (2 * z + 0.0000001) * (X**2 + Y**2))

    # Calculate the phase correction for the lens (L function from the textbook)
    L_lens = np.exp(1j * np.pi / (wavelength * f + 0.000001) * (X**2 + Y**2))

    # Apply both the Fresnel diffraction and the lens phase correction
    Uz = ifft2(fft2(wave) * fftshift(H_fresnel * L_lens))

    return Uz

def compute_focus_metric(image):
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var

def save_precomputed_images(wavefront, z_values, wavelength, pixel_size, output_dir):
    metrics_path = os.path.join(output_dir, "focus_metrics.csv")
    with open(metrics_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['z', 'metric', 'filename'])
        for i, z in enumerate(z_values):
            focused_wavefront = fresnel_propagation(wavefront, z, wavelength, pixel_size)
            focused_wavefront = fresnel_propagation_with_lens(wavefront, z, wavelength, pixel_size, 0.1)
            image_abs = np.abs(focused_wavefront)
            image_norm = image_abs / np.max(image_abs)
            focus_metric = compute_focus_metric(image_norm)
            filename = f"focused_image_{i}.npy"
            np.save(os.path.join(output_dir, filename), image_norm)
            writer.writerow([z, focus_metric, filename])

# Load and process hologram
hologram = load_hologram("/home/zyberg/bin/bakalauras/src/data/data/21057145-2024-03-26-150331.tif")

H_f = fftshift(fft2(hologram))
# mask = make_circle_mask(hologram.shape[0], 79, [1091, 1136])
mask = make_circle_mask(hologram.shape[0], 79, [1357, 1312])
H_filtered = H_f * mask
reconstructed_wavefront = ifft2(fftshift(H_filtered))

# Prepare precomputed images and save them
output_dir = "precomputed_focus_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

wavelength = 6.328e-7
pixel_size = 3.45e-6
z_values = np.linspace(0, 0.01, 30)  # Propagation distances in meters
save_precomputed_images(reconstructed_wavefront, z_values, wavelength, pixel_size, output_dir)

print("Precomputation and saving complete.")
