from PIL import Image
import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift

def load_hologram(file_path):
    hologram = Image.open(file_path)

    # Add padding to make hologram dimensions a power of 2
    hologram = np.array(hologram.convert('L'))
    if hologram.dtype != np.float32:
        hologram = hologram.astype(np.float32)
            
    max_dim = max(hologram.shape)
    
    pad_width = ((0, max_dim - hologram.shape[0]), (0, max_dim - hologram.shape[1]))
    
    hologram = np.pad(hologram, pad_width, mode='constant')

    return hologram


def make_circle_mask(size, mask_size, position):
    center_x, center_y = position
    y, x = np.ogrid[:size, :size]
    mask = np.zeros((size, size), dtype=bool)
    mask = (x - center_x)**2 + (y - center_y)**2 <= (mask_size)**2
    return mask

def fresnel_propagation(wave, z, wavelength, dx):
    k = 2 * np.pi / wavelength
    nx, ny = wave.shape
    Lx, Ly = nx * dx, ny * dx
    x = np.linspace(-Lx/2, Lx/2, nx)
    y = np.linspace(-Ly/2, Ly/2, ny)
    X, Y = np.meshgrid(x, y)
    H = np.exp(1j * k * z) * np.exp(1j * k / (2 * (z + 0.000000001)) * (X**2 + Y**2))
    Uz = ifft2(fft2(wave) * fftshift(H))
    return Uz

def update_focus(val):
    global reconstructed_wavefront, wavelength, pixel_size
    z = val * 1e-4  # Convert trackbar position to meters
    focused_wavefront = fresnel_propagation(reconstructed_wavefront, z, wavelength, pixel_size)
    image_to_show = np.abs(focused_wavefront) / np.max(np.abs(focused_wavefront))

    cv2.imshow('Focused Image', image_to_show)

# Load hologram
hologram = load_hologram("/home/zyberg/bin/bakalauras/src/data/data/21057145-2024-03-26-150331.tif")

# Fourier Transform of the hologram
H_f = fftshift(fft2(hologram))

# Apply a mask to filter

mask = make_circle_mask(hologram.shape[0], 79, [1091,1136])
H_filtered = H_f * mask

# Reconstruct the wavefront
reconstructed_wavefront = ifft2(fftshift(H_filtered))

# Setup display parameters
wavelength = 633e-9  # wavelength of light in meters
pixel_size = 10e-6   # pixel size in meters
cv2.namedWindow('Focused Image', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Propagation Distance', 'Focused Image', 0, 500, update_focus)

# Initialize display
update_focus(0)  # Start with initial focus

# Display loop
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
