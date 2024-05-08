import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image

# Constants
wavelength = 632.8e-9  # Wavelength of HeNe laser in meters
pixel_size = 1.0e-6  # Pixel size in meters, adjust according to your setup

# Load the hologram using PIL
def load_hologram(file_path):
    img = Image.open(file_path)
    img = img.convert('L')
    return np.array(img, dtype=float)

# Angular spectrum propagation method
def angular_spectrum_propagation(field, wavelength, distance, pixel_size):
    ny, nx = field.shape
    k = 2 * np.pi / wavelength
    dx = dy = pixel_size
    fx = np.fft.fftfreq(nx, dx)
    fy = np.fft.fftfreq(ny, dy)
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(1j * k * np.sqrt(1 - (wavelength * FX)**2 - (wavelength * FY)**2) * distance)
    field_fft = fftshift(fft2(ifftshift(field)))
    field_propagated_fft = field_fft * H
    field_propagated = fftshift(ifft2(ifftshift(field_propagated_fft)))
    return field_propagated

# Load and prepare hologram
hologram = load_hologram('center.tif')
hologram_fft = fftshift(fft2(hologram))
log_magnitude = np.log(np.abs(hologram_fft) + 1)

# Setting up the figure and axes
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(left=0.25, bottom=0.25)

# Initial display of Fourier transform magnitude and reconstructed image
im_fourier = axs[0].imshow(log_magnitude, cmap='gray')
im_reconstruction = axs[1].imshow(np.abs(hologram), cmap='gray')  # Initial placeholder

# Define axes for sliders
axcolor = 'lightgoldenrodyellow'
ax_x = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_y = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_size = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
ax_dist = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)

s_x = Slider(ax_x, 'X', 0, hologram.shape[1]-50, valinit=1000)
s_y = Slider(ax_y, 'Y', 0, hologram.shape[0]-50, valinit=1000)
s_size = Slider(ax_size, 'Size', 10, 1000, valinit=250)
s_dist = Slider(ax_dist, 'Distance (m)', 0.01, 0.1, valinit=0.01)

# Update function for sliders
def update(val):
    x = int(s_x.val)
    y = int(s_y.val)
    size = int(s_size.val)
    distance = s_dist.val
    mask = np.ones_like(hologram, dtype=bool)
    mask[y:y+size, x:x+size] = False
    filtered_fft = hologram_fft * mask
    object_wave = ifft2(ifftshift(filtered_fft))
    propagated_wave = angular_spectrum_propagation(object_wave, wavelength, distance, pixel_size)

    # Clear and update plots
    axs[0].clear()
    axs[0].imshow(log_magnitude, cmap='gray')
    axs[0].autoscale(False)
    axs[0].plot([x, x, x+size, x+size, x], [y, y+size, y+size, y, y], 'r-')
    axs[0].set_title('Fourier Transform with Mask')

    axs[1].clear()
    axs[1].imshow(np.abs(propagated_wave)**2, cmap='gray')
    axs[1].set_title(f'Reconstructed Intensity at {distance*100:.2f} cm')

    fig.canvas.draw_idle()

# Connect update function to sliders
s_x.on_changed(update)
s_y.on_changed(update)
s_size.on_changed(update)
s_dist.on_changed(update)

plt.show()
