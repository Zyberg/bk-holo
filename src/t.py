import cv2
import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from PIL import Image

def load_hologram():
    hologram = Image.open("/home/zyberg/bin/bakalauras/src/data/data/21057145-2024-03-26-150331.tif")
    hologram = np.array(hologram.convert('L'), dtype=np.float32)

    max_dim = max(hologram.shape)
    
    pad_width = ((0, max_dim - hologram.shape[0]), (0, max_dim - hologram.shape[1]))
    
    hologram = np.pad(hologram, pad_width, mode='constant')

    return hologram

# Define the function for the angular spectrum method with a lens
def angular_spectrum_method_with_lens(hologram, z, wavelength, dx, f):
    ny, nx = hologram.shape
    ly, lx = ny * dx, nx * dx
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    FX, FY = np.meshgrid(fx, fy)

    # Calculate the spatial frequency domain filter for the propagation distance z
    H = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    
    # Calculate the spatial frequency domain filter for the lens with focal length f
    L = np.exp(-1j * np.pi * wavelength * (1/f) * (FX**2 + FY**2))

    # Apply both filters to the angular spectrum of the hologram
    A = fft2(hologram)
    Uz = ifft2(A * H * L)
    
    # Calculate the intensity of the reconstructed wave at distance z
    intensity_image = np.abs(Uz)**2
    return intensity_image

def phase_factor(x, y, lambda_, d):
    return np.exp(1j * np.pi / (lambda_ * d) * (x**2 + y**2))

def propagate_field(hologram, propagation_distance, wavelength, pixel_size):
    lambda_ = wavelength
    d = propagation_distance
    field = fft2(hologram)

    # Apply the phase factor to the product
    Nx, Ny = field.shape
    x = np.linspace(-Nx/2, Nx/2, Nx)
    y = np.linspace(-Ny/2, Ny/2, Ny)
    X, Y = np.meshgrid(x, y)
    field *= phase_factor(X, Y, lambda_, d)
    
    # Compute the Fourier transform
    fft_result = np.fft.fft2(field)
    fft_result = np.fft.fftshift(fft_result)  # Shift the zero frequency component to the center
    
    # Multiply by the quadratic phase factor for reconstruction
    xi, eta = X, Y  # Assuming xi and eta are in the same range as x and y
    propagated_field = (1j / (lambda_ * d)) * np.exp(-1j * 2 * np.pi / lambda_ * d) * np.exp(1j * np.pi / (lambda_ * d) * (xi**2 + eta**2)) * fft_result
    
    intensity_image = np.abs(propagated_field)**2
    return intensity_image

# Load your hologram here
hologram = load_hologram()
wavelength = 6.328e-7 # Wavelength of light (m)
dx =  3.45e-6  # Sampling interval (m)


# Initial parameters for z and f
z_initial = 0.05  # Propagation distance (m)
f_initial = 0.1  # Focal length (m)

# Define the update function for the trackbars
def update_image(*args):
    z = cv2.getTrackbarPos('Propagation Distance', 'Reconstructed Image') * 1e-2  # Scale factor for z
    intensity_image = propagate_field(hologram, z, wavelength, dx)

    intensity_image_normalized = intensity_image / intensity_image.max()

    # Get the size of the screen
    screen_res = cv2.getWindowImageRect('Reconstructed Image')
    screen_height, screen_width = screen_res[3], screen_res[2]
    
    # Calculate the scaling factor
    height, width = intensity_image_normalized.shape
    scale_w = screen_width / width
    scale_h = screen_height / height
    scale = min(scale_w, scale_h)
    
    # Resize the image to fit the screen
    display_image = cv2.resize(intensity_image_normalized, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    
    # Show the resized image
    cv2.imshow('Reconstructed Image', display_image)

# Create a window and two trackbars for z and f
cv2.namedWindow('Reconstructed Image')
cv2.createTrackbar('Propagation Distance', 'Reconstructed Image', int(z_initial * 100), 1000, update_image)

# Initial update and display
update_image()

# Display loop
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cv2.destroyAllWindows()
