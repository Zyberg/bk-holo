from PIL import Image
import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift

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

# Define the function for the angular spectrum method with a lens
def angular_spectrum_method_with_lens_old(hologram, z, wavelength, dx, f):
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

def angular_spectrum_method_with_lens_old2(hologram, z, wavelength, dx, f):
    ny, nx = hologram.shape
    ly, lx = ny * dx, nx * dx
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    FX, FY = np.meshgrid(fftshift(fx), fftshift(fy))

    # Calculate the angular spectrum of the hologram
    A = fftshift(fft2(hologram))
    
    # Calculate the spatial frequency domain filter for the propagation distance z
    H = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    
    # Calculate the spatial frequency domain filter for the lens with focal length f
    L = np.exp(-1j * np.pi * wavelength * (1/f) * (FX**2 + FY**2))

    # Apply both filters to the angular spectrum of the hologram
    Uz = ifft2(fftshift(A * H * L))
    
    # Calculate the intensity of the reconstructed wave at distance z
    intensity_image = np.abs(Uz)**2
    return intensity_image


def angular_spectrum_method_with_lens_without_correction(hologram, z, wavelength, dx, f):
    ny, nx = hologram.shape
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    FX, FY = fftshift(FX), fftshift(FY)
    A = fft2(hologram)

    # Calculate the propagation transfer function
    H = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    
    # Calculate the lens phase correction factor
    L = np.exp(-1j * np.pi * wavelength * (1/f) * (FX**2 + FY**2))

    # Apply both the propagation transfer function and the lens phase correction
    # Note: fftshift is used before applying the filter, and ifftshift before ifft2
    Uz = ifft2(ifftshift(fftshift(A) * H * L))

    # Calculate the intensity of the reconstructed wave at distance z
    intensity_image = np.abs(Uz)**2
    return intensity_image

# TODO: Review all the formulas in Schnar's textbook to make sure that they are correct
def angular_spectrum_method_with_lens_oldish(hologram, z, wavelength, dx, f):
    # Compute spatial frequencies
    ny, nx = hologram.shape
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    FX, FY = np.meshgrid(fftshift(fx), fftshift(fy))

    # Compute the angular spectrum of the hologram
    A = fftshift(fft2(hologram))

    # Lens phase correction
    L = np.exp(-1j * (np.pi/(f * wavelength)) * (FX**2 + FY**2))

    # Propagation phase correction
    H = np.exp(-1j * np.pi /( wavelength * z) * (FX**2 + FY**2))

    # Apply the phase corrections
    Uz = ifft2(ifftshift(A * L * H))

    # Phase correction factor P to correct for aberrations
    # TODO: probably wrong coords?
    P = np.exp(1j * np.pi / (wavelength * f) * (FX**2 + FY**2))

    # Apply the correction factor P
    corrected_Uz = Uz * P

    # Calculate the intensity of the reconstructed wave
    intensity_image = np.abs(corrected_Uz)**2
    return intensity_image


def phase_factor(x, y, lambda_, d):
    return np.exp(1j * np.pi / (lambda_ * d) * (x**2 + y**2))

def propagate_field(hologram, propagation_distance, wavelength, pixel_size):
    lambda_ = wavelength
    d = propagation_distance
    field = hologram

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

# ---------------------------------------
# REAL ANALYSIS
# ---------------------------------------

# Load hologram
hologram = load_hologram("/home/zyberg/bin/bakalauras/src/data/data/21057145-2024-03-26-150331.tif")
wavelength = 6.328e-7 # Wavelength of light (m)
pixel_size = dx =  3.45e-6  # Sampling interval (m)

# Fourier Transform of the hologram
H_f = fftshift(fft2(hologram))

# Apply a mask to filter

mask = make_circle_mask(hologram.shape[0], 79, [1091,1136])
H_filtered = H_f * mask

# Reconstruct the wavefront
reconstructed_wavefront = ifft2(ifftshift(H_filtered))

# Compute the spatial frequencies only once
ny, nx = H_filtered.shape
fx = np.fft.fftfreq(nx, d=dx)
fy = np.fft.fftfreq(ny, d=dx)
FX, FY = np.meshgrid(fftshift(fx), fftshift(fy))

# Compute the angular spectrum of the hologram only once
A = H_filtered


# Precompute stuff
precomputed_lens_phase_correction = np.exp(-1j * (np.pi/(wavelength)) * (FX**2 + FY**2)) 
precomputed_propagation_phase_correction = np.exp(-1j * (np.pi /(wavelength)) * (FX**2 + FY**2))
precomputed_abberation_phase_correction = np.exp(1j * (np.pi / (wavelength)) * (FX**2 + FY**2))

def lens_phase_correction(f, wavelength):
    global FX, FY

    # return np.exp(-1j * (np.pi/(wavelength * f + 0.000001)) * (FX**2 + FY**2))
    return precomputed_lens_phase_correction * np.exp(1 / (f + 0.000001))

def propagation_phase_correction(z, wavelength):
    global FX, FY

    # return np.exp(-1j * np.pi /(wavelength * z + 0.000001) * (FX**2 + FY**2))
    return precomputed_propagation_phase_correction * np.exp(1 / (z + 0.000001))
    
# TODO: check

# def propagation_phase_correction(z):
#     # Correct for the propagation phase
#     return np.exp(-1j * np.pi * (wavelength * z) * (FX**2 + FY**2) / dx**2)


def aberration_phase_correction(f, wavelength):
    global FX, FY

    # return np.exp(1j * np.pi / (wavelength * f + 0.000001) * (FX**2 + FY**2))
    return precomputed_abberation_phase_correction * np.exp(1 / (f + 0.000001))


# Angular spectrum method with numerical lens correction
def angular_spectrum_method_with_lens(z, f):
    global FX, FY, A, wavelength, dx
    
    # Apply the propagation phase correction
    H = propagation_phase_correction(z, wavelength)
    
    # Apply the lens phase correction
    L = lens_phase_correction(f, wavelength)
    
    # Combine the corrections and compute the inverse FFT
    Uz = ifft2(ifftshift(A * H * L))

    # Apply the aberration phase correction
    P = aberration_phase_correction(f, wavelength)
    corrected_Uz = Uz * P
    
    # Calculate the intensity of the reconstructed wave
    intensity_image = np.abs(corrected_Uz)**2
    
    return intensity_image


def update_image(val):
    global wavelength, pixel_size
    z = cv2.getTrackbarPos('Propagation Distance', 'Reconstructed Image') * 1e-2  # Scale factor for z
    focused_wavefront = propagate_field(A, z, wavelength, pixel_size)
    image_to_show = np.abs(focused_wavefront) / np.max(np.abs(focused_wavefront))

    cv2.imshow('Reconstructed Image', image_to_show)



z_initial = 0.001  # Propagation distance (m)
z_max = 0.23

# Setup display parameters
cv2.namedWindow('Reconstructed Image', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Propagation Distance', 'Reconstructed Image', int(z_initial * 100), int(z_max * 100), update_image)

# Initialize display
update_image(0)  # Start with initial focus

# Display loop
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
