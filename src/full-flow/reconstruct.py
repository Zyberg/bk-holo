import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from mask import make_circle_mask

# def reconstruct_wave_field(hologram, reference_wave, wavelength, dx, dy, z):
#     # Constants
#     k = 2 * np.pi / wavelength  # Wavenumber

#     # Create coordinate grids
#     nx, ny = hologram.shape
#     x = np.linspace(-nx/2, nx/2, nx) * dx
#     y = np.linspace(-ny/2, ny/2, ny) * dy
#     X, Y = np.meshgrid(x, y)

#     # Fresnel propagation phase factor (quadratic phase term)
#     fresnel_phase = np.exp(-1j * k / (2 * z) * (X**2 + Y**2))

#     # Compute the Fourier transform of the hologram multiplied by the reference wave and the Fresnel phase
#     H = fftshift(fft2(hologram * reference_wave * fresnel_phase))

#     # Multiply by another quadratic phase term to complete the Fresnel diffraction integral
#     reconstructed_wave_field = H * np.exp(-1j * k * z) * np.exp(-1j * k / (2 * z) * (X**2 + Y**2))

#     # Inverse Fourier transform to get the reconstructed wave field in the image plane
#     reconstructed_wave_field = ifft2(reconstructed_wave_field)

#     return reconstructed_wave_field

# def reconstruct_wave_field(hologram, reference_wave_conjugate, wavelength, dx, dy, z):
#     # Multiply the hologram by the complex conjugate of the reference wave
#     hologram_with_reference = hologram * reference_wave_conjugate
    
#     # Compute the Fourier transform of the hologram multiplied by the reference wave
#     H_f = fftshift(fft2(hologram_with_reference))
    
#     # Apply the mask to filter out the unwanted components
#     mask = make_circle_mask(hologram.shape[0], 79, [1091,1136])
#     # mask = make_circle_mask(H_f.shape, mask_radius, mask_center)  # You need to define this function
#     H_filtered = H_f * mask
    
#     # Inverse Fourier transform to reconstruct the image
#     reconstructed_image = ifft2(ifftshift(H_filtered))
    
#     # Calculate the intensity of the reconstructed wave
#     intensity_image = np.abs(reconstructed_image)**2
    
#     return intensity_image
