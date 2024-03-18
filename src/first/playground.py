import numpy as np
import matplotlib.pyplot as plt

# Simulated hologram parameters
height, width = 256, 256
wavelength = 0.6328e-6  # Wavelength of laser light (in meters)
pixel_size = 6.45e-6    # Pixel size of the sensor (in meters)
distance = 0.1          # Distance from object to sensor (in meters)

# Create object and reference wavefronts
object_wavefront = np.zeros((height, width), dtype=complex)
object_wavefront[height//2 - 20:height//2 + 20, 
                 width//2 - 20:width//2 + 20] = 1  # Creating a simple object (e.g., square aperture)
reference_wavefront = np.exp(1j * np.random.rand(height, width))  # Random phase for reference wavefront

# Simulate recorded hologram (with off-axis geometry)
recorded_hologram = np.abs(object_wavefront + reference_wavefront)**2

# Display initial hologram
plt.figure(figsize=(14, 7))
plt.subplot(2, 3, 1)
plt.imshow(recorded_hologram, cmap='gray')
plt.title('Initial Hologram (Interference Pattern)')
plt.colorbar()

# Calibration: Characterize the response function (simplified example)
response_function = np.ones((height, width))  # Assuming a simple response function for illustration

# Normalize hologram by response function
normalized_hologram = recorded_hologram / response_function

# Reconstruction: Compute complex field
complex_field = np.fft.fftshift(np.fft.fft2(normalized_hologram))

# Display reconstructed amplitude and phase
plt.subplot(2, 3, 2)
plt.imshow(np.abs(complex_field), cmap='gray')
plt.title('Complex Field (Amplitude)')
plt.colorbar()

plt.subplot(2, 3, 3)
plt.imshow(np.angle(complex_field), cmap='hsv')
plt.title('Complex Field (Phase)')
plt.colorbar()

# Numerical propagation to reconstruct object field
propagated_field = np.fft.ifft2(np.fft.fftshift(complex_field) * np.exp(1j * 2 * np.pi * distance / wavelength))

# Display reconstructed amplitude and phase
plt.subplot(2, 3, 4)
plt.imshow(np.abs(propagated_field), cmap='gray')
plt.title('Reconstructed Amplitude')
plt.colorbar()

plt.subplot(2, 3, 5)
plt.imshow(np.angle(propagated_field), cmap='hsv')
plt.title('Reconstructed Phase')
plt.colorbar()

plt.show()

