import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
def angular_spectrum(field, wavelength, pixel_size, z):
    """
    Perform Angular Spectrum Propagation.
    
    Parameters:
        field (2D complex array): Input complex field.
        z (float): Propagation distance (in meters).
        wavelength (float): Wavelength of light (in meters).
        pixel_size (float): Pixel size of the holographic sensor (in meters).
        
    Returns:
        propagated_field (2D complex array): Propagated complex field.
    """
    # Perform 2D Fourier Transform
    freq_field = fftshift(field)
    
    # Calculate spatial frequencies
    u, v = np.meshgrid(np.fft.fftfreq(field.shape[1], pixel_size), np.fft.fftfreq(field.shape[0], pixel_size))
    
    # Calculate transfer function for propagation
    transfer_function = np.exp(-1j * np.pi * wavelength * z * (u**2 + v**2))
    
    # Apply transfer function
    propagated_freq_field = freq_field * transfer_function
    
    # Perform inverse Fourier Transform
    propagated_field = ifft2(ifftshift(propagated_freq_field))
    
    return propagated_field

# import numpy as np
# from numpy.fft import fft2, ifft2, fftshift, ifftshift

# def propagate(field, wavelength, pixel_size, z):
#     # Perform 2D Fourier Transform and shift zero-frequency component to the center
#     freq_field = fftshift(fft2(field))
    
#     # Calculate spatial frequencies with proper scaling by pixel_size
#     ny, nx = field.shape
#     u = np.fft.fftfreq(nx, pixel_size)
#     v = np.fft.fftfreq(ny, pixel_size)
#     u, v = np.meshgrid(u, v)  # Create 2D grid of frequencies
    
#     # Calculate transfer function for propagation using the Fresnel approximation
#     transfer_function = np.exp(-1j * np.pi * wavelength * z * (u**2 + v**2))
    
#     # Apply the transfer function to the frequency-domain field
#     propagated_freq_field = freq_field * transfer_function
    
#     # Inverse Fourier Transform to return to spatial domain
#     propagated_field = ifft2(ifftshift(propagated_freq_field))
    
#     return propagated_field



def propagate(field, wavelength, pixel_size, z):
    # Perform 2D Fourier Transform
    freq_field = fftshift(fft2(field))
    
    # Calculate spatial frequencies
    u, v = np.meshgrid(np.fft.fftfreq(field.shape[1], pixel_size), np.fft.fftfreq(field.shape[0], pixel_size))
    
    # Calculate transfer function for propagation
    transfer_function = np.exp(-1j * np.pi * wavelength * z * (u**2 + v**2))
    
    # Apply transfer function
    propagated_freq_field = freq_field * transfer_function
    
    # Perform inverse Fourier Transform
    propagated_field = ifft2(ifftshift(propagated_freq_field))
    
    return propagated_field

def angular_spectrum_old(complex_field_fft, wavelength, pixel_size, propagation_distance):
    """
    Perform numerical propagation using the angular spectrum method.

    Parameters:
    - complex_field_fft: 2D array containing the complex field distribution at the hologram plane (FFT)
    - wavelength: Wavelength of the light (in meters)
    - pixel_size: Size of a pixel on the sensor (in meters)
    - propagation_distance: Distance to propagate the field (in meters)

    Returns:
    - propagated_field: 2D array containing the complex field distribution at the desired distance
    """

    # Compute the size of the input field
    rows, cols = complex_field_fft.shape
    nyquist_sampling = 1 / (2 * pixel_size)

    # Compute the frequencies in the x and y directions
    u = np.fft.fftfreq(cols, pixel_size)
    v = np.fft.fftfreq(rows, pixel_size)

    # Compute the spatial frequencies grid
    u_grid, v_grid = np.meshgrid(u, v, indexing='xy')

    # Compute the wave number
    k = 2 * np.pi / wavelength

    # Compute the transfer function
    transfer_function = np.exp(1j * k * np.sqrt(1 - (wavelength * u_grid) ** 2 - (wavelength * v_grid) ** 2) * propagation_distance)

    # Apply transfer function in the frequency domain
    complex_field_propagated_fft = complex_field_fft * transfer_function

    # Perform inverse Fourier transform to obtain the propagated field
    propagated_field = ifft2(complex_field_propagated_fft)

    return propagated_field


class FocusSolver:
    def __init__(self, image):
        self.image = image

    def evaluate_focus(self):
        return {
            'tenengrad': self.tenengrad(self.image),
            'normalized_gradient_variance': self.normalized_gradient_variance(self.image),
            # Add other sharpness metrics here
        }

    def tenengrad(self, image):
        """
        Compute the Tenengrad sharpness metric.
        
        Returns:
        - tenengrad: Tenengrad sharpness metric
        """
        sobel_x = ndimage.sobel(image, axis=0, mode='reflect')
        sobel_y = ndimage.sobel(image, axis=1, mode='reflect')
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        tenengrad = np.mean(gradient_magnitude)
        return tenengrad

    def normalized_gradient_variance(self, image):
        """
        Compute the Normalized Gradient Variance (NGV) metric.
        
        Returns:
        - ngv: NGV metric
        """
        sobel_x = ndimage.sobel(image, axis=0, mode='reflect')
        sobel_y = ndimage.sobel(image, axis=1, mode='reflect')
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        ngv = np.var(gradient_magnitude) / np.mean(gradient_magnitude)
        return ngv


class AutofocusManager:
    def __init__(self, complex_field, wavelength, pixel_size):
        self.complex_field = complex_field
        self.wavelength = wavelength
        self.pixel_size = pixel_size

    def autofocus(self):
        # Define a range of propagation distances to consider
        # min_distance = 0.01  # Minimum propagation distance in meters
        # max_distance = 0.1   # Maximum propagation distance in meters
        # step_size = 0.001    # Step size for iterating over distances
        min_distance = 0.129  # Minimum propagation distance in meters
        max_distance = 0.3   # Maximum propagation distance in meters
        step_size = 0.001    # Step size for iterating over distances

        # Compute sharpness metrics for each propagation distance
        distances = np.arange(min_distance, max_distance + step_size, step_size)
        sharpness_metrics = []
        for distance in distances:
            print(f'Performing analysis on distance:\t{distance}')
            propagated_field = angular_spectrum(self.complex_field, self.wavelength, self.pixel_size, distance)
            
            reconstructed_image = np.abs(propagated_field)**2

            # TODO: move this out to some other place
            fig, ax = plt.subplots()

            ax.imshow(reconstructed_image, cmap='gray')
            fig.savefig(f'/home/zyberg/bin/bakalauras/src/second/autofocus/left/{round(distance, 3)}.png', dpi=300)
            
            plt.close(fig)

            sharpness_values = [
                self.tenengrad(reconstructed_image),
                self.normalized_gradient_variance(reconstructed_image),
                # Add other sharpness metrics here
            ]
            sharpness_metrics.append(sharpness_values)

            print(f'Obtained sharpness:\t{sharpness_values[0]}\t{sharpness_values[1]}')

        # Normalize sharpness metrics (optional)
        sharpness_metrics = np.array(sharpness_metrics)
        sharpness_metrics_normalized = (sharpness_metrics - sharpness_metrics.min(axis=0)) / (sharpness_metrics.max(axis=0) - sharpness_metrics.min(axis=0))

        # Combine metrics (e.g., simple weighted sum)
        weights = np.array([0.5, 0.5])  # Adjust weights based on importance
        combined_metric = np.dot(sharpness_metrics_normalized, weights)

        # Select optimal distance based on combined metric
        optimal_distance_index = np.argmax(combined_metric)
        optimal_distance = distances[optimal_distance_index]

        print('All sharpness metrics', sharpness_metrics_normalized)

        print("Optimal propagation distance:", optimal_distance)

        return optimal_distance

    def laplacian_of_gaussian(self, image, sigma=1):
        """
        Compute the Laplacian of Gaussian (LoG) filter response.
        
        Parameters:
        - sigma: Standard deviation of the Gaussian kernel
        
        Returns:
        - LoG: LoG filter response
        """
        blurred = ndimage.gaussian_filter(image, sigma)
        LoG = filters.laplace(blurred)
        return LoG

    def tenengrad(self, image):
        """
        Compute the Tenengrad sharpness metric.
        
        Returns:
        - tenengrad: Tenengrad sharpness metric
        """
        sobel_x = ndimage.sobel(image, axis=0, mode='reflect')
        sobel_y = ndimage.sobel(image, axis=1, mode='reflect')
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        tenengrad = np.mean(gradient_magnitude)
        return tenengrad

    def normalized_gradient_variance(self, image):
        """
        Compute the Normalized Gradient Variance (NGV) metric.
        
        Returns:
        - ngv: NGV metric
        """
        sobel_x = ndimage.sobel(image, axis=0, mode='reflect')
        sobel_y = ndimage.sobel(image, axis=1, mode='reflect')
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        ngv = np.var(gradient_magnitude) / np.mean(gradient_magnitude)
        return ngv

