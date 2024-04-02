import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

class HologramReconstructor:
    def __init__(self, reference_hologram_image, object_hologram_image):
        self.reference_hologram_image = reference_hologram_image
        self.object_hologram_image = object_hologram_image

        self.reconstructed_phase = None
        self.reconstructed_intensity = None

        self.plot_manager = None

        self.isolated_image_fft = None

        # These variables are set for debug purposes only
        self.interference_pattern = None
        self.shifted_interference_pattern = None


        # TODO: don't use these
        self.reference_hologram = self.reference_hologram_image.array.astype(np.float32)
        self.object_hologram = self.object_hologram_image.array.astype(np.float32)


    def extract_twin_image(self):
        reference_fft = self.reference_hologram_image.get_fft()
        object_fft = self.object_hologram_image.get_fft()

        # Compute the ratio of the Fourier transforms to obtain the interference pattern
        self.interference_pattern = object_fft / reference_fft # * np.conj(reference_fft)

        # Shift the zero-frequency (DC) component to the center of the Fourier domain
        self.shifted_interference_pattern = fftshift(self.interference_pattern)

        mask = make_first_quadrant_mask(self.interference_pattern.shape[0])
        # mask = make_true_mask(self.interference_pattern.shape[0])

        self.isolated_image_fft = self.interference_pattern * mask

    def reconstruct_phase_and_intensity(self):
        if self.isolated_image_fft is None:
            self.extract_twin_image()

        # isolated_image_fft = self.isolated_image_fft[:self.isolated_image_fft.shape[0] // 2, :self.isolated_image_fft.shape[1] // 2]

        reconstructed_image = ifft2(self.isolated_image_fft)
        self.reconstructed_phase = np.angle(reconstructed_image)
        self.reconstructed_intensity = np.abs(reconstructed_image)**2


        # Normalize intensity for visualization
        # self.reconstructed_intensity = (self.reconstructed_intensity - np.min(self.reconstructed_intensity)) / (np.max(self.reconstructed_intensity) - np.min(self.reconstructed_intensity))


        print(np.min(self.reconstructed_intensity), np.max(self.reconstructed_intensity))

    def reconstruct(self):
        self.extract_twin_image()

        self.reconstruct_phase_and_intensity()

    def plot_unmasked_region(self):
        if self.isolated_image_fft is None:
            self.extract_twin_image()

        ax = self.__get_ax()

        ax.imshow(np.log(np.abs(self.isolated_image_fft) + 1), cmap='gray')
        ax.set_title('Isolated twin image (FFT)')
        return ax

    def plot_interference_pattern(self):
        if self.interference_pattern is None:
            self.extract_twin_image()

        ax = self.__get_ax()

        ax.imshow(np.log(np.abs(self.interference_pattern) + 1), cmap='gray')
        ax.set_title('Interference pattern (FFT)')
        return ax

    def plot_shifted_interference_pattern(self):
        if self.shifted_interference_pattern is None:
            self.extract_twin_image()

        ax = self.__get_ax()

        ax.imshow(np.log(np.abs(self.shifted_interference_pattern) + 1), cmap='gray')
        ax.set_title('Shifted interference pattern (FFT)')
        return ax

    def plot_object_original(self):
        ax = self.__get_ax()

        ax.imshow(self.object_hologram_image.padded, cmap='gray')
        ax.set_title('Padded Object Hologram')
        return ax

    def plot_reconstructed_phase(self):
        if self.reconstructed_phase is None:
            self.reconstruct()

        ax = self.__get_ax()

        ax.imshow(self.reconstructed_phase, cmap='gray')
        ax.set_title('Reconstructed Phase')
        return ax

    def plot_reconstructed_intensity(self):
        if self.reconstructed_intensity is None:
            self.reconstruct()

        ax = self.__get_ax()

        ax.imshow(self.reconstructed_intensity, cmap='gray')
        ax.set_title('Reconstructed Intensity')
        return ax

    def plot(self):
        self.plot_manager = PlotManager()

        self.plot_object_original()
        self.plot_interference_pattern()
        # self.plot_shifted_interference_pattern()
        self.plot_unmasked_region()
        self.plot_reconstructed_phase()
        self.plot_reconstructed_intensity()

        plt.tight_layout()
        plt.show()
    
    def __get_ax(self):
        if self.plot_manager is None:
            fig, ax = plt.subplots()
        else:
            ax = self.plot_manager.get_new_axis()

        return ax


 
# TODO: move to another file
class PlotManager:
    def __init__(self):
        self.figure = plt.figure()

        # Initial plot
        self.plot_column = 0


    # TODO: probably not a good idea to expose this
    def get_new_axis(self):
        return self.__get_new_axis()

    def add_subplot_image(self, image, title, cmap='gray'):
        ax = self.__get_new_axis()

        ax.imshow(image, cmap=cmap)

        if title is not None:
            ax.set_title(title)


    def plot(self):
        plt.thight_layout()
        plt.show()

    def __get_new_axis(self):
        self.plot_column += 1

        gs = gridspec.GridSpec(1, self.plot_column)

        # Reposition existing subplots
        for i, ax in enumerate(self.figure.axes):
            ax.set_position(gs[i].get_position(self.figure))
            ax.set_subplotspec(gs[i])

        # Add new subplot
        return self.figure.add_subplot(gs[self.plot_column-1])



# TODO: move to another file
def make_first_quadrant_mask(size):
    mask = np.zeros((size, size))
    mask[:size//2, :size//2] = 1
    return mask

def make_true_mask(size):
    mask = np.ones((size, size))
    return mask



# TODO: most of these functions are faulty and were used only for debugging purposes
# TODO: import these from another file in the future
def reconstruct_phase_off_axis(reference_beam, hologram):
    # Take the Fourier transform of the reference beam and the hologram
    reference_fft = fft2(reference_beam)
    hologram_fft = fft2(hologram)
    
    # Divide the Fourier transform of the hologram by the Fourier transform of the reference beam
    complex_field = hologram_fft / reference_fft
    
    # Take the phase of the complex field
    reconstructed_phase = np.angle(complex_field)
    
    return reconstructed_phase

def reconstruct_phase_TIE(interference_pattern):
    gradient_x, gradient_y = np.gradient(interference_pattern)

    fourier_transform = fft2(interference_pattern)
    
    # Calculate Laplacian of phase
    laplacian_phase = - (gradient_x**2 + gradient_y**2)
    
    return np.angle(ifft2(fourier_transform * laplacian_phase))

def gs_algorithm(reference_hologram, object_hologram, num_iterations=100, beta=0.9):
    # Subtract the reference from the object hologram
    interference_pattern = object_hologram - reference_hologram

    # Initialize object phase randomly
    reconstructed_phase = np.random.rand(*interference_pattern.shape)

    # Iterative phase retrieval using GS algorithm
    for _ in range(num_iterations):
        # Fourier Transform of current object phase
        fourier_transform = fft2(reconstructed_phase)

        # Use magnitude of interference pattern and phase of Fourier transform for update
        updated_magnitude = np.abs(interference_pattern)
        updated_phase = np.angle(fourier_transform)

        # Update object phase in Fourier domain
        updated_fourier_transform = updated_magnitude * np.exp(1j * updated_phase)

        # Inverse Fourier Transform to get updated object phase
        reconstructed_phase = np.real(ifft2(updated_fourier_transform))

        # Apply constraint to keep phase between 0 and 2*pi
        reconstructed_phase = np.mod(reconstructed_phase, 2 * np.pi)

    return reconstructed_phase

def reconstruct_object_image(reconstructed_phase, original_intensity):
    # Compute complex field
    complex_field = np.sqrt(original_intensity) * np.exp(1j * reconstructed_phase)

    object_image = np.abs(ifft2(fft2(complex_field)))

    # Normalize the reconstructed object image
    # object_image /= np.max(object_image)

    # object_image = np.abs(ifft2(complex_field))

    return object_image

def gersbach_phase_retrieval(reference_beam, object_hologram, num_iterations=10):
    """
    Implements the Gersbach phase retrieval algorithm for off-axis holography.

    Args:
        object_hologram (np.ndarray): Object hologram (complex field).
        reference_beam (np.ndarray): Reference beam image (complex field).

    Returns:
        np.ndarray: Reconstructed phase information.
    """
    # Initialize the complex field
    complex_field = object_hologram * np.exp(1j * reference_beam)

    for _ in range(num_iterations):
        # Compute the Fourier transform
        fourier_transform = fftshift(fft2(complex_field))

        # Apply a circular band-pass filter
        filtered_field = apply_band_pass_filter(fourier_transform)

        # Inverse Fourier transform to get the complex field in the object plane
        reconstructed_complex_field = ifft2(ifftshift(filtered_field))

        # Extract the phase information
        reconstructed_phase = np.angle(reconstructed_complex_field)

        # Update the complex field with the new phase
        complex_field = np.abs(object_hologram) * np.exp(1j * reconstructed_phase)

    return reconstructed_phase


def apply_band_pass_filter(fourier_transform, radius=0.1):
    """
    Applies a circular band-pass filter to the Fourier transform.

    Args:
        fourier_transform (np.ndarray): Fourier transform of the complex field.
        radius (float): Radius of the circular filter (0 to 1).

    Returns:
        np.ndarray: Filtered Fourier transform.
    """
    rows, cols = fourier_transform.shape
    center_row, center_col = rows // 2, cols // 2

    # Create a circular mask
    y, x = np.ogrid[:rows, :cols]
    mask = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2) <= radius * min(rows, cols)

    # Apply the mask to the Fourier transform
    filtered_field = fourier_transform * mask.astype(float)

    return filtered_field
