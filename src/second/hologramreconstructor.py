import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from matplotlib.widgets import Slider, RadioButtons

MASK_SHAPE_OPTIONS = ('Square', 'Circle')

INITIAL_MASK_POSITION = 1000
INITIAL_MASK_WIDTH = 200
INITIAL_MASK_SHAPE_INDEX = 1

def crop_image_with_mask(image, mask):
    """
    Crop the image to include only the region specified by the mask 

    Parameters:
        image (ndarray): Input image.
        mask (ndarray): Mask specifying the region of interest.

    Returns:
        ndarray: Cropped image containing only the region specified by the mask.
    """
    masked_image = image * mask

    # Find the bounding box of the mask
    rows, cols = np.nonzero(mask)
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    
    # Crop the image according to the bounding box
    cropped_image = masked_image[min_row:max_row+1, min_col:max_col+1]
    
    
    return cropped_image

def crop_and_center_image(image, mask):
    """
    Crop the image to include only the region specified by the mask, and center it on a black image with the same dimensions as the initial image.

    Parameters:
        image (ndarray): Input image.
        mask (ndarray): Mask specifying the region of interest.

    Returns:
        ndarray: Cropped and centered image on a black background.
    """
    # Obtain the cropped image based on the mask
    cropped_image = crop_image_with_mask(image, mask)

    # Create a black image with the same dimensions as the initial image
    black_image = np.zeros_like(image, dtype=image.dtype)

    # Calculate the position to place the cropped image in the black image to center it
    start_row = (black_image.shape[0] - cropped_image.shape[0]) // 2
    start_col = (black_image.shape[1] - cropped_image.shape[1]) // 2

    # Place the cropped image onto the black image at the calculated position
    black_image[start_row:start_row+cropped_image.shape[0], 
                start_col:start_col+cropped_image.shape[1]] = cropped_image

    return black_image

class HologramReconstructor:
    def __init__(self, reference_hologram_image, object_hologram_image):
        self.reference_hologram_image = reference_hologram_image
        self.object_hologram_image = object_hologram_image

        self.reconstructed_phase = None
        self.reconstructed_intensity = None

        self.plot_manager = None
        self.plot_interactive_controls = None

        self.isolated_image_fft = None

        self.mask_position = INITIAL_MASK_POSITION

        maximum_mask_width = self.object_hologram_image.array.shape[0] // 2
        self.mask_width = INITIAL_MASK_WIDTH if INITIAL_MASK_WIDTH < maximum_mask_width else maximum_mask_width
        self.mask_shape = MASK_SHAPE_OPTIONS[INITIAL_MASK_SHAPE_INDEX]

        # Variables for interactive plotting
        self.axis_unmasked_region = None
        self.axis_reconstructed_intensity = None 

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
        self.interference_pattern = object_fft #/ reference_fft # * np.conj(reference_fft)

        # Shift the zero-frequency (DC) component to the center of the Fourier domain
        self.shifted_interference_pattern = fftshift(self.interference_pattern)

        self.mask_size = self.shifted_interference_pattern.shape[0]
        mask = None
        if self.mask_shape == MASK_SHAPE_OPTIONS[0]:
            print('making square')
            mask = make_square_mask(self.mask_size, self.mask_width, self.mask_position)
        elif self.mask_shape == MASK_SHAPE_OPTIONS[1]:
            print('making circle')
            mask = make_circle_mask(self.mask_size, self.mask_width, self.mask_position)
        # mask = make_first_quadrant_mask(self.shifted_interference_pattern.shape[0])
        # mask = make_true_mask(self.interference_pattern.shape[0])

        isolated_image_fft = crop_and_center_image(self.shifted_interference_pattern, mask)

        self.isolated_image_fft = isolated_image_fft

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
        if self.axis_unmasked_region is not None or self.isolated_image_fft is None:
            self.extract_twin_image()

        if self.axis_unmasked_region is None:
            self.axis_unmasked_region = self.__get_ax()

        self.axis_unmasked_region.imshow(np.log(np.abs(self.isolated_image_fft) + 1), cmap='gray')
        self.axis_unmasked_region.set_title('Isolated twin image (FFT)')
        return self.axis_unmasked_region

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
        if self.axis_reconstructed_intensity is not None or self.reconstructed_intensity is None:
            self.reconstruct()

        if self.axis_reconstructed_intensity is None:
            self.axis_reconstructed_intensity = self.__get_ax()

        self.axis_reconstructed_intensity.imshow(self.reconstructed_intensity, cmap='gray')
        self.axis_reconstructed_intensity.set_title('Reconstructed Intensity')
        return self.axis_reconstructed_intensity

    def plot(self):
        self.plot_manager = PlotManager()

        self.plot_object_original()
        # self.plot_interference_pattern()
        # self.plot_shifted_interference_pattern()
        self.plot_unmasked_region()
        # self.plot_reconstructed_phase()
        self.plot_reconstructed_intensity()

        plt.tight_layout()
        plt.show()

    # TODO: this method is a little too coupled
    def callback_update_interactive_plots(self, value):
        # Update state
        self.mask_position = int(self.plot_interactive_controls.slider_mask_position.val)
        self.mask_width = int(self.plot_interactive_controls.slider_mask_width.val)
        self.mask_shape = self.plot_interactive_controls.radio_mask_shape.value_selected

        # Update relevant plots
        self.plot_unmasked_region()
        self.plot_reconstructed_intensity()

        # Redraw
        self.plot_manager.figure.canvas.draw_idle()

    def plot_interactive(self):
        self.plot_manager = PlotManager()

        max_mask_position = self.object_hologram_image.array.shape[0] - self.mask_width
        max_mask_width = self.object_hologram_image.array.shape[0] // 2
        self.plot_interactive_controls = PlotInteractiveControls(max_mask_position, max_mask_width)

        # Set event handlers
        self.plot_interactive_controls.slider_mask_position.on_changed(self.callback_update_interactive_plots)
        self.plot_interactive_controls.slider_mask_width.on_changed(self.callback_update_interactive_plots)
        self.plot_interactive_controls.radio_mask_shape.on_clicked(self.callback_update_interactive_plots)

        # Set window positions
        self.plot_interactive_controls.figure.canvas.manager.window.setGeometry(1200, 0, 800, 400)
        self.plot_manager.figure.canvas.manager.window.setGeometry(0, 0, 1200, 1000)

        # Set window titles
        self.plot_interactive_controls.figure.canvas.manager.window.setWindowTitle("Options")
        self.plot_manager.figure.canvas.manager.window.setWindowTitle("Holography Reconstruction Plots")

        # Plot with initial arguments
        self.plot_unmasked_region()
        self.plot_reconstructed_intensity()
        plt.show()

    
    def __get_ax(self):
        if self.plot_manager is None:
            fig, ax = plt.subplots()
        else:
            ax = self.plot_manager.get_new_axis()

        return ax


# TODO: move to another file
class PlotInteractiveControls:
    def __init__(self, max_mask_position, max_mask_width):
        self.figure = plt.figure(figsize=(6, 3)) 

        self.slider_mask_position = None
        self.slider_mask_width = None 
        self.radio_mask_shape = None

        self.max_mask_position = max_mask_position
        self.max_mask_width = max_mask_width

        self.__initialize_controls()

    def __initialize_controls(self):
        # Define the positions and sizes of sliders and radio buttons
        slider_mask_position_args = [0.2, 0.5, 0.65, 0.1]
        slider_mask_width_args = [0.2, 0.4, 0.65, 0.1]
        radio_buttons_args = [0.2, 0.25, 0.5, 0.15]

        # Define sliders
        self.slider_mask_position = Slider(
            self.__get_control_axis(slider_mask_position_args),
            'Mask Position (xy)',
            0,
            self.max_mask_position,
            valinit=INITIAL_MASK_POSITION
        )

        self.slider_mask_width = Slider(
            self.__get_control_axis(slider_mask_width_args),
            'Mask Width',
            0,
            self.max_mask_width,
            valinit=INITIAL_MASK_WIDTH
        )

        # Define radio buttons
        self.radio_mask_shape = RadioButtons(
            self.__get_control_axis(radio_buttons_args),
            MASK_SHAPE_OPTIONS,
            active = INITIAL_MASK_SHAPE_INDEX
        )

    def __get_control_axis(self, position, facecolor = 'lightgoldenrodyellow'):
        return self.figure.add_axes(position, facecolor=facecolor)

 
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
def make_square_mask(size, mask_size, position):
    mask = np.zeros((size, size))
    mask[position:position+mask_size, position:position+mask_size] = 1
    return mask

def make_circle_mask(size, mask_size, position):
    y, x = np.ogrid[:size, :size]
    mask = np.zeros((size, size), dtype=bool)
    mask = (x - position)**2 + (y - position)**2 <= (mask_size)**2
    return mask

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
