import argparse
import numpy as np
# TODO: change function location
from second.autofocus import angular_spectrum

def phase_factor(x, y, lambda_, d):
    return np.exp(1j * np.pi / (lambda_ * d) * (x**2 + y**2))

def propagate_field(field, wavelength, pixel_size, propagation_distance):
    lambda_ = wavelength
    d = propagation_distance

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
    

    return propagated_field

if __name__ == "__main__":
    ## ------------------------------
    # Read input 
    ## ------------------------------
    parser = argparse.ArgumentParser(description="Read saved numpy array from .npy file")
    parser.add_argument("input_path", type=str, help="Path to the saved .npy file (without extension)")
    parser.add_argument("propagation_distance", type=float, help="Propagation distance in meters")
    parser.add_argument("output_path", type=str, help="Path to save the loaded numpy array")
    args = parser.parse_args()

    reconstructed_field = np.load(args.input_path)
    propagation_distance = args.propagation_distance

    ## ------------------------------
    # Process 
    ## ------------------------------

    # TODO: pass these as config
    wavelength = 6.328e-7
    pixel_size = 3.45e-6

    propagated_field = propagate_field(reconstructed_field, wavelength, pixel_size, propagation_distance)

    ## ------------------------------
    # Persist results
    ## ------------------------------
    np.save(args.output_path, propagated_field)

    print("Loaded numpy array saved successfully to:", args.output_path)
