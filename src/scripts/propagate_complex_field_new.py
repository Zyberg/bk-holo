import argparse
import numpy as np
# TODO: change function location
from second.autofocus import angular_spectrum
from numpy.fft import fft2, fftshift, ifftshift, ifft2


def propagate_field(field, wavelength, pixel_size, propagation_distance):
    upsample_scale = 1
    n = upsample_scale * field.shape[0]
    grid_size = pixel_size * n

    # Inverse space
    space_magnitude = (n-1)*(1/grid_size)
    fx = np.linspace(-space_magnitude / 2, space_magnitude / 2, n)
    fy = np.linspace(-space_magnitude / 2, space_magnitude / 2, n)
    Fx, Fy = np.meshgrid(fx, fy)

    H = np.exp(1j * np.pi * wavelength * propagation_distance * (Fx**2 + Fy**2))

    E0fft = fftshift(field)

    # Multiply spectrum with fresnel phase-factor
    G = E0fft * H

    constant_phase_multiplier = np.exp(1j * 2 * np.pi * propagation_distance / wavelength) / (1j * wavelength * propagation_distance)

    Ef = constant_phase_multiplier * ifft2(ifftshift(G))

    return Ef

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
    wavelength = 540
    pixel_size = 3.75e-6

    propagated_field = propagate_field(reconstructed_field, wavelength, pixel_size, propagation_distance)

    ## ------------------------------
    # Persist results
    ## ------------------------------
    np.save(args.output_path, propagated_field)

    print("Loaded numpy array saved successfully to:", args.output_path)
