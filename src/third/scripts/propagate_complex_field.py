import argparse
import numpy as np
# TODO: change function location
from second.autofocus import angular_spectrum

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

    propagated_field = angular_spectrum(reconstructed_field, wavelength, pixel_size, propagation_distance)

    ## ------------------------------
    # Persist results
    ## ------------------------------
    np.save(args.output_path, propagated_field)

    print("Loaded numpy array saved successfully to:", args.output_path)
