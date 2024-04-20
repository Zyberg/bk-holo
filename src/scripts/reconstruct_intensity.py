import argparse
import numpy as np
import matplotlib.pyplot as plt
# TODO: change function location
from second.autofocus import angular_spectrum

if __name__ == "__main__":
    ## ------------------------------
    # Read input 
    ## ------------------------------
    parser = argparse.ArgumentParser(description="Read saved numpy array from .npy file")
    parser.add_argument("input_path", type=str, help="Path to the saved .npy file (without extension)")
    parser.add_argument("output_path", type=str, help="Path to save the loaded numpy array")
    args = parser.parse_args()

    propagated_field = np.load(args.input_path)

    ## ------------------------------
    # Process 
    ## ------------------------------

    reconstructed_image = np.abs(propagated_field)**2

    ## ------------------------------
    # Persist results
    ## ------------------------------
    np.save(args.output_path, reconstructed_image)

    print("Reconstructed intensity image saved successfully to:", args.output_path)
