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

    reconstructed_image = np.load(args.input_path)

    ## ------------------------------
    # Persist results
    ## ------------------------------
    fig, ax = plt.subplots()

    ax.imshow(reconstructed_image, cmap='gray')
    fig.savefig(args.output_path, dpi=300)
    
    plt.close(fig)


    print("Reconstructed intensity image saved successfully to:", args.output_path)
