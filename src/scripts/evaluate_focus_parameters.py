import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from second.image import Image
from second.autofocus import FocusSolver

if __name__ == "__main__":
    ## ------------------------------
    # Read input 
    ## ------------------------------
    parser = argparse.ArgumentParser(description="Evaluate focus parameters for a given file")
    parser.add_argument("input_path", type=str, help="Path to the reconstructed image file")
    parser.add_argument("propagation_distance", type=float, help="Propagation distance in meters")
    parser.add_argument("output_path", type=str, help="Path to save the loaded numpy array")
    args = parser.parse_args()


    ## ------------------------------
    # Process 
    ## ------------------------------
    reconstructed_image = np.load(args.input_path)

    focus_solver = FocusSolver(reconstructed_image)

    results = focus_solver.evaluate_focus()
    results['distance'] = args.propagation_distance

    ## ------------------------------
    # Persist results
    ## ------------------------------
    with open(args.output_path, "w") as f:
        json.dump(results, f)


    print("Focus sharpness values saved successfully to:", args.output_path)
