import argparse
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from second.image import Image
from second.autofocus import FocusSolver

def combine_all_focus_parameters_through_distances(directory_path):
    results = {}

    print("TEST",  os.listdir(directory_path))
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            propagation_distance = float(re.search(r'focus--(\d+\.\d+)', filename).group(1))
            
            # Key according to propagation distance
            key = str(propagation_distance)

            results[key] = {}

            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r") as file:
                data = json.load(file)

                results[key]['tenengrad'] = data.get("tenengrad", float('-inf'))
                results[key]['normalized_gradient_variance'] = data.get("normalized_gradient_variance", float('-inf'))
                results[key]['propagation_distance'] = propagation_distance


    return results

if __name__ == "__main__":
    ## ------------------------------
    # Read input 
    ## ------------------------------
    parser = argparse.ArgumentParser(description="Evaluate focus parameters for a given file")
    parser.add_argument("input_path", type=str, help="Path to the reconstructed image file")
    parser.add_argument("output_path", type=str, help="Path to save the loaded numpy array")
    args = parser.parse_args()

    ## ------------------------------
    # Process 
    ## ------------------------------
    results = combine_all_focus_parameters_through_distances(args.input_path)
    
    ## ------------------------------
    # Persist results
    ## ------------------------------
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)


    print("Focus sharpness values saved successfully to:", args.output_path)
