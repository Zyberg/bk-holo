import argparse
import os
import json
import matplotlib.pyplot as plt

def get_normalized_gradient_variance(directory_path):
    distances = []
    variances = []
    
    # Traverse the directory and find JSON files
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r") as file:
                data = json.load(file)
                distances.append(data.get("distance", float('-inf')))
                variances.append(data.get("normalized_gradient_variance", float('-inf')))
                
    return distances, variances


if __name__ == "__main__":
    ## ------------------------------
    # Read input 
    ## ------------------------------
    parser = argparse.ArgumentParser(description="Find JSON file with highest 'distance' value")
    parser.add_argument("output_path", type=str, help="Output JSON file path")
    parser.add_argument("directory_path", type=str, help="Directory path containing JSON files")
    args = parser.parse_args()

    ## ------------------------------
    # Process 
    ## ------------------------------

    distances, variances = get_normalized_gradient_variance(args.directory_path)

    ## ------------------------------
    # Persist results
    ## ------------------------------
    fig, ax = plt.subplots()

    ax.scatter(distances, variances, color='red')

    fig.savefig(args.output_path, dpi=300)
    
    plt.close(fig)

    print("Gradient variance plot saved successfully to:", args.output_path)

