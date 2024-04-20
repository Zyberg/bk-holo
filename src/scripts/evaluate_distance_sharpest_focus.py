import argparse
import os
import json

def find_sharpest_focus(directory_path):
    max_distance = float('-inf')
    max_distance_json = None
    
    # Traverse the directory and find JSON files
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r") as file:
                data = json.load(file)
                distance = data.get("distance", float('-inf'))
                tenengrad = data.get("tenengrad", float('-inf'))
                normalized_gradient_variance = data.get("normalized_gradient_variance", float('-inf'))
                
                #TODO: add better logic to figuring out the best focus point
                if distance > max_distance:
                    max_distance = distance
                    max_distance_json = data
    
    return max_distance_json


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

    sharpest_focus_file_data = find_sharpest_focus(args.directory_path)

    ## ------------------------------
    # Persist results
    ## ------------------------------
    if sharpest_focus_file_data:
        with open(args.output_path, "w") as file:
            json.dump(sharpest_focus_file_data, file, indent=4)

        print(f"Sharpest docus JSON file found and saved to {args.output_path}")
    else:
        print("No JSON files found or 'distance' key not found in any JSON file.")
