import argparse
import json
import os
from second.image import Image
from second.twinimagecropper import TwinImageCropper


if __name__ == "__main__":
    ## ------------------------------
    # Read input 
    ## ------------------------------
    parser = argparse.ArgumentParser(description="Open an image file.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("output_path", type=str, help="Path to save the output image file")

    args = parser.parse_args()

    ## ------------------------------
    # Process 
    ## ------------------------------
    hologram = Image(args.image_path)

    twin_image_cropper = TwinImageCropper(hologram)
    twin_images_coordinates, twin_image_radius = twin_image_cropper.find_intensity_spots()

    ## ------------------------------
    # Persist results
    ## ------------------------------
    results = {
        "twin_images_coordinates": twin_images_coordinates.tolist(),
        "twin_image_radius": twin_image_radius
    }

    # Ensure the directory of the output file exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)


    # Save results to a JSON file
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)

    print("Results saved successfully to:", args.output_path)

