import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from second.image import Image
from second.hologramreconstructionmanager import HologramReconstructionManager

def load_twin_coordinates(twin_coordinates_path):
    try:
        with open(twin_coordinates_path, 'r') as f:
            twin_data = json.load(f)
            twin_images_coordinates = np.array(twin_data["twin_images_coordinates"])
            twin_image_radius = twin_data["twin_image_radius"]
            return twin_images_coordinates, twin_image_radius
    except FileNotFoundError:
        print("Twin coordinates file not found. Please provide a valid path.")
        return None, None
    except Exception as e:
        print("An error occurred while loading twin coordinates:", e)
        return None, None

if __name__ == "__main__":
    ## ------------------------------
    # Read input 
    ## ------------------------------
    parser = argparse.ArgumentParser(description="Process twin coordinates.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("twin_coordinates_path", type=str, help="Path to the twin coordinates file")
    parser.add_argument("twin_image_direction", type=str, help="Twin image direction")
    parser.add_argument("output_path", type=str, help="Path to save the output file")
    args = parser.parse_args()

    # Load twin coordinates
    twin_images_coordinates, twin_image_radius = load_twin_coordinates(args.twin_coordinates_path)

    if twin_images_coordinates is None and twin_image_radius is None:
        raise "Provide correct path for twin coordinates .json file!"

    ## ------------------------------
    # Process 
    ## ------------------------------
    hologram = Image(args.image_path)

    print("Twin images coordinates:", twin_images_coordinates)
    print("Twin image radius:", twin_image_radius)

    twin_image_index = 0 if args.twin_image_direction == 'left' else 2
    # TODO: read args for mask_shape_index


    coords = twin_images_coordinates[twin_image_index]
    coords[0] = coords[0] - 15
    coords[1] = coords[1] - 15

    hologram_reconstruction_manager = HologramReconstructionManager(
        hologram,
        [1200, 1270], # twin_images_coordinates[twin_image_index],
        800, #int(twin_image_radius),
        # 1 # Circle
        0 # Square
    )

    reconstructed_field, mask = hologram_reconstruction_manager.reconstruct()

    ## ------------------------------
    # Persist results
    ## ------------------------------
    np.save(args.output_path, reconstructed_field)


    base_output_path, output_extension = os.path.splitext(args.output_path)

    fig, ax = plt.subplots()
    ax.imshow(np.log(np.abs(hologram.get_shifted()) + 1), cmap='gray')
    ax.imshow(np.ma.masked_where(mask == False, mask), cmap='gray', alpha=0.5)

    fig.savefig(f'{base_output_path}.mask.png', dpi=300)
    
    plt.close(fig)


    print("Output saved successfully to:", args.output_path)
