from .imageprocessor import ImageProcessor

def main():
    directory = 'data'
    image_processor = ImageProcessor(directory)

    image_processor.display_image_options()

    while True:
        try:
            selection = int(input("Enter the number of the image you want to select: "))
            if 1 <= selection <= len(image_processor.images) + 1:
                break
            else:
                print("Invalid selection. Please enter a number within the range.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    image_processor.perform_fft(selection if selection <= len(image_processor.images) else None)

# if __name__ == "__main__":
    # main()

