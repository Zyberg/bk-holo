import argparse
import sys

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Execute code in main.py")
    parser.add_argument('parameter', choices=['pirmas'], help="Specify the parameter")
    args = parser.parse_args()

    # Check if parameter is 'pirmas'
    if args.parameter == 'pirmas':
        try:
            # Import the module
            from src.first import main
        except ImportError:
            print("Error: Unable to import src/first/main.py.")
            sys.exit(1)
        
        # Call the main function from the imported module
        main.main()
    else:
        print("Invalid parameter. Please specify 'pirmas'.")
        sys.exit(1)

if __name__ == "__main__":
    main()

