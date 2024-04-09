import argparse
import sys

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Execute code in main.py")
    parser.add_argument('parameter', choices=['pirmas', 'antras'], help="Specify the parameter")
    parser.add_argument('--sequence', help="Optional arguments for the scrip to accept")
    args = parser.parse_args()

    if args.parameter == 'pirmas':
        try:
            from src.first import main
        except ImportError:
            print("Error: Unable to import src/first/main.py.")
            sys.exit(1)
        
        main.main()
    elif args.parameter == 'antras':
        from src.second import main
        
        main.main(args.sequence)
    else:
        print("Invalid parameter. Please specify 'pirmas'.")
        sys.exit(1)

if __name__ == "__main__":
    main()

