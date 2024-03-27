from .statemanager import StateManager
import matplotlib.pyplot as plt
import numpy as np

class ApplicationError(Exception):
    pass

class Application:
    def __init__(self, data_directory):
        self.state_manager = StateManager()
        self.state_manager.load_directory(data_directory)

        self.actions = [
            # 'Plot all images',
            # 'Perform FFT',
            'Select Reference',
            'Select Object',
            # 'List Current Selections',
            'FFT',
            'Reconstruct',
            'Display Reference',
            'Close'
        ]


    def run(self, answer=''):
        action = None
        if answer != '':
            action = self.actions[int(answer[0]) - 1]
            answer = answer[1:]
        else:
            action = self.actions[self.__display_options_base(self.actions, 'Select an action:')]

        if action == 'Close':
            return
        
        consumed_answers = self.__perform_action(action, answer if answer != '' else None)
        answer = answer[consumed_answers:]

        self.run(answer)


    def __perform_action(self, action, answer = None):
        print(f'Performing action "{action}"')

        if action == 'Select Reference':
            selection = None
            if answer == None:
                selection = self.__display_options_base(
                    self.state_manager.paths,
                    'Select an image to act as a Reference:'
                )
            else:
                selection = int(answer[0]) - 1

            self.state_manager.set_reference(selection)

            return 1 if answer else 0

        elif action == 'Select Object':
            selection = None
            if answer == None:
                selection = self.__display_options_base(
                    self.state_manager.paths,
                    'Select an image to act as an Object:'
                )
            else:
                selection = int(answer[0]) - 1

            self.state_manager.set_object(selection)

            return 1 if answer else 0

        elif action == 'Display Reference':
            plt.imshow(self.state_manager.reference.original)

            plt.show()

        elif action == 'FFT':
            self.state_manager.reference.get_fft()
            self.state_manager.object.get_fft()

        elif action == 'Reconstruct':
            fft_reference = self.state_manager.reference.get_fft()
            fft_object = self.state_manager.object.get_fft()

            fft_subtracted = fft_object - fft_reference
            fft_reconstructed = np.fft.ifft2(fft_subtracted)

            magnitude_spectrum = np.abs(fft_reconstructed)

            # TODO: finish the algorithm :)

            plt.imshow(magnitude_spectrum, cmap='gray')

            # TODO: probably need to decouple this step somehow

            plt.show()

        else:
            raise ApplicationError("Selected action is not defined")

        return 0


    def __display_initial_options(self):
        options = self.state_manager.paths
        options.append('Display FFT of all images')
        
        return self.__display_options_base(options, 'Select an image to load to memory:')

    def __display_options_base(self, options, prompt): 
        print(prompt)
        for i, option in enumerate(options):
            print(f"{i+1}. {option}")

        selection = None
        while True:
            
            try:
                selection = int(input("Enter the number of the option you want to select: "))
                if 1 <= selection <= len(options):
                    break
                else:
                    print("Invalid selection. Please enter a number within the range.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        return selection - 1


        # Plot -> what to plot?
        # fft -> on last piece
        # select reference 
        # select object 


def main(sequence):
    print('Starting the script...')
    directory = 'data'

    app = Application(directory)

    app.run(sequence if sequence is not None else '')

    print('Closing the script...')


# if __name__ == "__main__":
    # main()

