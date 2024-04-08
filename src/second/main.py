from .statemanager import StateManager
from .constants import *
from .hologramreconstructor import HologramReconstructor
from .twinimagecropper import TwinImageCropper
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

class ApplicationError(Exception):
    pass

class Application:
    def __init__(self, data_directory, steps_to_run=''):
        self.state_manager = StateManager()
        self.state_manager.load_directory(data_directory)

        # Application automation path
        self.steps_to_run = steps_to_run
        self.steps_current = steps_to_run

        self.actions = [
            # 'Plot all images',
            ACTION_SELECT_REFERENCE,
            ACTION_SELECT_OBJECT,
            # 'List Current Selections',
            ACTION_PERFORM_FFT,
            ACTION_RECONSTRUCT,
            ACTION_DISPLAY_REFERENCE,
            ACTION_DISPLAY_OBJECT,
            ACTION_CLOSE,
            ACTION_DISPLAY_FFT_REFERENCE,
            ACTION_DISPLAY_FFT_OBJECT,
            # ACTION_DISPLAY_RECONSTRUCTED_PHASE,
        ]


    def run(self):
        action = None
        if self.steps_current != '':
            action = self.actions[int(self.__act_step()) - 1]
        else:
            action = self.actions[self.__display_options_base(self.actions, 'Select an action:')]

        if action == ACTION_CLOSE:
            return
        
        self.__perform_action(action)

        self.run()

    def __perform_action(self, action):
        print(f'Performing action "{action}"')

        if action == ACTION_SELECT_REFERENCE:
            selection = None
            if self.steps_current == '':
                selection = self.__display_options_base(
                    self.state_manager.paths,
                    'Select an image to act as a Reference:'
                )
            else:
                selection = int(self.__act_step()) - 1

            self.state_manager.set_reference(selection)

        elif action == ACTION_SELECT_OBJECT:
            selection = None
            if self.steps_current == '':
                selection = self.__display_options_base(
                    self.state_manager.paths,
                    'Select an image to act as an Object:'
                )
            else:
                selection = int(self.__act_step()) - 1

            self.state_manager.set_object(selection)

        elif action == ACTION_DISPLAY_REFERENCE:
            plt.imshow(self.state_manager.reference.original)

            plt.show()

        elif action == ACTION_DISPLAY_OBJECT:
            plt.imshow(self.state_manager.object.original)

            plt.show()

        elif action == ACTION_PERFORM_FFT:
            self.state_manager.reference.get_fft()
            self.state_manager.object.get_fft()

        elif action == ACTION_DISPLAY_FFT_REFERENCE:
            magnitude_spectrum = np.abs(self.state_manager.reference.get_fft())
            plt.imshow(np.log(magnitude_spectrum + 1), cmap='gray')

            plt.show()
            
        elif action == ACTION_DISPLAY_FFT_OBJECT:
            magnitude_spectrum = np.abs(self.state_manager.object.get_fft())
            plt.imshow(np.log(magnitude_spectrum + 1), cmap='gray')

            plt.show()

        elif action == ACTION_RECONSTRUCT:
            twin_image_cropper = TwinImageCropper(self.state_manager.object)
            twin_images_coordinates, twin_image_radius = twin_image_cropper.find_intensity_spots()


            # Use the top left twin image
            self.state_manager.twin_image_coordinates = twin_images_coordinates[0]
            # self.state_manager.twin_image_coordinates[0] = self.state_manager.twin_image_coordinates[0] + twin_image_radius//2
            # self.state_manager.twin_image_coordinates[1] = self.state_manager.twin_image_coordinates[1] - twin_image_radius//2
            self.state_manager.twin_image_radius = twin_image_radius

            mask_position = self.state_manager.twin_image_coordinates
            # maximum_mask_width = self.object_hologram_image.array.shape[0] // 2
            mask_width = self.state_manager.twin_image_radius
            mask_shape_index = self.state_manager.twin_image_shape_index

            propagation_distance = 10
            reconstructor = HologramReconstructor(self.state_manager.reference, self.state_manager.object, propagation_distance, mask_position, mask_width, mask_shape_index, twin_images_coordinates)

            # reconstructor.plot()
            reconstructor.plot_interactive()

            plt.show()


        elif action == ACTION_DISPLAY_RECONSTRUCTED_PHASE:
            pass

        else:
            raise ApplicationError("Selected action is not defined")

        return 0


    def __act_step(self):
        step_to_act = self.steps_current[0]
        self.steps_current = self.steps_current[1:]
        return step_to_act

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

    app = Application(directory, sequence if sequence is not None else '')

    app.run()

    print('Closing the script...')


# if __name__ == "__main__":
    # main()

