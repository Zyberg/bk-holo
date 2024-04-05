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
            reconstructor = HologramReconstructor(self.state_manager.reference, self.state_manager.object)
            
            twin_image_cropper = TwinImageCropper(self.state_manager.object)

            # Adjust the threshold value as needed
            threshold_value = 100  # Adjust this value based on the intensity distribution of your hologram FFT image

            # Find the cropping region around the twin images
            # cropping_region = twin_image_cropper.find_twin_image_cropping_region()

            # # Print the cropping region (xmin, ymin, xmax, ymax)
            # print("Cropping Region:", cropping_region)

            
            # reconstructor.plot()
            reconstructor.plot_interactive()

            # # fft_reference = self.state_manager.reference.get_fft()
            # # fft_object = self.state_manager.object.get_fft()

            # fft_reference = fft2(reference_hologram)
            # fft_object = fft2(object_hologram)

            # ff = fft_object/fft_reference
            # mask = ff[:ff.shape[0]//2, :ff.shape[1]//2]

            # # plt.imshow(np.log(np.abs(mask) + 1), cmap='gray')
            # # plt.show()
            # # interference_pattern = object_hologram - reference_hologram

            # # plt.imshow(interference_pattern, cmap='gray')

            # # plt.show()
            # # Extract twin images from the hologram Fourier transform
            # # twin_image1, twin_image2 = extract_twin_images(reference_hologram, object_hologram)
            # # chosen_twin_image = twin_image1 if np.sum(np.abs(twin_image1)) > np.sum(np.abs(twin_image2)) else twin_image2

            # reconstructed_phase, reconstructed_intensity = reconstruct_phase_and_intensity(mask, reference_hologram)

            # # Create a mask highlighting the space that is masked out
            # # mask = np.abs(twin_image1)
            
            # # mask[np.where(chosen_twin_image == 0)] = 1

            # # Reconstruct object from the first twin image
            # # reconstructed_object_image = reconstruct_object_from_twin(twin_image1)

            # # fft_reconstructed = ifft2(fft_subtracted)

            # # magnitude_spectrum = np.abs(fft_reconstructed)
            # # reconstructed_phase_gs = reconstruct_phase_off_axis(reference_hologram, object_hologram)
            # # reconstructed_phase_gs = gersbach_phase_retrieval(reference_hologram, object_hologram)
            # # phase_spectrum = reconstruct_phase_TIE(interference_pattern)

            # # Reconstruct real object image
            # # reconstructed_object_image = reconstruct_object_image(reconstructed_phase_gs, object_hologram)

            # # plt.imshow(phase_spectrum, cmap='gray')
            # # plt.imshow(magnitude_spectrum, cmap='gray')


            # # TODO: probably need to decouple this step somehow

            # # Plotting
            # fig, axes = plt.subplots(1, 4, figsize=(18, 6))

            # # Original image
            # axes[0].imshow(object_hologram, cmap='gray')
            # axes[0].set_title('Original Image')

            # # Reconstructed phase
            # axes[1].imshow(reconstructed_phase, cmap='gray')
            # axes[1].set_title('Reconstructed Phase (GS Algorithm)')

            # # Reconstructed real object image
            # axes[2].imshow(reconstructed_intensity, cmap='gray')
            # axes[2].set_title('Reconstructed Real Object Image')

            # axes[3].imshow(np.abs(mask), cmap='gray')
            # axes[3].set_title('Masked Out Space')

            # plt.tight_layout()
            # plt.show()


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

