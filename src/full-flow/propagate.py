import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift


class PropagationManager:
    def __init__(self, fx, fy, A):
        self.FX = fx
        self.FY = fy
        self.A = A


    def propagate_simple(self, d, wavelength):
        # modulation = (1j / (wavelength * d)) * np.exp(-1j * 2 * np.pi / wavelength * d) * 

        Uz = np.exp(1j * np.pi / (wavelength * d) * (self.FX**2 + self.FY**2)) * self.A
        
        # Calculate the intensity of the reconstructed wave
        intensity_image = np.abs(Uz)**2
        
        return intensity_image

    def lens_phase_correction(self, f, wavelength):
        return np.exp(-1j * (np.pi/(wavelength * f + 0.000001)) * (self.FX**2 + self.FY**2))

    def propagation_phase_correction(self, z, wavelength):
        return np.exp(-1j * np.pi /(wavelength * z + 0.000001) * (self.FX**2 + self.FY**2))
        # return precomputed_propagation_phase_correction * np.exp(1 / (z + 0.000001))
        
    # TODO: check

    # def propagation_phase_correction(z):
    #     # Correct for the propagation phase
    #     return np.exp(-1j * np.pi * (wavelength * z) * (FX**2 + FY**2) / dx**2)


    def aberration_phase_correction(self, f, wavelength):
        return np.exp(1j * np.pi / (wavelength * f + 0.000001) * (self.FX**2 + self.FY**2))


    def angular_spectrum_method_with_lens(self, z, f, wavelength):
        # Apply the propagation phase correction
        H = self.propagation_phase_correction(z, wavelength)
        
        # Apply the lens phase correction
        L = self.lens_phase_correction(f, wavelength)
        
        # Combine the corrections and compute the inverse FFT
        Uz = ifft2(ifftshift(self.A * H * L))

        # Apply the aberration phase correction
        P = self.aberration_phase_correction(f, wavelength)
        corrected_Uz = Uz * P
        
        # Calculate the intensity of the reconstructed wave
        intensity_image = np.abs(corrected_Uz)**2
        
        return intensity_image