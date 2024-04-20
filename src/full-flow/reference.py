import numpy as np

def model_reference_wave(nx, ny, wavelength, angle, pixel_size):
    # Assuming the reference wave is incident in the x-direction at an angle
    k = 2 * np.pi / wavelength
    dx = pixel_size
    x = np.linspace(-nx/2, nx/2, nx) * dx
    # Calculate the wavevector component in the x-direction
    kx = k * np.sin(angle)
    
    # Calculate the phase shift for each point on the reference wave
    reference_wave = np.exp(1j * kx * x)
    
    # Extend to 2D
    reference_wave = np.tile(reference_wave, (ny, 1))
    
    return reference_wave

def model_spherical_reference_wave(nx, ny, dx, dy, wavelength, z):
    # Create coordinate grid
    x = np.linspace(-nx//2, nx//2, nx) * dx
    y = np.linspace(-ny//2, ny//2, ny) * dy
    X, Y = np.meshgrid(x, y)

    # Calculate distance R from each point on the grid to the source
    R = np.sqrt(X**2 + Y**2 + z**2)

    # Wave number
    k = 2 * np.pi / wavelength

    # Calculate the complex amplitude of the spherical wave
    E_R = np.exp(1j * k * R) / R  # Including the 1/R term to account for spreading

    return E_R
