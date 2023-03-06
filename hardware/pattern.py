import numpy as np
from PIL import Image, ImageTk

class GradientPattern:
    def __init__(self, resolution, frequency=0):
        # Tuple Resolution of display/screen in pixels (Width, Height)
        if resolution[0] < resolution[1]:
            self.resolution = reversed(resolution)
        else:
            self.resolution = resolution
            
        self.frequency = frequency
        self.calibration = None
        self.patterns = None
        # Set frequency depending on screen resolution

    def createGradientXY(self, n=2, red=1.0, green=1.0, blue=1.0):
        # Number gradient shifts n:
        # Set up pattern list to store phase shift images (X and Y direction)
        self.patterns = np.zeros((self.resolution[1], self.resolution[0], 3, n*2+1))
        # Create constant pattern
        c = np.ones((self.resolution[1], self.resolution[0]))
        # Create gradient
        x = np.linspace(0, 1, self.resolution[0])
        y = np.linspace(0, 1, self.resolution[1])
        # Reverse gradient
        xR = np.flipud(x)
        yR = np.flipud(y)
        # Create mesh grid
        [gradientX, gradientY] = np.meshgrid(x, y)
        [gradientXR, gradientYR] = np.meshgrid(xR, yR)

        # Set up pattern
        self.patterns[:, :, 0, 0] = red * gradientX
        self.patterns[:, :, 1, 0] = green * gradientX
        self.patterns[:, :, 2, 0] = blue * gradientX
        self.patterns[:, :, 0, 1] = red * gradientXR
        self.patterns[:, :, 1, 1] = green * gradientXR
        self.patterns[:, :, 2, 1] = blue * gradientXR
        self.patterns[:, :, 0, 2] = red * gradientY
        self.patterns[:, :, 1, 2] = green * gradientY
        self.patterns[:, :, 2, 2] = blue * gradientY
        self.patterns[:, :, 0, 3] = red * gradientYR
        self.patterns[:, :, 1, 3] = green * gradientYR
        self.patterns[:, :, 2, 3] = blue * gradientYR
        self.patterns[:, :, 0, 4] = red * c
        self.patterns[:, :, 1, 4] = green * c
        self.patterns[:, :, 2, 4] = blue * c
        return self.patterns


class SinusoidalPattern:
    def __init__(self, resolution, frequency=0):
        super().__init__(resolution, frequency)
        self.x = np.linspace(1, self.resolution[0], self.resolution[0])
        self.y = np.linspace(1, self.resolution[1], self.resolution[1])
        [self.X, self.Y] = np.meshgrid(self.x, self.y)
        self.patterns = None
        # Set frequency depending on screen resolution
        self.frequency = 1 / self.resolution[0]

    def createSinusXY(self, nph, red=1.0, green=1.0, blue=1.0):
        # Number of phase shifts: nph
        # Set up pattern list to store phase shift images (X and Y direction)
        self.patterns = np.zeros((self.resolution[1], self.resolution[0], 3, nph * 2))
        # Set up phase map
        phaseX = self.frequency * self.X
        phaseY = self.frequency * self.Y
        # Loop of number_of_phase_shifts to create sinusoidal patterns in X and Y direction
        for i in range(nph):
            # Calculate Phase Shifts in X and Y direction
            # Simple formula to create fringes between 0 and 1:
            phase_shift = i * 2 * np.pi / nph
            self.patterns[:, :, 0, i] = red * (np.cos(phaseX - phase_shift) + 1) / 2
            self.patterns[:, :, 1, i] = green * (np.cos(phaseX - phase_shift) + 1) / 2
            self.patterns[:, :, 2, i] = blue * (np.cos(phaseX - phase_shift) + 1) / 2
            self.patterns[:, :, 0, i + nph] = red * (np.cos(phaseY - phase_shift) + 1) / 2
            self.patterns[:, :, 1, i + nph] = green * (np.cos(phaseY - phase_shift) + 1) / 2
            self.patterns[:, :, 2, i + nph] = blue * (np.cos(phaseY - phase_shift) + 1) / 2
        return self.patterns

