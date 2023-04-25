import numpy as np
import cv2

def generate_chessboard(resolution: tuple, checker_pixel: int, num_col: int, num_row: int):
    '''
    Generate chessboard for geometric calibration -> generate chessboard image to display on the screen
    
    Parameters:
        @resolution: monitor resolution
        @checker_pixel: pixel dimension of a single checker
        @num_row: chessboard number of rows
        @num_col: chessboard number of columns
    '''

    # grab resolution
    width = resolution[0]
    height = resolution[1]

    # exception handling (invalid inputs)
    if checker_pixel > min(width, height):
        raise Exception('Single checker pixel dimension is larger than the width/height of the monitor!')
    elif (num_col * checker_pixel) > width:
        raise Exception('Number of columns exceeds the board dimensions!')
    elif (num_row * checker_pixel) > height:
        raise Exception('Number of rows exceeds the board dimensions!')

    checker_width = num_col * checker_pixel
    checker_height = num_row * checker_pixel

    # find chessboard drawing starting point
    pading_x = (width - checker_width) // 2 # calculate width offset
    pading_y = (height - checker_height) // 2 # calculate height offset

    chessboard = np.full((height, width), 255, dtype=np.uint8)
    white = False
    for i in range(num_row):
        for j in range(num_col):
            x1, y1 = j * checker_pixel + pading_x, i * checker_pixel + pading_y
            x2, y2 = x1 + checker_pixel, y1 + checker_pixel
            if not white:
                chessboard[y1:y2, x1:x2] = 0
            white = not white
        white = not white
    
    cv2.imwrite('./data/geometric_chessboard.png', chessboard)
    

class GradientPattern:
    def __init__(self, resolution, nph=2, frequency=0):
        # Tuple Resolution of display/screen in pixels (Width, Height)
        if resolution[0] < resolution[1]:
            self.resolution = reversed(resolution)
        else:
            self.resolution = resolution
            
        self.frequency = frequency # Set frequency depending on screen resolution
        self.calibration = None
        self.nph = nph
        self.patterns = self.createGradientXY()

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
    def __init__(self, resolution, nph=4, frequency=4):
        if resolution[0] < resolution[1]:
            self.resolution = reversed(resolution)
        else:
            self.resolution = resolution
        
        self.frequency = frequency
        self.nph = nph
        self.patterns = self.createSinusXY()
    
    def createSinusXY(self):
        # Number of phase shifts: nph
        # Set up pattern list to store phase shift images (X and Y direction)
        self.patterns = np.zeros((self.resolution[1], self.resolution[0], 3, self.nph * 2))
        
        # Loop of number_of_phase_shifts to create sinusoidal patterns in X and Y direction
        x_phase = []
        y_phase = []
        for i in range(self.nph):
            k = i - 1
            period = self.frequency * 2
            sin_x = 0.5 + 0.5 * np.sin(np.linspace(0, (period * np.pi), self.resolution[0]) + 0.5 * k * np.pi)
            img = np.tile(sin_x, (self.resolution[1], 1))
            x_phase.append(img)

            period = (self.resolution[1] * self.frequency * 2) / self.resolution[0]
            sin_y = 0.5 + 0.5 * np.sin(np.linspace(0, (period * np.pi), self.resolution[1]) + 0.5 * k * np.pi)
            img = np.rot90(np.tile(sin_y[:self.resolution[1]], (self.resolution[0], 1)), k=3)
            y_phase.append(img)

        x_phase = np.stack(x_phase, axis=-1)
        y_phase = np.stack(y_phase, axis=-1)
        self.patterns = np.concatenate((x_phase, y_phase), axis=-1)
        return self.patterns
