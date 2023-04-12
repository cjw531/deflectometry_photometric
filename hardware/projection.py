import os
import numpy as np
import cv2
import time
import warnings
from hardware.pattern import *
warnings.filterwarnings("ignore", message="module not found")


class Screen:
    def __init__(self, camera=None, monitor_list=None, monitor_index=0):
        self.projection_monitor = monitor_list[monitor_index] # connected monitors

        print("Screen resolution: ", (self.projection_monitor.width, self.projection_monitor.height))
        resolution = (self.projection_monitor.width, self.projection_monitor.height)

        # Tuple Resolution of display/screen in pixels (Width, Height)
        if resolution[0] < resolution[1]:
            self.resolution = reversed(resolution)
        else:
            self.resolution = resolution
        
        self.pattern = None
        self.camera = camera

    def displayCalibrationPattern(self, camera, path_calib='CalibrationImages/8_24_checker.png', save_img='Geometric/geo'):
        self.camera = camera
        img = cv2.imread(path_calib)
        cv2.namedWindow('Checkerboard', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Checkerboard', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Checkerboard', img)
        time.sleep(2)
        self.camera.getImage(name=save_img, calibration=True)
        cv2.waitKey(0)  # any key
        cv2.destroyWindow('Checkerboard')

    def capture_multi_frequency(self, object_folder='./data/obj/', nph=4, max_frequency=16):
        '''
        Multi-frequency capture automation based on maximum period
    
        Parameters:
            @object_folder: parent folder to store all the sub multi-frequence projection
            @nph: number of phase shift
            @max_frequency: maximum period for the sinusoidal pattern; suppose to be a value of 2^n
        '''

        if not os.path.exists(object_folder): # if folder does not exist
            os.makedirs(object_folder) # create a new folder

        power = 0
        period = 2 ** power # start with single-period sinusoidal pattern, 2^0
        while (period <= max_frequency): # only up until max freq
            sub_folder_path = os.path.join(object_folder, 'period_' + str(period))
            self.set_pattern(SinusoidalPattern(self.resolution, nph=nph, frequency=period)) # set fringe pattern
            self.capture_with_pattern(img_folder_path=sub_folder_path) # capture data
            power += 1 # increment exponential
            period = 2 ** power # re-assign period

    def capture_with_pattern(self, img_folder_path='./data/capture_img/'):
        window_name = 'Pattern'
        
        for i in range(0, self.pattern.shape[-1]):
            modulation = (self.pattern[..., i] * 255).astype(np.uint8)
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.imshow(window_name, modulation)
            cv2.moveWindow(window_name, self.projection_monitor.x, self.projection_monitor.y)
            cv2.resizeWindow(window_name, self.projection_monitor.width, self.projection_monitor.height)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.waitKey(1000) # delay required so that pattern can be displayed
    
            if self.camera is None:
                print("No camera initialized.")
            else: # Take snapshot
                if self.camera.hdr_exposures is None:
                    self.camera.getImage(name='capture_' + str(i), img_folder_path=img_folder_path)
                else:
                    self.camera.getHDRImage(name='capture_' + str(i))

                # Flir exposure time is 30 microseconds while opencv takes milliseconds
                cv2.waitKey(int(self.camera.exposure / 900)) # not / 1000 and give more time
        
        cv2.destroyAllWindows()

    def set_pattern(self, pattern_type):
        # Sets pattern to project
        self.pattern = pattern_type.patterns

    def get_resolution(self):
        # Returns tuple of resolution (width x height)
        return self.resolution

    def set_resolution(self, resolution):
        # Sets tuple of resolution (width x height)
        self.resolution = resolution

