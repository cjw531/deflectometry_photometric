from abc import ABC
from PIL import Image, ImageTk
import numpy as np
import cv2
import time
import warnings
warnings.filterwarnings("ignore", message="module not found")


class Screen:
    def __init__(self, frequency=0, monitor_list=None, monitor_index=0):
        self.projection_monitor = monitor_list[monitor_index] # connected monitors

        print("Screen resolution: ", (self.projection_monitor.width, self.projection_monitor.height))
        resolution = (self.projection_monitor.width, self.projection_monitor.height)

        # Tuple Resolution of display/screen in pixels (Width, Height)
        if resolution[0] < resolution[1]:
            self.resolution = reversed(resolution)
        else:
            self.resolution = resolution
        # Frequency Projections
        self.frequency = frequency
        self.calibration = None
        
        self.count = 0
        # Create empty pattern and camera
        self.pattern = None
        self.camera = None

    def displayCalibrationPattern(self, camera, path_calib='CalibrationImages/8_24_checker.png',
                                  save_img='Geometric/geo'):
        self.camera = camera
        img = cv2.imread(path_calib)
        cv2.namedWindow('Checkerboard', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Checkerboard', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Checkerboard', img)
        time.sleep(2)
        self.camera.getImage(name=save_img, calibration=True)
        cv2.waitKey(0)  # any key
        cv2.destroyWindow('Checkerboard')

    def displayPatterns(self, camera=None):
        self.camera = camera
        window_name = 'Pattern'
        
        for i in range(0, self.pattern.patterns.shape[-1]):
            modulation = (self.pattern.patterns[..., i] * 255).astype(np.uint8)
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
                    self.camera.getImage(name='capture_'+str(i))
                else:
                    self.camera.getHDRImage(name='capture_'+str(i))
                cv2.waitKey(int(self.camera.exposure))
        
        cv2.destroyAllWindows()

    def setPattern(self, pattern_type):
        # Sets pattern to project
        self.pattern = pattern_type.patterns

    def getResolution(self):
        # Returns tuple of resolution (width x height)
        return self.resolution

    def setResolution(self, resolution):
        # Sets tuple of resolution (width x height)
        self.resolution = resolution

    def quit_and_close(self):
        # Close the projection
        cv2.destroyAllWindows()
