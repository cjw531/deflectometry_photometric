from hardware.camera import *

'''
Capture images with tablet (disconnected) monitor
1. Automated logic based on exposure time
2. Manual capture

Assumption:
Android tablet must have an app that creates chessboard pattern (geometric calibration) and sinusoidal pattern
'''

''' 0. Camera Initialization '''
cam = Flir() # Flir camera init
cam.setExposure(100000) # set exposure time (microseconds) 100k now == 0.1 seconds
cam.setGain(0) # set gain

''' 1. Capture Session: define manual capture or not '''
cam.capture_tablet(object_path='./data/capture_img/', nph=4, max_frequency=16, manual=True)
