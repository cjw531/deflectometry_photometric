from hardware.camera import *
from hardware.projection import *
from hardware.pattern import *
from reconstruction.gradient_reconstruction import *

from screeninfo import get_monitors

''' 1. Camera Initialization '''
cam = Flir() # Flir camera init
cam.setExposure(100000) # set exposure time (microseconds) 100k now
cam.setGain(0) # set gain

''' 2. Monitor Initialization '''
monitor_list = get_monitors() # get monitor list (offset values: x, y ; screen resolution: width, height)
projection = Screen(camera=cam, monitor_list=monitor_list, monitor_index=1, object_folder='./data/temp/')

''' 3. Generate Chessboard '''
generate_chessboard(resolution=projection.resolution, checker_pixel=100, num_col=10, num_row=6) # chessboard saved in: ./data/geometric_chessboard.png

''' 4. Capture '''
projection.capture_geometric_calibration()
projection.capture_multi_frequency(nph=4, max_frequency=16)

# Flir camera caputer images -capture one image
cam.getImage(name='test') # This image is stored as "CapturedImages/test.png"

'''
# generate chessborad to display
generate_chessboard(projection=projection, checker_pixel=80, num_col=18, num_row=10) # chessboard saved in: ./data/geometric_chessboard.png
display_chessboard(projection=projection)
'''

