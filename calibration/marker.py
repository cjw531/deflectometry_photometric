import cv2
import numpy as np
from hardware.projection import Screen

def generate_chessboard(projection: Screen, checker_pixel: int, num_col: int, num_row: int):
    '''
    Generate chessboard for geometric calibration -> generate chessboard image to display on the screen
    
    Parameters:
        @projection: Screen class, for monitor information
        @checker_pixel: pixel dimension of a single cheker
        @num_row: chessboard number of rows
        @num_col: chessboard number of columns
    '''

    # grab resolution
    width = projection.resolution[0]
    height = projection.resolution[1]

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


def display_chessboard(projection: Screen):
    '''
    Display generated chessboard image for geometric calibration, press any key to quit the projection
    
    Parameters:
        @projection: Screen class, for monitor information
    '''

    chess_img = cv2.imread("./data/geometric_chessboard.png") # read the generated chessboard image

    window_name = 'Chessboard'
    cv2.startWindowThread()
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.imshow(window_name, chess_img)
    cv2.moveWindow(window_name, projection.projection_monitor.x, projection.projection_monitor.y)
    cv2.resizeWindow(window_name, projection.projection_monitor.width, projection.projection_monitor.height)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
