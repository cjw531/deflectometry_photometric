import os
import cv2
from cv2 import aruco
import glob
import matplotlib.pyplot as plt

def verify_mirror_aruco(img_path: str):
    '''
    Check if 8 aruco markers on the mirror are being detected
    
    Parameters:
        @img_path: geometric calibration image (chessboard + 8 aruco markers in fov)
    '''

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    [markerCorners, markerIds, rejectedImgPoints] = cv2.aruco.detectMarkers(gray, aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250))
    
    if len(markerIds) != 8:
        raise ValueError('Number of markers detected: ' + str(len(markerIds)))
    else:
        print('8 Aruco markers are detected!')

def verify_overexposure(img_folder, filename=None, img_extension='*.tif'):
    '''
    Multiple ways to check the potential overexposure
    
    Parameters:
        @img_folder: captured data folder (has subfolders of period_*)
        @filename: specific file name, if not set then arbitrarily selected for plot
        @img_extension: image file type, if not set then by default *.tif format
    '''

    subfolder_count = len(glob.glob(os.path.join(img_folder, 'period_*'))) # sub-folders, period_*

    for i in range(subfolder_count):
        folder_period = os.path.join(img_folder, 'period_' + str(2 ** i)) # get subfolder
        
        # if specific filename not provided, arbitrarily select one
        if filename is None: 
            file_list = sorted(glob.glob(os.path.join(folder_period, img_extension)))
            file_plot = file_list[0]
        else:
            file_plot = os.path.join(folder_period, filename)

        max_pixel_value(folder_period, img_extension)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax_pix = pixel_distribution(file_plot, ax) # pixel distribution (0-255) histogram plot
        ax = ax_pix
        ax_red = mark_overexposure(file_plot, ax) # mark 255 values in red color
        ax = ax_red
        plt.show()

def max_pixel_value(img_folder, img_extension):
    '''
    Plot the distribution of pixel values to check the potential overexposure
    
    Parameters:
        @img_folder: captured data folder
        @img_extension: image file type, if not set then by default *.tif format
    '''

    pixel_values = []
    for filename in sorted(glob.glob(os.path.join(img_folder, img_extension))):
        gray_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # read image in grayscale
        max_val = gray_img.max() # maximum pixel value in the image
        print(f'Max pixel value in {filename}: {max_val}')
        pixel_values.append(max_val)

    print(f'Global Max: {max(pixel_values)} | Global Min: {min(pixel_values)}')

def pixel_distribution(filename, ax):
    '''
    Plot the distribution of pixel values to check the potential overexposure
    
    Parameters:
        @filename: specific file name, if not set then arbitrarily selected for plot
        @ax: axis for subplot
    '''
    
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # read image in grayscale

    # plot pixel distribution histogram
    ax[0].hist(img.ravel(), 256, [0, 256], color='gray')
    ax[0].set_xlabel('Pixel Value')
    ax[0].set_ylabel('Pixel Count')
    ax[0].set_title('Grayscale Histogram: ' + filename)

    return ax

def mark_overexposure(filename, ax):
    '''
    Mark pixels into red if it has a pixel value of 255
    
    Parameters:
        @filename: specific file name, if not set then arbitrarily selected for plot
        @ax: axis for subplot
    '''

    gray_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # read image in grayscale
    rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB) # convert the grayscale image to a 3-channel BGR image

    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            if gray_img[i, j] == 255: # if the pixel is white (255), color it red
                rgb_img[i, j] = (255, 0, 0)

    ax[1].imshow(rgb_img)
    ax[1].axis('off')
    ax[1].set_title("255 in Red")

    return ax
