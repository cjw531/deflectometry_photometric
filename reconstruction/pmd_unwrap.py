import os, sys, glob
import numpy as np
import cv2
from hardware.pattern import *

def phase_unwrap(obj_folder_path, img_pattern='*.tif'):
    subfolder_count = len(glob.glob(os.path.join(obj_folder_path, 'period_*'))) # subfolders w/ diff periods are saved as 'period_*'
    phase_v_prev, phase_h_prev = None, None # phases

    for i in range(subfolder_count):
        # read undistorted files, check number of images captured
        files = glob.glob(os.path.join(obj_folder_path, 'period_' + str(2 ** i), 'undistort', img_pattern))
        files.sort()
        assert (len(files) == 8), ("Images in the directory should be 8. Now is ", len(files)) 

        # set up numpy arrays to store phase/sinusodial processed images
        img = cv2.imread(files[0])
        img_set_deflectometry = np.zeros((8, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)
        img_set_deflectometry[0, ...] = img

        for j in range(len(files) - 1):
            img_set_deflectometry[j + 1, ...] = cv2.imread(files[j + 1], cv2.IMREAD_COLOR)

        # set up numpy arrays to store vertical and horizontal phase values
        [height, width, nColor] = img_set_deflectometry[0, ...].shape
        img_set_deflectometry = (img_set_deflectometry / np.max(img_set_deflectometry)).astype(np.dtype(np.float32))
        img_V_set = np.zeros((4, height, width), dtype=np.dtype(np.float32))
        img_H_set = np.zeros((4, height, width), dtype=np.dtype(np.float32))

        # separate vertical and horizontal
        for n in range(8):
            if n < int(4):
                img_V_set[n, ...] = cv2.cvtColor(img_set_deflectometry[n, ...], cv2.COLOR_BGR2GRAY)
            elif n >= int(4):
                img_H_set[n - 4, ...] = cv2.cvtColor(img_set_deflectometry[n, ...], cv2.COLOR_BGR2GRAY)

        # phase unwrap
        phase_v = np.arctan2((img_V_set[1, ...] - img_V_set[3, ...]), (img_V_set[0, ...] - img_V_set[2, ...]))
        phase_v += np.pi
        phase_h = np.arctan2((img_H_set[1, ...] - img_H_set[3, ...]), (img_H_set[0, ...] - img_H_set[2, ...]))
        phase_h += np.pi

        if i > 0: # for single period, X need further unwrapping
            F = (2 ** i) / (2 ** (i - 1))
            phase_h = phase_h + 2 * np.pi * ((F * phase_h_prev - phase_h) / (2 * np.pi))
            phase_v = phase_v + 2 * np.pi * ((F * phase_v_prev - phase_v) / (2 * np.pi))

        # store prev phase maps
        phase_v_prev = phase_v
        phase_h_prev = phase_h

    return phase_v, phase_h

def screen_phase_unwrap(resolution, obj_folder_path):
    subfolder_count = len(glob.glob(os.path.join(obj_folder_path, 'period_*'))) # subfolders w/ diff periods are saved as 'period_*'
    phase_x_prev, phase_y_prev = None, None # phases

    for i in range(subfolder_count):
        screen_pattern = SinusoidalPattern(resolution, nph=4, frequency=(2 ** i)).patterns

        phase_x = np.arctan2((screen_pattern[..., 1]* 255 - screen_pattern[..., 3]* 255), (screen_pattern[..., 0]* 255 - screen_pattern[..., 2]* 255))
        phase_x = phase_x + np.pi
        phase_y = np.arctan2((screen_pattern[..., 5]* 255 - screen_pattern[..., 7]* 255), (screen_pattern[..., 4]* 255 - screen_pattern[..., 6]* 255))
        phase_y = phase_y + np.pi

        if i > 0: # for single period, X need further unwrapping
            F = (2 ** i) / (2 ** (i - 1))
            phase_x = phase_x + 2 * np.pi * ((F * phase_x_prev - phase_x) / (2 * np.pi))
            phase_y = phase_y + 2 * np.pi * ((F * phase_y_prev - phase_y) / (2 * np.pi))

        # store prev phase maps
        phase_x_prev = phase_x
        phase_y_prev = phase_y

    return phase_x[0,...], phase_y[...,0]