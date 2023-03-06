from abc import ABC
from reconstruction.mesh import *
from hardware.pattern import *
import wavepy
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import cv2
from tifffile import imsave
import wavepy

np.seterr(divide='ignore', invalid='ignore')


class GradientReconstruction:
    def __init__(self, capture_path='./data/capture_numpy/capture_%i.npy', n=2):
        self.path = capture_path
        # Number of gradient images per viewing direction
        self.n = n
        self.frames_x = None
        self.frames_y = None
        self.frame_reflectivity = None
        self.diff_x = None
        self.diff_y = None
        self.magH = None
        self.magV = None
        self.normals = None
        self.depth = None
        self.albedo = None

    def loadData(self):
        #  Load first frame for dimensions and expand along 2nd axis to create a 3D array for all phase shifts
        self.frames_x = np.load(self.path % 0)
        self.frames_y = np.load(self.path % self.n)
        # RGB frames
        if len(self.frames_x.shape) == 3 or len(self.frames_y.shape) == 3:
            for i in range(1, self.n):
                self.frames_x = np.stack((self.frames_x, np.load(self.path % i)), axis=3)
                self.frames_y = np.stack((self.frames_y, np.load(self.path % (i + self.n))), axis=3)
            self.frame_reflectivity = np.load(self.path % (self.n * 2))
        # Grayscale frames
        else:
            self.frames_x = np.expand_dims(self.frames_x, axis=2)
            self.frames_y = np.expand_dims(self.frames_y, axis=2)
            for i in range(1, self.n):
                self.frames_x = np.dstack((self.frames_x, np.load(self.path % i)))
                self.frames_y = np.dstack((self.frames_y, np.load(self.path % (i + self.n))))
            self.frame_reflectivity = np.load(self.path % (self.n * 2))

    def saveTiff(self):
        # Saves normals and captured images in tiff file format
        cv2.imwrite('normals.tif',
                    cv2.cvtColor(np.array((self.normals + 1) / 2.0 * 255, dtype=np.uint8), cv2.COLOR_RGB2BGR))
        for i in range(self.n):
            imsave('captureX_%i.tif' % i, self.frames_x[:, :, i].astype(np.float16))
            imsave('captureY_%i.tif' % i, self.frames_y[:, :, i].astype(np.float16))

    def computeAlbedo(self):
        # Average over all frames to retrieve an albedo estimation
        # RGB frames
        if len(self.frames_x.shape) == 4 or len(self.frames_y.shape) == 4:
            stack = np.concatenate((self.frames_x, self.frames_y), axis=3)
            self.albedo = np.mean(stack, axis=3)
        # Grayscale
        else:
            stack = np.dstack((self.frames_x, self.frames_y))
            self.albedo = np.mean(stack, axis=2)

        alb = (self.albedo - np.min(self.albedo)) / (np.max(self.albedo)-np.min(self.albedo))
        alb *= 255.0
        # Save the albedo as PNG
        # RBG
        if len(self.frames_x.shape) == 4 or len(self.frames_y.shape) == 4:
            alb = cv2.cvtColor(np.array(alb, dtype=np.uint8), cv2.COLOR_BGR2RGB)
            cv2.imwrite('Results/albedo.PNG', alb)
        # Grayscale
        else:
            cv2.imwrite('Results/albedo.PNG', np.array(alb, dtype=np.uint8))

    def computeNormalMapSingle(self, gamma):
        if self.n > 2:
            print("Gradient shifting requires, two gradients per viewing direction. Average your data to fit.")
        if self.frames_y is None or self.frames_x is None:
            print("Load data first, using loadData")
        # Instead of normalizing frame by frame we normalize by the last captured image
        # this image is taken under constant maximum illumination and can therefore correct
        # for different colors and normalize pixel by pixel
        # RGB to grayscale

        if len(self.frames_x.shape) == 4 or len(self.frames_y.shape) == 4:
            frames_x_n = np.dstack(
                (color.rgb2gray(self.frames_x[..., 0]), color.rgb2gray(self.frames_x[..., 1]))).astype(np.float64)
            frames_y_n = np.dstack(
                (color.rgb2gray(self.frames_y[..., 0]), color.rgb2gray(self.frames_y[..., 1]))).astype(np.float64)
            frame_r_n = color.rgb2gray(self.frame_reflectivity).astype(np.float64) + 1e-9
        # Grayscale
        else:
            frames_x_n = self.frames_x.astype(np.float64)**gamma
            frames_y_n = self.frames_y.astype(np.float64)**gamma
            frame_r_n = self.frame_reflectivity.astype(np.float64)**gamma

        # Get overall maximum value
        max_norm = max(np.max(frames_x_n), np.max(frames_y_n))
        # It is required to remove the mean from the captured images
        frames_x_n = (frames_x_n / max_norm)
        frames_y_n = (frames_y_n / max_norm)

        # Compute difference of gradient illuminations in opposing directions
        self.diff_x = frames_x_n[..., 1] - frames_x_n[..., 0]
        self.diff_y = frames_y_n[..., 1] - frames_y_n[..., 0]
        # Compute normals
        z = np.sqrt(1 - np.square(self.diff_x) - np.square(self.diff_y))
        norm = np.sqrt(np.square(self.diff_x) + np.square(self.diff_y) + np.square(z))
        x = self.diff_x / norm
        y = self.diff_y / norm
        z /= norm
        # Create normals array
        self.normals = np.stack((x, y, z), axis=2)
        # Save as PNG file
        cv2.imwrite('Results/normals.PNG', cv2.cvtColor(np.array((self.normals + 1) / 2.0 * 255, dtype=np.uint8),
                                                        cv2.COLOR_RGB2BGR))
        np.save('Results/normals.npy', self.normals)

    def computeNormalMapRadiance(self, gamma):
        if self.n > 2:
            print("Gradient shifting requires, two gradients per viewing direction. Average your data to fit.")
        if self.frames_y is None or self.frames_x is None:
            print("Load data first, using loadData")
        # Instead of normalizing frame by frame we normalize by the last captured image
        # this image is taken under constant maximum illumination and can therefore correct
        # for different colors and normalize pixel by pixel
        # RGB to grayscale
        if len(self.frames_x.shape) == 4 or len(self.frames_y.shape) == 4:
            frames_x_n = np.dstack(
                (color.rgb2gray(self.frames_x[..., 0]), color.rgb2gray(self.frames_x[..., 1]))).astype(np.float64)
            frames_y_n = np.dstack(
                (color.rgb2gray(self.frames_y[..., 0]), color.rgb2gray(self.frames_y[..., 1]))).astype(np.float64)
            frame_r_n = color.rgb2gray(self.frame_reflectivity).astype(np.float64) + 1e-9
        # Grayscale
        else:
            frames_x_n = self.frames_x.astype(np.float64)**gamma
            frames_y_n = self.frames_y.astype(np.float64)**gamma
            frame_r_n = self.frame_reflectivity.astype(np.float64)**gamma + 1e-9

        for i in range(self.n):
            frames_x_n[..., i] = frames_x_n[..., i] #/ frame_r_n
            frames_y_n[..., i] = frames_y_n[..., i] #/ frame_r_n

        # Get overall maximum value
        max_norm = max(np.max(frames_x_n), np.max(frames_y_n))
        # It is required to remove the mean from the captured images
        frames_x_n = (frames_x_n / max_norm)
        frames_y_n = (frames_y_n / max_norm)

        # Compute difference of gradient illuminations in opposing directions
        self.diff_x = frames_x_n[..., 1] - frames_x_n[..., 0]
        self.diff_y = frames_y_n[..., 1] - frames_y_n[..., 0]
        # Compute normals
        z = np.sqrt(1 - np.square(self.diff_x) - np.square(self.diff_y))
        norm = np.sqrt(np.square(self.diff_x) + np.square(self.diff_y) + np.square(z))
        x = self.diff_x / norm
        y = self.diff_y / norm
        z /= norm
        # Create normals array
        self.normals = np.stack((x, y, z), axis=2)
        # Save as PNG file
        cv2.imwrite('Results/normals.PNG', cv2.cvtColor(np.array((self.normals + 1) / 2.0 * 255, dtype=np.uint8),
                                                        cv2.COLOR_RGB2BGR))
        np.save('Results/normals.npy', self.normals)

    def computePointCloud(self, crop=((0, 0), (0, 0))):
        # Crop by ((start_of_crop_x, height_of_crop_x), (start_of_crop_y, height_of_crop_y))
        # IMPORTANT: Note the second tuple parameters are the cropping height/width, not the x/y crop end positions !!
        if crop[0][1] == 0 or crop[0][1] == 0:
            height = self.frames_x.shape[0]
            width = self.frames_x.shape[1]
        else:
            height = crop[0][1]
            width = crop[1][1]
        crop_x0 = crop[0][0]
        crop_y0 = crop[1][0]
        crop_x1 = crop_x0 + height
        crop_y1 = crop_y0 + width
        # Create mesh object and set normals, albedo, and compute depth
        meshGI = Mesh("GradientIllumination", width, height)
        meshGI.setNormal(self.normals[crop_y0:crop_y1, crop_x0:crop_x1, ...])
        meshGI.setTexture(self.albedo[crop_y0:crop_y1, crop_x0:crop_x1, ...])
        self.depth = meshGI.setDepth()
        depth_png = (self.depth - np.min(self.depth)) / (np.max(self.depth) - np.min(self.depth))
        depth_png *= 255.0
        cv2.imwrite('Results/depth.PNG', depth_png)
        # Create mesh obj to export
        meshGI.exportOBJ("Results/", True)

    def highPassFilter(self):
        # Smooth filter operation
        kernel = np.ones((5, 5), np.float32) / 25
        self.normals = cv2.filter2D(self.normals, -1, kernel)

class GradientCapture:
    def __init__(self, camera, projection, image_processing, calibration=None, n=4):
        self.camera = camera
        self.projection = projection
        # Set gradient pattern as it's dimensions depends on the projection resolution
        self.projection.setPattern(GradientPattern(projection.resolution))
        self.calibration = calibration
        self.image_processing = image_processing
        # Number of gradient images: default 4 - X&Y x R
        self.n = n

    def capture(self, red=1.0, green=1.0, blue=1.0):
        self.projection.pattern.createGradientXY(self.n, red=red, blue=blue, green=green)
        # Display patterns, take photos and save as np and jpg
        self.projection.displayPatterns(self.camera)

    def compute(self):
        if self.projection.root is not None:
            self.projection.quit_and_close()
        self.image_processing.loadData()
        gamma = self.calibration.radio_calib.gamma
        if self.camera.hdr_exposures is None:
            self.image_processing.computeNormalMapSingle(gamma)
        else:
            self.image_processing.computeNormalMapRadiance(gamma)
        self.image_processing.computeAlbedo()
        #self.image_processing.saveTiff()

    def calibrate(self, calibration):
        # Calibrate camera using a calibration object obtained from a Calibration Session
        self.calibration = calibration
        self.camera.setCalibration(calibration)