import os, math, time
import PySpin
import numpy as np
import cv2
from IPython.display import clear_output, Image, display, update_display
import PIL
# from Cameras.liveDisplay import ptgCamStream, imgShow, patternDisplay

class Flir:
    def __init__(self, exposure=0.01, white_balance=1, auto_focus=False, grayscale=False):
        self.sessionDir = None
        self._isMonochrome = True
        self._is16bits = True

        # # self.NumPatterns = NUM_PATTERN
        # self.displayWidth = DISPLAY_WIDTH
        # self.displayHeight = DISPLAY_HEIGHT
        # self.setDefPattern()

        self.Cam = PySpinCapture(0, self._isMonochrome, self._is16bits)
        self.height = self.Cam.height
        self.width = self.Cam.width

        fps = self.getFPS() # get frame rate
        resolution = self.getResolution() # get camera resolution
        
        self.exposure = exposure # Exposure passed as float in seconds
        self.white_balance = white_balance # White balanced passed as a float
        self.auto_focus = auto_focus # Auto_focus passed as boolean
        self.fps = fps # FPS in float
        self.resolution = resolution # Resolution as tuple (Width, Height)
        self.grayscale = grayscale # Grayscale in boolean
        self.capture = None # Capture object may be in cv2.capture, pypylon, PySpin etc.
        self.calibration = None # Calibration object
        self.hdr_exposures = None

    def getImage(self, name='test', img_folder_path='./data/capture_img/', saveImage=True, calibration=False, calibrationName=None):
        if not os.path.exists(img_folder_path): # if folder does not exist, recreate
            os.makedirs(img_folder_path)

        filenamePNG = '' # init img and numpy save path name
        if calibration:
            if calibrationName is None: # calibration subfolder name NOT defined
                filenamePNG = os.path.join(img_folder_path, name + '.tif')
            else: # calibration subfolder name defined
                filenamePNG = os.path.join(img_folder_path + calibrationName,  name + '.tif')
        else: # simple capture, non-calibration
            filenamePNG = os.path.join(img_folder_path, name + '.tif')

        try:
            _, img = self.Cam.grabFrame() # Take and return current camera frame
            if saveImage: # image save
                cv2.imwrite(filenamePNG, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            return img

        except SystemError:
            self.quit_and_open()
            return None

    def capture_tablet(self, object_path='./data/capture_img/', nph=4, max_frequency=16, manual=True):
        '''
        Image capture for tablet, allow either exposure time based or manual capture with keyboard input
        '''

        if not os.path.exists(self.object_folder): # if folder does not exist
            os.makedirs(self.object_folder) # create a new folder

        power = 0
        period = 2 ** power # start with single-period sinusoidal pattern, 2^0
        while (period <= max_frequency): # only up until max freq
            if manual:
                capture_input = input("Capture ready?: ") # make a pause if it is a manual capture
                
            sub_folder_path = os.path.join(object_path, 'period_' + str(period)) # create period_* subfolder under object folder
            
            for i in range(nph * 2): # number of phase shift * 2 (x & y direction)
                self.getImage(name='capture_' + str(i), img_folder_path=sub_folder_path) # capture
            
            power += 1 # increment exponential
            period = 2 ** power # re-assign period

    def setExposure(self, exposure):
        self.exposure = exposure
        self.Cam.setExposure(exposure)

    def getFPS(self):
        return self.Cam.getFPS()

    def setFPS(self, fps):
        self.Cam.setFPS(fps)

    def setAutoGain(self):
        self.Cam.setCamAutoProperty()

    def getGain(self):
        return self.Cam.getGain()

    def setGain(self, gain):
        self.Cam.setGain(gain)

    def getResolution(self):
        return self.Cam.getResolution()

    def setResolution(self, resolution):
        self.Cam.setWidth(resolution[0])
        self.Cam.setHeight(resolution[1])

    def viewCameraStream(self):
        img = self.getImage(saveImage=False, saveNumpy=False)
        
        while True:
            _, img = self.Cam.grabFrameCont()
            cv2.imshow('FLIR camera image', img)
            c = cv2.waitKey(1)
            if c != -1:
                self.Cam._camera.EndAcquisition() # When everything done, release the capture
                cv2.destroyAllWindows()
                self.quit_and_open()
                break

    def quit_and_close(self):
        self.Cam.release()

    def quit_and_open(self):
        self.Cam.release()
        self.Cam = PySpinCapture(1, self._isMonochrome, self._is16bits)


class PySpinCapture:
    def __init__(self, index=0, isMonochrome=False, is16bits=False):
        # [Current Support] Single camera usage(select by index)
        self._system = PySpin.System.GetInstance()
        # Get current library version
        self._version = self._system.GetLibraryVersion()
        print('Library version: {}.{}.{}.{}\n'.format(self._version.major, self._version.minor, self._version.type,
                                self._version.build))
        self.index = index
        self.getNumCams()
        self._camera = self._cameraList.GetByIndex(index)
        self._camera.Init()
        self._isMonochrome = isMonochrome
        self._is16bits = is16bits

        self._nodemap = self._camera.GetNodeMap()
        self.getNode()

        self.setAcquisitMode(1)
        self.setPixel()
        self.setSize()
        self.setCamAutoProperty(False)
        #self._camera.BeginAcquisition()

    def __enter__(self):
        return self

    def reset(self):
        self.__init__()

    def getNumCams(self):
        self._cameraList = self._system.GetCameras()

    def print_retrieve_node_failure(self, node, name):
        print('Unable to get {} ({} {} retrieval failed.)'.format(node, name, node))
        print('The {} may not be available on all camera models...'.format(node))
        print('Please try a Blackfly S camera.')

    def getNode(self):
        # Acquisition Mode Node
        self.nodeAcquisitionMode = PySpin.CEnumerationPtr(self._nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(self.nodeAcquisitionMode) or not PySpin.IsWritable(self.nodeAcquisitionMode):
            print("Unable to set acquisition mode to continuous (enum retrieval). Aborting...")
            return False

        # Retrieve entry node from enumeration node
        nodeContinuousAcquisition = self.nodeAcquisitionMode.GetEntryByName("Continuous")
        if not PySpin.IsAvailable(nodeContinuousAcquisition) or not PySpin.IsReadable(
                nodeContinuousAcquisition):
            print("Unable to set acquisition mode to continuous (entry retrieval). Aborting...")
            return False

        self.nodeAcquisitionContinuous = nodeContinuousAcquisition.GetValue()

        # Retrieve entry node from enumeration node
        nodeSingleAcquisition = self.nodeAcquisitionMode.GetEntryByName("SingleFrame")
        if not PySpin.IsAvailable(nodeSingleAcquisition) or not PySpin.IsReadable(
                nodeSingleAcquisition):
            print("Unable to set acquisition mode to Single Frame (entry retrieval). Aborting...")
            return False
        self.nodeAcquisitionSingle = nodeSingleAcquisition.GetValue()

        # Pixel Format Node
        self.nodePixelFormat = PySpin.CEnumerationPtr(self._nodemap.GetNode('PixelFormat'))

        nodePixelFormatMono8 = PySpin.CEnumEntryPtr(self.nodePixelFormat.GetEntryByName('Mono8'))
        self.pixelFormatMono8 = nodePixelFormatMono8.GetValue()

        nodePixelFormatMono16 = PySpin.CEnumEntryPtr(self.nodePixelFormat.GetEntryByName('Mono16'))
        self.pixelFormatMono16 = nodePixelFormatMono16.GetValue()

        # nodePixelFormatRGB8 = PySpin.CEnumEntryPtr(
        #     self.nodePixelFormat.GetEntryByName('BayerRG8'))
        # self.pixelFormatRGB8 = nodePixelFormatRGB8.GetValue()
        #
        # nodePixelFormatRGB16 = PySpin.CEnumEntryPtr(
        #     self.nodePixelFormat.GetEntryByName('BayerRG16'))
        # self.pixelFormatRGB16 = nodePixelFormatRGB16.GetValue()

        # Image Size Node
        self.nodeWidth = PySpin.CIntegerPtr(self._nodemap.GetNode('Width'))
        self.nodeHeight = PySpin.CIntegerPtr(self._nodemap.GetNode('Height'))

        # Exposure Node
        self._nodeExposureAuto = PySpin.CEnumerationPtr(self._nodemap.GetNode('ExposureAuto'))
        if not PySpin.IsAvailable(self._nodeExposureAuto) or not PySpin.IsWritable(self._nodeExposureAuto):
            self.print_retrieve_node_failure('node', 'ExposureAuto')
            return False

        self._exposureAutoOff = self._nodeExposureAuto.GetEntryByName('Off')
        if not PySpin.IsAvailable(self._exposureAutoOff) or not PySpin.IsReadable(self._exposureAutoOff):
            self.print_retrieve_node_failure('entry', 'ExposureAuto Off')
            return False

        if PySpin.IsAvailable(self._nodeExposureAuto) and PySpin.IsWritable(self._nodeExposureAuto):
            self._exposureAutoOn = self._nodeExposureAuto.GetEntryByName('Continuous')

        self._nodeExposure = PySpin.CFloatPtr(self._nodemap.GetNode('ExposureTime'))
        self._exposureMin = self._nodeExposure.GetMin()
        self._exposureMax = self._nodeExposure.GetMax()

        # Gain Node
        self._nodeGainAuto = PySpin.CEnumerationPtr(self._nodemap.GetNode('GainAuto'))
        if not PySpin.IsAvailable(self._nodeGainAuto) or not PySpin.IsWritable(self._nodeGainAuto):
            self.print_retrieve_node_failure('node', 'GainAuto')
            return False

        self._gainAutoOff = self._nodeGainAuto.GetEntryByName('Off')
        if not PySpin.IsAvailable(self._gainAutoOff) or not PySpin.IsReadable(self._gainAutoOff):
            self.print_retrieve_node_failure('entry', 'GainAuto Off')
            return False

        if PySpin.IsAvailable(self._nodeGainAuto) and PySpin.IsWritable(self._nodeGainAuto):
            self._gainAutoOn = self._nodeGainAuto.GetEntryByName('Continuous')

        self._nodeGain = PySpin.CFloatPtr(self._nodemap.GetNode('Gain'))
        self._gainMin = self._nodeGain.GetMin()
        self._gainMax = self._nodeGain.GetMax()

        self.nodeGammaEnable = PySpin.CBooleanPtr(self._nodemap.GetNode('GammaEnable'))
        if not PySpin.IsAvailable(self.nodeGammaEnable) or not PySpin.IsWritable(self.nodeGammaEnable):
            self.print_retrieve_node_failure('node', 'GammaEnable')
            return False

        nodeAcquisitionFrameRate = PySpin.CFloatPtr(self._nodemap.GetNode('AcquisitionFrameRate'))
        if not PySpin.IsAvailable(nodeAcquisitionFrameRate) or not PySpin.IsWritable(nodeAcquisitionFrameRate):
            print("Unable to set acquisition Frame Rate (enum retrieval). Aborting...")
            return False
        else:
            nodeAcquisitionFrameRate.SetValue(5)

    def setAcquisitMode(self, mode=0):
        if mode==0:
            #Single Frame mode
            self.nodeAcquisitionMode.SetIntValue(self.nodeAcquisitionSingle)
        elif mode==1:
            # Continuous mode
            self.nodeAcquisitionMode.SetIntValue(self.nodeAcquisitionContinuous)
            #self.nodeAcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

        print(self.nodeAcquisitionMode.GetIntValue())
    
    def setFPS(self, fps):
        nodeAcquisitionFrameRate = PySpin.CFloatPtr(self._nodemap.GetNode('AcquisitionFrameRate'))
        if not PySpin.IsAvailable(nodeAcquisitionFrameRate) or not PySpin.IsWritable(nodeAcquisitionFrameRate):
            print("Unable to set acquisition Frame Rate (enum retrieval). Aborting...")
            return False
        else:
            nodeAcquisitionFrameRate.SetValue(fps)
    
    def getFPS(self):
        nodeAcquisitionFrameRate = PySpin.CFloatPtr(self._nodemap.GetNode('AcquisitionFrameRate'))
        return nodeAcquisitionFrameRate.GetValue()

    def setPixel(self):
        # Set the pixel format.
        if self._isMonochrome:
            # Enable Mono8 mode.
            if self._is16bits:
                self.nodePixelFormat.SetIntValue(self.pixelFormatMono16)
            else:
                self.nodePixelFormat.SetIntValue(self.pixelFormatMono8)
        # else:
        #     # Enable RGB8 mode.
        #     if self._is16bits:
        #         self.nodePixelFormat.SetIntValue(self.pixelFormatRGB16)
        #     else:
        #         self.nodePixelFormat.SetIntValue(self.pixelFormatRGB8)

    def setSize(self):
        self.width = self.nodeWidth.GetMax()
        self.nodeWidth.SetValue(self.width)

        self.height = self.nodeHeight.GetMax()
        self.nodeHeight.SetValue(self.height)

    def setCamAutoProperty(self, switch=True):
        # [Current Support] Gain, Exposure time
        # In order to manual set value, turn off auto first
        if switch:
            if PySpin.IsAvailable(self._exposureAutoOn) and PySpin.IsReadable(self._exposureAutoOn):
                self._nodeExposureAuto.SetIntValue(self._exposureAutoOn.GetValue())
                print('Turning automatic exposure back on...')

            if PySpin.IsAvailable(self._gainAutoOn) and PySpin.IsReadable(self._gainAutoOn):
                self._nodeGainAuto.SetIntValue(self._gainAutoOn.GetValue())
                print('Turning automatic gain mode back on...\n')

            self.nodeGammaEnable.SetValue(True)
        else:
            self._nodeExposureAuto.SetIntValue(self._exposureAutoOff.GetValue())
            print('Automatic exposure disabled...')

            self._nodeGainAuto.SetIntValue(self._gainAutoOff.GetValue())
            print('Automatic gain disabled...')

            #self.nodeGammaEnable.SetValue(False)
            #print('Gamma disabled...')
    
    def getSize(self):
        width = self.width
        height = self.height
        return width, height

    def getResolution(self):
        width,height = self.getSize()
        return (width,height)

    def setGain(self, gain_to_set):

        if float(gain_to_set) < self._gainMin or float(gain_to_set) > self._gainMax:
            print("[WARNING]: Gain value should be within {} to {}.(Input:{}) Set to half."
                  .format(self._gainMin, self._gainMax, float(gain_to_set)))
            self._nodeGain.SetValue(math.floor(self._gainMax + self._gainMin))
            print("Gain: {}".format(self._nodeGain.GetValue()))
        else:
            self._nodeGain.SetValue(float(gain_to_set))
            print("Gain: {}".format(self._nodeGain.GetValue()))

    def setExposure(self, exposure_to_set):

        exposure_to_set = float(exposure_to_set)
        if exposure_to_set < self._exposureMin or exposure_to_set > self._exposureMax:
            print("[WARNING]: Gain value should be within {} to {}.(Input:{}) Set to half."
                  .format(self._exposureMin, self._exposureMax, exposure_to_set))
            self._nodeExposure.SetValue(math.floor(self._exposureMax + self._exposureMin))
            print("Exposure: {}".format(self._nodeExposure.GetValue()))
        else:
            self._nodeExposure.SetValue(exposure_to_set)
            print("Exposure: {}".format(self._nodeExposure.GetValue()))

    def grabFrame(self):
        self.setAcquisitMode(0)
        self._camera.BeginAcquisition()
        cameraBuff = self._camera.GetNextImage()
        if cameraBuff.IsIncomplete():
            return False, None

        cameraImg = cameraBuff.GetData().reshape(self.height, self.width)
        image = cameraImg.copy()
        cameraBuff.Release()
        self._camera.EndAcquisition()

        if self._isMonochrome:
            return True, image
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BayerRG2BGR)
            return True, image_rgb

    def grabFrameCont(self):
        cameraBuff = self._camera.GetNextImage()
        if cameraBuff.IsIncomplete():
            return False, None

        cameraImg = cameraBuff.GetData().reshape(self.height, self.width)
        image = cameraImg.copy()
        cameraBuff.Release()
        #self._camera.EndAcquisition()

        if self._isMonochrome:
            return True, image
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BayerRG2BGR)
            return True, image_rgb

    def beginAcquisit(self, switch=True):
        if switch:
            self._camera.BeginAcquisition()
        else:
            self._camera.EndAcquisition()

    def getNextImage(self):
        return self._camera.GetNextImage()

    def setupHDR(self):
        if PySpin.IsAvailable(self._exposureAutoOn) and PySpin.IsReadable(self._exposureAutoOn):
            self._nodeExposureAuto.SetIntValue(self._exposureAutoOn.GetValue())
            print('Turning automatic exposure back on...')

        time.sleep(0.5)
        series = [2 ** (-2), 2 ** (-1), 1, 2 ** 1, 2 ** 2]
        midExposure = self._nodeExposure.GetValue()
        print("midExposure: ", midExposure)
        self.exposureHDRList = [midExposure * x for x in series]
        if self.exposureHDRList[0] < self._exposureMin:
            self.exposureHDRList[0] = self._exposureMin
        if self.exposureHDRList[-1] > self._exposureMax:
            self.exposureHDRList[-1] = self._exposureMax

        print("HDR Exposure List: ", self.exposureHDRList)

        self._nodeExposureAuto.SetIntValue(self._exposureAutoOff.GetValue())
        print('Automatic exposure disabled...')

    def captureHDR(self):
        if not hasattr(self, 'exposureHDRList'):
            print("[ERROR]: Need to setup HDR Exposure list first!!!")
            return 0
        imgs = np.zeros((len(self.exposureHDRList), self.height, self.width))
        for index, x in enumerate(self.exposureHDRList):
            self.setExposure(x)
            flag, tmp = self.grabFrame()
            if flag:
                imgs[index, ...] = tmp
            else:
                print("[WARNING]: Invalid Capture!!!")

        return imgs

    def release(self):
        # Turn auto gain and exposure back on in order to return the camera to tis default state
        self.setCamAutoProperty(True)
        if self._camera.IsStreaming():
            self._camera.EndAcquisition()
        self._camera.DeInit()
        #del self._camera
        # self._cameraList.Clear()
        # self._system.ReleaseInstance()

    def __exit__(self):
        self.release()
