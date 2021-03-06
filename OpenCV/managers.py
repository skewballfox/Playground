import cv2
import numpy as np
import time

class CaptureManager(object):
    """
    implementation of CaptureManager. Although it relies on
    cv2.VideoCapture, we could make other implementations that do not use OpenCV for input.
    For example, we could make a subclass that is instantiated with a socket connection, whose
    byte stream could be parsed as a stream of images. Also, we could make a subclass that
    uses a third-party camera library with different hardware support than what OpenCV
    provides.

    """
    def __init__(self, capture, previewWindowManager = None,
        shouldMirrorPreview=False):
        """
        """
        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview
        self._capture = capture
        self._channel = 0
        self._enteredFrame = False
        self._imageFilename = None
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None
        self._startTime = None
        self._framesElapsed = 0
        self._fpsEstimate = None

    @property
    def channel(self):
        """
        """
        return self._channel
    @channel.setter
    def channel(self,value):
        if self._channel != value:
            self._channel=value
            self._frame = None

    @property
    def frame(self):
        """
        """
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve(
                self._frame, self.channel)
        return self._frame
            )

     @property
     def isWritingImage(self) -> bool:
         """check to confirm that"""
         return self_imageFilename is not None

     @property isWritingVideo(self) --> bool:
         """
         """
         return self._videoFilename is not None

     def enterFrame(self):
         """ Capture the next frame, if any"""

         # but first, check that any previous frame was exited.
         assert not self._enteredFrame, \
            'previous enterFrame() had no matching exitFrame()'
         if self._capture is not None:
             self._enteredFrame = self._capture.grab()

     def exitFrame(self):
         """Draw to the window. Write to files. Release the frame"""

         # check whether any grabbed frame is retrievable
         # the getter may retrieve and cache the frame
         if self.frame is None:
             self._enteredFrame=False
             return

         # Update the FPS estimate and related variables
         if self._framesElapsed == 0:
             self._startTime = time.time()

         else:
             timeElapsed=time.time() - self._startTime
             self._fpsEstimate = self._framesElapsed / timeElapsed
         self._framesElapsed += 1

         # Draw to the window, if any.
         if self.previewWindowManager is not None:
             if self.shouldMirrorPreview:
                 mirroredFrame = numpy.fliplr(self._frame)
                 self.previewWindowManager.show(mirroredFrame)
             else:
                 self.previewWindowManager.show(self._frame)

         # Write to the image file, if any
         if self.isWritingImage:
             cv2.imwrite(self._imageFilename, self._frame)
             self._imageFilename = None

         # Write to the video file, if any.
         self._writeVideoFrame()

         # Release the frame.
         self._frame = None
         self._enteredFrame = False
         

    def writeImage(self, filename):
        """Write the next exited frame to an image file."""
        self._imageFilename = filename

    def startWritingVideo(self, filename,
                          encoding= cv2.VideoWriter_fourcc('M','J','P','G')):
        """Start writing exited frames to a video file."""
        self._videoFilename = filename
        self._videoEncoding = encoding

    def stopWritingVideo(self):
        """stop writing exited frames to a video file."""
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None


