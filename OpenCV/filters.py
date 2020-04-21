import cv2
import numpy as np
import utils


def strokeEdges(source_image, destination_image, blur_kernel_size=7, edgeKsize=5):
    """
    parameters
    source_image
    """
    # NOTE: kernel is just a set of weights that determines how each output pixel
    # is calculated from a neighborhood of input pixels
    if blur_kernel_size >= 3:
        blurred_source = cv2.medianBlur(source_image, blur_kernel_size)
        gray_source = cv2.cvtColor(blurred_source, cv2.COLOR_BGR2GRAY)
    else:
        gray_source = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(gray_source, cv2.CV_8U, gray_source, ksize=edgeKsize)
    normalizedInverseAlpha = (1.0 / 255) * (255 - gray_source)
    channels = cv2.split(source_image)
    for channel in channels:  # this might something that needs to be rewritten later
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, destination_image)


class VConvolutionFilter(object):
    """A filter that applies a convolution to V (or all of BGR)."""

    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, source_image, destination_image):
        """Apply the filter with a BGR or gray source/destination."""
        cv2.filter2D(source_image, -1, self._kernel, destination_image)


class SharpenFIlter(VConvolutionFilter):
    """A sharpen filter with a 1-pixel radius."""

    def __init__(self):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class BlurFilter(VConvolutionFilter):
    """A blur filter with a 2-pixel radius."""

    def __init__(self):
        kernel = np.array(
            [
                [0.04, 0.04, 0.04, 0.04, 0.04],
                [0.04, 0.04, 0.04, 0.04, 0.04],
                [0.04, 0.04, 0.04, 0.04, 0.04],
                [0.04, 0.04, 0.04, 0.04, 0.04],
                [0.04, 0.04, 0.04, 0.04, 0.04],
            ]
        )
        VConvolutionFilter.__init__(self, kernel)


class EmbossFilter(VConvolutionFilter):
    """An emboss filter with a 1-pixel radius."""

    def __init__(self):
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        VConvolutionFilter.__init__(self, kernel)
