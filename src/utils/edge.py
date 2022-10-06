"""
Module about edge detection functions.
"""
import cv2 as _cv
import numpy as _np
import math as _math

from .general import get_highlighted_roi_by_mask


def apply_gaussian_blur_and_canny(image: _np.ndarray, sigma: float, threshold_1: float,
                                  threshold_2: float) -> _np.ndarray:
    """
    Function that blurs the image through Gaussian Blur and then applies Canny's edge detection algorithm on it.

    Parameters
    ----------
    image: ndarray
        Image to blur and apply edge detection on
    sigma: float
        Sigma value of the Gaussian Blur function
    threshold_1: float
        First threshold used for the Hysteresis step. If a pixel gradient value is below it, then it is rejected
    threshold_2: float
        Second threshold used for the Hysteresis step. If a pixel gradient is higher than it, the pixel is accepted as
        an edge. If the pixel gradient is between the two thresholds, then it will be accepted only if it is connected
        to a pixel that is above `threshold_2`

    Returns
    -------
    edge_mask: ndarray
        Grayscale image showing the edges of the input image
    """
    # Compute the kernel size through the rule-of-thumb
    k = _math.ceil(3 * sigma)
    kernel_size = (2 * k + 1, 2 * k + 1)

    # Apply Gaussian Blur on the image
    blur_image = _cv.GaussianBlur(image, kernel_size, sigma)

    # Get the edges of the image through Canny's Algorithm
    return _cv.Canny(blur_image, threshold_1, threshold_2)


def get_highlighted_edges_on_image(image: _np.ndarray, edge_mask: _np.ndarray, size: int = 3,
                                   highlight_channel: str = 'red') -> _np.ndarray:
    """
    Function that highlights edges on an image.

    Parameters
    ----------
    image: ndarray
        Image over which the edges are highlighted
    edge_mask: ndarray
        Grayscale image showing the edges of the input image
    size: int, optional
        The size of the edges to highlight (default: 3)
    highlight_channel: str, optional
        Colour of the highlighted edges: 'blue'; 'green' or 'red' (default: 'red')

    Returns
    -------
    highlighted_edges: ndarray
        Highlighted edges over the input image
    """
    element = _cv.getStructuringElement(_cv.MORPH_RECT, (size, size))
    dilated_edge_mask = _cv.dilate(edge_mask, element)
    return get_highlighted_roi_by_mask(image, dilated_edge_mask, highlight_channel)
