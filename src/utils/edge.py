import cv2 as cv
import numpy as np
import math

from .graphics import get_highlighted_roi_by_mask


def apply_gaussian_blur_and_canny(image: np.ndarray, sigma: float, threshold_1: float,
                                  threshold_2: float) -> np.ndarray:
    # Compute the kernel size through the rule-of-thumb
    k = math.ceil(3 * sigma)
    kernel_size = (2 * k + 1, 2 * k + 1)

    # Apply Gaussian Blur on the image
    blur_image = cv.GaussianBlur(image, kernel_size, sigma)

    # Get the edges of the image through Canny's Algorithm
    return cv.Canny(blur_image, threshold_1, threshold_2)


def get_highlighted_edges_on_image(image: np.ndarray, edge_mask: np.ndarray,
                                   size: int = 3, highlight_channel: str = 'red') -> np.ndarray:
    element = cv.getStructuringElement(cv.MORPH_RECT, (size, size))
    dilated_edge_mask = cv.dilate(edge_mask, element)
    return get_highlighted_roi_by_mask(image, dilated_edge_mask, highlight_channel)