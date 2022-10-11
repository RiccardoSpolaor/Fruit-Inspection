import argparse
import cv2 as cv
import json
import numpy as np
import os
import cv2 as cv
import math
import sys
from time import time

from typing import List, Tuple

# Utils libraries
from utils.edge import *
from utils.colour import *
from utils.colour_threshold import *
from utils.threshold import *
from utils.general import *

def final_challenge(colour_image: np.ndarray, nir_image: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray,
                  roi_threshold: int, class_threshold: float = 3, tweak_factor: float = .4,
                  sigma: float = 1., threshold_1: int = 50, threshold_2: int = 850,
                  image_name: str = '', verbose: bool = True) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Function that executes the task of detecting the russet regions of a fruit.

    It firstly detects the class of the fruit, then it looks for the russet regions that can be present according to the
    class of the fruit.

    If the task is run in `verbose` mode, then the procedure of the detection of the fruit class is plotted along with
    the visualization of the russet regions in the fruit.

    Parameters
    ----------
    image: np.ndarray
        Colour image of the fruit whose russet has to be detected
    h_means: List[np.ndarray]
        List of the mean LAB colour values of the healthy part of the fruits (one mean per fruit class)
    h_inv_covs: List[np.ndarray]
        List of inverse covariance matrices of the healthy fruit parts computed on the LAB colour space (one covariance
        matrix per fruit class)
    roi_means: List[List[np.ndarray]]
        List of list of mean LAB colour values of the russet regions of the fruits (one or multiple per fruit class)
    roi_inv_covs: List[List[np.ndarray]]
        List of list of inverse covariance matrices of the russet regions of the fruits computed on the LAB colour space
        (one or multiple per fruit class)
    roi_thresholds: List[List[int]]
        List of list of thresholds. Pixels of the colour image having a Mahalanobis distance greater than a certain
        thresholds are not considered part of the corresponding russet region (one or multiple per fruit class)
    class_threshold: float, optional
        Threshold to compute the fruit class according to the colour distance from its healthy part. Pixels of the
        colour image having a Mahalanobis distance greater than it are not considered part of the corresponding healthy
        fruit region (default: 3)
    image_name: str, optional
        Optional name of the image to visualize during the plotting operations
    tweak_factor: float, optional
        Tweak factor to apply to the "Tweaked Otsu's Algorithm" in order to obtain the binary segmentation mask
        (default: 0.4)
    verbose: bool, optional
        Whether to run the function in verbose mode or not (default: True)

    Returns
    -------
    retval: int
        Number of russet regions found in the fruit
    stats: np.ndarray
        Array of statistics about each russet region:
            - The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction;
            - The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction;
            - The horizontal size of the bounding box;
            - The vertical size of the bounding box;
            - The total area (in pixels) of the russet.
    centroids: np.ndarray
        Array of centroid points about each russet region.
    """
    # Filter the image by median blur
    f_img = cv.medianBlur(nir_image, 7)

    # Get the fruit mask through Tweaked Otsu's algorithm
    mask = get_fruit_segmentation_mask(f_img, ThresholdingMethod.TWEAKED_OTSU, tweak_factor=tweak_factor)

    # Perform two openings to clean the mask 
    se1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20,5))
    se2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,20))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, se1)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, se2)

    mask = get_largest_blob_in_mask(mask)

    # Apply the mask to original image 
    m_colour_image = apply_mask_to_image(colour_image, mask)

    # Apply medianBlur to colour image
    filtered_m_colour_image = cv.medianBlur(m_colour_image,5)

    # Turn BGR image to LAB and extract mask using Mahalanobis distance

    lab_image = ColourSpace('LAB').bgr_to_colour_space(filtered_m_colour_image)
    channels = (1, 2)

    roi_mask = get_mahalanobis_distance_segmented_image(lab_image, mean, inv_cov, roi_threshold, channels)

    # Apply Closing operation to close small gaps in the ROI mask
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    roi_mask = cv.morphologyEx(roi_mask, cv.MORPH_OPEN, element)
    roi_mask = cv.morphologyEx(roi_mask, cv.MORPH_CLOSE, element)

    m_nir_image = apply_mask_to_image(f_img, roi_mask)

    # Perform a connected components labeling to detect defects
    edge_mask = apply_gaussian_blur_and_canny(m_nir_image, sigma, threshold_1, threshold_2)

    # Erode the mask to get rid of the edges of the bound of the fruit
    erode_element = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    eroded_mask = cv.erode(mask, erode_element)

    # Remove background edges from the edge mask
    edge_mask = apply_mask_to_image(edge_mask, eroded_mask)

    # Apply Closing operation to fill the defects according to the edges and obtain the defect mask
    close_element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (50, 50))
    defect_mask = cv.morphologyEx(edge_mask, cv.MORPH_CLOSE, close_element)
    defect_mask = cv.medianBlur(defect_mask, 7)

    # Perform a connected components labeling to detect the defects
    retval, labels, stats, centroids = cv.connectedComponentsWithStats(defect_mask)

    if verbose:
        print(f'Detected {retval - 1} defect{"" if retval - 1 == 1 else "s"} for image {image_name}.')

        # Get highlighted defects on the fruit
        highlighted_roi = get_highlighted_roi_by_mask(colour_image, defect_mask, 'red')

        circled_defects = np.copy(colour_image)

        for i in range(1, retval):
            s = stats[i]
            # Draw a red ellipse around the defect
            cv.ellipse(circled_defects, center=tuple(int(c) for c in centroids[i]),
                       axes=(s[cv.CC_STAT_WIDTH] // 2 + 10, s[cv.CC_STAT_HEIGHT] // 2 + 10),
                       angle=0, startAngle=0, endAngle=360, color=(0, 0, 255), thickness=3)

        plot_image_grid([highlighted_roi, circled_defects],
                        ['Detected defects ROI', 'Detected defects areas'],
                        f'Defects of the fruit {image_name}')
    return retval - 1, stats[1:], centroids[1:]


def _main():

   # TODO


if __name__ == '__main__':
    _main()
