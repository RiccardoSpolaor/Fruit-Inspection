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

def analyze_fruit(colour_image: np.ndarray, nir_image: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray,
                  roi_threshold: int = 10,  tweak_factor: float = .4,
                  sigma: float = 1., threshold_1: int = 50, threshold_2: int = 85,
                  image_name: str = '', verbose: bool = True) -> Tuple[int, np.ndarray, np.ndarray]:

    # Filter the image by median blur
    f_img = cv.medianBlur(nir_image, 7)

    # Get the fruit mask through Tweaked Otsu's algorithm
    mask = get_fruit_segmentation_mask(f_img, ThresholdingMethod.TWEAKED_OTSU, tweak_factor=tweak_factor)

    # Perform two openings to clean up the mask 
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
    parser = argparse.ArgumentParser(description='Script for defect location on a fruit.')

    parser.add_argument('fruit-image-path', metavar='Fruit image path', type=str,
                        help='The path of the colour image of the fruit.')

    parser.add_argument('fruit-nir-image-path', metavar='Fruit image path', type=str,
                        help='The path of the Near Infra-Red image of the fruit.')

    parser.add_argument('image-name', metavar='Image name', type=str, help='The name of the image.', default='',
                        nargs='?')

    parser.add_argument('--tweak-factor', '-tf', type=float, default=.3, nargs='?',
                        help='Tweak factor for obtaining the binary mask.', required=False)

    parser.add_argument('--sigma', '-s', type=float, default=1., nargs='?',
                        help="Sigma to apply to the Gaussian Blur operation before Canny's algorithm",
                        required=False)

    parser.add_argument('--threshold-1', '-t1', type=int, default=50, nargs='?',
                        help="First threshold that is used in Canny's algorithm.", required=False)

    parser.add_argument('--threshold-2', '-t2', type=int, default=85, nargs='?',
                        help="Second threshold that is used in Canny's algorithm.", required=False)

    parser.add_argument('--no-verbose', '-nv', action='store_true', help='Skip the visualization of the results.')

    # TODO aggiungere il np.array (chiedere a riccardo come fare)

    parser.add_argument('--mean-file-path', '-meanf', type=str, help='The path of the mean.npy file.',
                    default=os.path.join(os.path.dirname(__file__), f'data/mean_final_challenge.npy'), nargs='?',
                    required=False)

    parser.add_argument('--cov-file-path', '-covf', type=str, help='The path of the inv_cov.npy file.',
                default=os.path.join(os.path.dirname(__file__), f'data/inv_cov_final_challenge.npy'), nargs='?',
                required=False)


    parser.add_argument('--roi-threshold', '-ct', type=int, default=10, nargs='?',
                        help='Distance threshold to compute the class of the fruit.', required=False)

    parser.add_argument('--no-verbose', '-nv', action='store_true', help='Skip the visualization of the results.')

    arguments = parser.parse_args()

    # Read colour image
    fruit_image_path = vars(arguments)['fruit-image-path']
    colour_image = cv.imread(fruit_image_path)

    # Read NIR image
    fruit_nir_image_path = vars(arguments)['fruit-nir-image-path']
    nir_image = cv.imread(fruit_nir_image_path, cv.IMREAD_GRAYSCALE)

    image_name = vars(arguments)['image-name']

    tweak_factor = arguments.tweak_factor
    sigma = arguments.sigma
    threshold_1 = arguments.threshold_1
    threshold_2 = arguments.threshold_2
    roi_threshold = arguments.roi_threshold
    verbose = not arguments.no_verbose


    mean = np.load(arguments.mean_file_path)
    inv_cov = np.load(arguments.inv_cov_file_path)

    analyze_fruit(colour_image, nir_image, mean, inv_cov, roi_threshold, tweak_factor, sigma, threshold_1, threshold_2, '', verbose)


if __name__ == '__main__':
    _main()
