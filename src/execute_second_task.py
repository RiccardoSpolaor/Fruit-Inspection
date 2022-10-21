import argparse
import cv2 as cv
import json
import numpy as np
import os

from typing import List, Tuple

from utils.colour import *
from utils.colour_threshold import *
from utils.general import *
from utils.threshold import *


def detect_russet(image: np.ndarray, h_means: List[np.ndarray], h_inv_covs: List[np.ndarray],
                  roi_means: List[List[np.ndarray]], roi_inv_covs: List[List[np.ndarray]],
                  roi_thresholds: List[List[int]], class_threshold: float = 3, tweak_factor: float = .4,
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
    # Filter the image by bilateral filter
    f_img = cv.bilateralFilter(image, d=5, sigmaColor=75, sigmaSpace=75)

    # Convert the image to gray-scale and median blur it by a 5 x 5 kernel
    f_gray = cv.medianBlur(cv.cvtColor(image, cv.COLOR_BGR2GRAY), 5)

    # Get the fruit mask through Tweaked Otsu's algorithm
    mask = get_fruit_segmentation_mask(f_gray, ThresholdingMethod.TWEAKED_OTSU, tweak_factor=tweak_factor)

    # Apply the mask to the filtered colour image
    m_image = cv.cvtColor(mask, cv.COLOR_GRAY2BGR) + f_img

    # Turn BGR image to LAB
    m_lab_image = ColourSpace('LAB').bgr_to_colour_space(m_image)
    channels = (1, 2)

    # Get the fruit class
    fruit_class = get_fruit_class(m_lab_image, h_means, h_inv_covs, channels=channels, threshold=class_threshold,
                                  display_image=image if verbose else None)

    if verbose:
        print(f'Class of fruit = {fruit_class}')

    # Erode the mask to get rid of artefacts to the bound of the fruit after the russet detection is applied
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    eroded_mask = cv.erode(mask, element)

    # Initialize the mask of the russet (ROI) as an array of 0s
    roi_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Get the mask of each possible russet of the fruit and apply a bitwise
    # OR between it and the previous ROI mask
    for m, c, t in zip(roi_means[fruit_class], roi_inv_covs[fruit_class], roi_thresholds[fruit_class]):
        roi_mask += get_mahalanobis_distance_segmented_image(m_lab_image, m, c, t, channels)

    # Remove artifacts from the ROI mask
    roi_mask = roi_mask & eroded_mask

    # Apply median blur to de-noise the mask and smooth it
    roi_mask = cv.medianBlur(roi_mask, 5)

    # Apply Closing operation to close small gaps in the ROI mask
    roi_mask = cv.morphologyEx(roi_mask, cv.MORPH_CLOSE, element)

    # Perform a connected components labeling to detect defects
    retval, labels, stats, centroids = cv.connectedComponentsWithStats(roi_mask)

    if verbose:
        print(f'Detected {retval - 1} defect{"" if retval - 1 == 1 else "s"} for image {image_name}.')

        # Get highlighted russet on the fruit
        highlighted_roi = get_highlighted_roi_by_mask(image, roi_mask)

        circled_russets = np.copy(image)

        for i in range(1, retval):
            s = stats[i]
            # Draw a red ellipse around the russet
            cv.ellipse(circled_russets, center=tuple(int(c) for c in centroids[i]),
                       axes=(s[cv.CC_STAT_WIDTH] // 2 + 10, s[cv.CC_STAT_HEIGHT] // 2 + 10),
                       angle=0, startAngle=0, endAngle=360, color=(0, 0, 255), thickness=3)

        plot_image_grid([highlighted_roi, circled_russets], ['Detected russets ROI', 'Detected russets areas'],
                        f'Russets of the fruit in image {image_name}')
    return retval - 1, stats[1:], centroids[1:]


def _main():
    parser = argparse.ArgumentParser(description='Script for applying russet detection on a fruit.')

    parser.add_argument('fruit-image-path', metavar='Fruit image path', type=str,
                        help='The path of the colour image of the fruit.')

    parser.add_argument('image-name', metavar='Image name', type=str, help='The name of the image.', default='',
                        nargs='?')

    parser.add_argument('--config-file-path', '-cf', type=str, help='The path of the configuration file.',
                        default=os.path.join(os.path.dirname(__file__), f'config/config.json'), nargs='?',
                        required=False)

    parser.add_argument('--data-folder-path', '-d', type=str, help='The path of the data folder.',
                        default=os.path.join(os.path.dirname(__file__), f'data'), nargs='?', required=False)

    parser.add_argument('--tweak-factor', '-tf', type=float, default=.4, nargs='?',
                        help='Tweak factor for obtaining the binary mask.', required=False)

    parser.add_argument('--class-threshold', '-ct', type=float, default=3, nargs='?',
                        help='Distance threshold to compute the class of the fruit.', required=False)

    parser.add_argument('--no-verbose', '-nv', action='store_true', help='Skip the visualization of the results.')

    # Initialize parser
    arguments = parser.parse_args()

    # Read colour image
    fruit_image_path = vars(arguments)['fruit-image-path']
    colour_image = cv.imread(fruit_image_path)

    image_name = vars(arguments)['image-name']

    config_file_path = arguments.config_file_path
    data_folder_path = arguments.data_folder_path
    tweak_factor = arguments.tweak_factor
    class_threshold = arguments.class_threshold
    verbose = not arguments.no_verbose

    # Get config file
    with open(config_file_path, 'r') as j:
        config_dictionary = json.load(j)

    # Get means, inverse covariance matrices and other information from the config file
    healthy_fruit_means = config_dictionary['healthy_fruit_means']
    healthy_fruit_inv_covs = config_dictionary['healthy_fruit_inv_covs']
    roi_means = config_dictionary['roi_means']
    roi_inv_covs = config_dictionary['roi_inv_covs']
    roi_thresholds = config_dictionary['roi_thresholds']
    roi_related_fruit = config_dictionary['roi_related_fruit']

    healthy_fruit_means = [np.load(os.path.join(data_folder_path, n)) for n in healthy_fruit_means]
    healthy_fruit_inv_covs = [np.load(os.path.join(data_folder_path, n)) for n in healthy_fruit_inv_covs]
    roi_means = [np.load(os.path.join(data_folder_path, n)) for n in roi_means]
    roi_inv_covs = [np.load(os.path.join(data_folder_path, n)) for n in roi_inv_covs]

    # Assign roi to each fruit
    roi_means_sorted = [[] for _ in range(len(healthy_fruit_means))]
    roi_inv_cov_sorted = [[] for _ in range(len(healthy_fruit_means))]
    roi_thresholds_sorted = [[] for _ in range(len(healthy_fruit_means))]

    for m, c, t, r in zip(roi_means, roi_inv_covs, roi_thresholds, roi_related_fruit):
        roi_means_sorted[r].append(m)
        roi_inv_cov_sorted[r].append(c)
        roi_thresholds_sorted[r].append(t)

    # Apply russet detection
    detect_russet(colour_image, healthy_fruit_means, healthy_fruit_inv_covs, roi_means_sorted, roi_inv_cov_sorted,
                  roi_thresholds_sorted, class_threshold=class_threshold, tweak_factor=tweak_factor,
                  image_name=image_name, verbose=verbose)


if __name__ == '__main__':
    _main()
