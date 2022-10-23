import argparse
import cv2 as cv
import numpy as np
from typing import Tuple

from utils.edge import *
from utils.general import *
from utils.threshold import *


def detect_defects(colour_image: np.ndarray, nir_image: np.ndarray, image_name: str = '', tweak_factor: float = .3,
                   sigma: float = 1., threshold_1: int = 60, threshold_2: int = 130,
                   verbose: bool = True) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Function that executes the task of detecting the defects of a fruit.

    It firstly masks the fruit, then it looks for the defects.

    If the task is run in `verbose` mode, the visualization of the defect regions of the fruit is plotted.

    Parameters
    ----------
    colour_image: np.ndarray
        Colour image of the fruit whose defects have to be detected
    nir_image: np.ndarray
        Near Infra-Red image of the same fruit represented in `colour_image`
    image_name: str, optional
        Optional name of the image to visualize during the plotting operations
    tweak_factor: float, optional
        Tweak factor to apply to the "Tweaked Otsu's Algorithm" in order to obtain the binary segmentation mask
        (default: 0.3)
    sigma: float, optional
        Value of sigma to apply to the Gaussian Blur operation before the use of Canny's algorithm (default: 1)
    threshold_1: int, optional
        Value of the first threshold that is used in Canny's algorithm (default: 60)
    threshold_2: int, optional
        Value of the second threshold that is used in Canny's algorithm (default: 120)
    verbose: bool, optional
        Whether to run the function in verbose mode or not (default: True)

    Returns
    -------
    retval: int
        Number of defect regions found in the fruit
    stats: np.ndarray
        Array of statistics about each defect region:
            - The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction;
            - The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction;
            - The horizontal size of the bounding box;
            - The vertical size of the bounding box;
            - The total area (in pixels) of the defect.
    centroids: np.ndarray
        Array of centroid points about each defect region.
    """
    # Filter the NIR image by median blur
    f_nir_image = cv.medianBlur(nir_image, 5)

    # Get the fruit mask through Tweaked Otsu's algorithm
    mask = get_fruit_segmentation_mask(f_nir_image, ThresholdingMethod.TWEAKED_OTSU, tweak_factor=tweak_factor)

    # Apply the mask to the filtered NIR image
    m_nir_image = apply_mask_to_image(f_nir_image, mask)

    # Get the edge mask through Gaussian Blur and Canny's method
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
    parser = argparse.ArgumentParser(description='Script for applying defect detection on a fruit.')

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

    parser.add_argument('--threshold-1', '-t1', type=int, default=60, nargs='?',
                        help="First threshold that is used in Canny's algorithm hysteresis process.", required=False)

    parser.add_argument('--threshold-2', '-t2', type=int, default=130, nargs='?',
                        help="Second threshold that is used in Canny's algorithm hysteresis process.", required=False)

    parser.add_argument('--no-verbose', '-nv', action='store_true', help='Skip the visualization of the results.')

    # Initialize parser
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
    verbose = not arguments.no_verbose

    detect_defects(colour_image, nir_image, image_name=image_name, tweak_factor=tweak_factor, sigma=sigma,
                   threshold_1=threshold_1, threshold_2=threshold_2, verbose=verbose)


if __name__ == '__main__':
    _main()
