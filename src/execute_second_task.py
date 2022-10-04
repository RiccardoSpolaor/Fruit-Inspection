import argparse
import json
import os

from utils import *


def detect_russet(image: np.ndarray, h_means: List[np.ndarray],
                  h_inv_covs: List[np.ndarray],
                  roi_means: List[List[np.ndarray]],
                  roi_inv_covs: List[List[np.ndarray]],
                  roi_thresholds: List[List[int]],
                  tweak_factor, image_name: str = None,
                  verbose: bool = True) -> None:
    # Apply median filter to the image
    f_img = cv.medianBlur(image, 5)

    # Convert the image to gray-scale
    gray_img = cv.cvtColor(f_img, cv.COLOR_BGR2GRAY)

    # Get the mask of the fruit
    mask = get_fruit_mask(gray_img, ThresholdingMethod.TWEAKED_OTSU,
                          tweak_factor=tweak_factor)

    # Mask the filtered image
    m_image = cv.cvtColor(mask, cv.COLOR_GRAY2BGR) + f_img

    # Turn BGR image to LAB
    m_lab_image = ColourSpace('LAB').bgr_to_color_space(m_image)
    channels = (1, 2)

    # Get fruit class
    fruit_class = get_fruit_class(m_lab_image, h_means, h_inv_covs, channels,
                                  display_image=image if verbose else None)

    if verbose:
        print(f'Class of fruit = {fruit_class}')

    # Erode the mask
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    eroded_mask = cv.erode(mask, element)

    # Initialize the mask of the russet (ROI) as an array of 0s
    roi_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Get the mask of each possible russet of the fruit and apply a bitwise
    # OR between it and the previous ROI mask
    for m, c, t in zip(roi_means[fruit_class], roi_inv_covs[fruit_class],
                       roi_thresholds[fruit_class]):
        roi_mask += get_mahalanobis_distance_img(m_lab_image, m, c,
                                                 t, channels)

    # Apply a bitwise AND between the eroded mask and the ROI mask
    roi_mask = roi_mask & eroded_mask

    # Apply median blur to de-noise the mask and smooth it
    roi_mask = cv.medianBlur(roi_mask, 5)

    # Apply close operation to close small gaps in the mask
    roi_mask = cv.morphologyEx(roi_mask, cv.MORPH_CLOSE, element)

    # Perform a connected components labeling to detect defects
    retval, labels, stats, centroids = cv.connectedComponentsWithStats(
        roi_mask)

    if verbose:
        print(f'Detected {retval - 1} russet areas.')

        # Get highlighted russet on the fruit
        highlighted_roi = get_highlighted_roi_by_mask(image, roi_mask)

        circled_russets = np.copy(image)

        for i in range(1, retval):
            s = stats[i]
            # Draw a red ellipse around the defect
            cv.ellipse(circled_russets,
                       center=tuple(int(c) for c in centroids[i]),
                       axes=(s[cv.CC_STAT_WIDTH] // 2 + 10,
                             s[cv.CC_STAT_HEIGHT] // 2 + 10),
                       angle=0, startAngle=0, endAngle=360,
                       color=(0, 0, 255), thickness=3)

        if image_name is None:
            image_name = ''

        plot_image_grid([highlighted_roi, circled_russets],
                        ['Detected russets ROI', 'Detected russets areas'],
                        f'Russets of the fruit in image {image_name}')


if __name__ == '__main__':
    # Image names
    nir_names, colour_names = [[f'C{j}_00000{i}.png' for i in range(1, 4)] for
                               j in [0, 1]]

    parser = argparse.ArgumentParser(
        description='Script for applying russet detection on a fruit.')

    parser.add_argument('fruit-image-path', metavar='Fruit image path',
                        type=str, help='The path of the colour image of the '
                                       'fruit.')

    parser.add_argument('image-name', metavar='Image name',
                        type=str, help='The name of the image.',
                        default=None, nargs='?')

    parser.add_argument('--config-file-path', '-cf', type=str,
                        help='The path of the configuration file.',
                        default=os.path.join(os.path.dirname(__file__),
                                             f'config/config.json'),
                        nargs='?', required=False)

    parser.add_argument('--data-folder-path', '-d', type=str,
                        help='The path of the data folder.',
                        default=os.path.join(os.path.dirname(__file__),
                                             f'data'),
                        nargs='?', required=False)

    parser.add_argument('--tweak-factor', '-tf', type=float, default=.4,
                        nargs='?', help="Tweak factor for obtaining the "
                                        "binary mask.",
                        required=False)

    parser.add_argument('--no-verbose', '-nv', action='store_true',
                        help='Skip the visualization of the results.')

    arguments = parser.parse_args()

    fruit_image_path = vars(arguments)['fruit-image-path']
    colour_image = cv.imread(fruit_image_path)

    image_name = vars(arguments)['image-name']

    config_file_path = arguments.config_file_path
    data_folder_path = arguments.data_folder_path
    tweak_factor = arguments.tweak_factor
    verbose = not arguments.no_verbose

    with open(config_file_path, 'r') as j:
        config_dictionary = json.load(j)

    healthy_fruit_means = config_dictionary['healthy_fruit_means']
    healthy_fruit_inv_covs = config_dictionary['healthy_fruit_inv_covs']
    roi_means = config_dictionary['roi_means']
    roi_inv_covs = config_dictionary['roi_inv_covs']
    roi_thresholds = config_dictionary['roi_thresholds']
    roi_related_fruit = config_dictionary['roi_related_fruit']

    healthy_fruit_means = [np.load(os.path.join(data_folder_path, n))
                           for n in healthy_fruit_means]
    healthy_fruit_inv_covs = [np.load(os.path.join(data_folder_path, n))
                              for n in healthy_fruit_inv_covs]
    roi_means = [np.load(os.path.join(data_folder_path, n))
                 for n in roi_means]
    roi_inv_covs = [np.load(os.path.join(data_folder_path, n))
                    for n in roi_inv_covs]

    roi_means_sorted = [[] for _ in range(len(healthy_fruit_means))]
    roi_inv_cov_sorted = [[] for _ in range(len(healthy_fruit_means))]
    roi_thresholds_sorted = [[] for _ in range(len(healthy_fruit_means))]

    for m, c, t, r in zip(roi_means, roi_inv_covs, roi_thresholds,
                          roi_related_fruit):
        roi_means_sorted[r].append(m)
        roi_inv_cov_sorted[r].append(c)
        roi_thresholds_sorted[r].append(t)

    detect_russet(colour_image, healthy_fruit_means, healthy_fruit_inv_covs,
                  roi_means_sorted, roi_inv_cov_sorted, roi_thresholds_sorted,
                  tweak_factor, image_name = None)
