import argparse

from utils import *

_TWEAK_FACTOR = .3
_SIGMA = 1.
_THRESHOLD_1 = 60
_THRESHOLD_2 = 130


def detect_defects(colour_image: np.ndarray, nir_image: np.ndarray, image_name: str, verbose: bool = True) -> None:
    # Filter the NIR image by median blur
    f_nir_image = cv.medianBlur(nir_image, 5)

    # Get the fruit mask through Tweaked Otsu's algorithm
    mask = get_fruit_mask(f_nir_image, ThresholdingMethod.TWEAKED_OTSU, tweak_factor=_TWEAK_FACTOR)

    # Apply the mask to the filtered NIR image
    m_nir_image = mask + f_nir_image

    # Obtain edge mask through Gaussian Blur and Canny's method
    edge_mask = apply_gaussian_blur_and_canny(m_nir_image, _SIGMA, _THRESHOLD_1, _THRESHOLD_2)

    # Erode the mask to get rid of the edges of the bound of the fruit
    erode_element = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    eroded_mask = cv.erode(mask, erode_element)

    # Remove background edges from the edge mask
    edge_mask = edge_mask & eroded_mask

    # Apply Closing operation to fill the defects according to the edges and obtain the defect mask
    close_element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (50, 50))
    defect_mask = cv.morphologyEx(edge_mask, cv.MORPH_CLOSE, close_element)
    defect_mask = cv.medianBlur(defect_mask, 7)

    # Perform a connected components labeling to detect the defects
    retval, labels, stats, centroids = cv.connectedComponentsWithStats(defect_mask)

    if verbose:
        print(f'Detected {retval - 1} defect{"" if retval - 1 == 1 else "s"} for image {image_name}.')

        highlighted_roi = get_highlighted_roi_by_mask(colour_image, defect_mask, 'red')

        circled_defects = np.copy(colour_image)

        # Get stats of all connected components except the background (position: 0)
        for i in range(1, retval):
            s = stats[i]
            # Draw a red ellipse around the defect
            cv.ellipse(circled_defects, tuple(int(c) for c in centroids[i]),
                       (s[cv.CC_STAT_WIDTH] // 2 + 10, s[cv.CC_STAT_HEIGHT] // 2 + 10),
                       0, 0, 360, (0, 0, 255), 3)

        plot_image_grid([highlighted_roi, circled_defects],
                        ['Detected defects ROI', 'Detected defects areas'],
                        f'Defects of the fruit {image_name}')


if __name__ == '__main__':
    # Image names
    nir_names, colour_names = [[f'C{j}_00000{i}.png' for i in range(1, 4)] for j in [0, 1]]

    # Directory where the images are saved
    DIR = 'images/first task/'

    parser = argparse.ArgumentParser(description='Script for applying defect detection on a fruit.')

    parser.add_argument('fruit-index', metavar='fruit.index', type=int, choices=np.arange(1, 4),
                        help='The index of the fruit whose defect have to be detected.')

    arguments = parser.parse_args()

    fruit_index = vars(arguments)['fruit-index']

    nir_img = cv.imread(f'{DIR}{nir_names[fruit_index]}', cv.IMREAD_GRAYSCALE)
    colour_img = cv.imread(f'{DIR}{colour_names[fruit_index]}')

    detect_defects(colour_img, nir_img, colour_names[fruit_index])
