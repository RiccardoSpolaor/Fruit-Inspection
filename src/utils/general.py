import cv2
import numpy as np


def change_contrast_grayscale(image: np.array, alpha: float = 1.5, beta: float = 2.0) -> np.array:
    new_image = np.copy(image)

    # Set new intensity `alpha` * intensity of p + `beta` to each pixel p of `image`
    # and clip the result between 0 and 255
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            new_image[i, j] = np.clip(alpha * image[i, j] + beta, 0, 255)

    return new_image


class ColourSpace:
    _COLOR_SPACE_IDS = {
        'BGR': None,
        'HSV': cv2.COLOR_BGR2HSV_FULL,
        'HLS': cv2.COLOR_BGR2HLS_FULL,
        'LUV': cv2.COLOR_BGR2Luv,
        'LAB': cv2.COLOR_BGR2LAB,
        'YCrCb': cv2.COLOR_BGR2YCrCb
    }

    _COLOR_SPACE_CHANNEL_NAMES = {
        'BGR': ['B', 'G', 'R'],
        'HSV': ['H', 'S', 'V'],
        'HLS': ['H', 'L', 'S'],
        'LUV': ['L', 'U', 'V'],
        'LAB': ['L', 'A', 'B'],
        'YCrCb': ['Y', 'Cr', 'Cb']
    }

    def __init__(self, colour_space: str):
        self.name = colour_space
        if self._COLOR_SPACE_IDS[self.name] is None:
            self.bgr_to_color_space = lambda img: img
        else:
            self.bgr_to_color_space = lambda img: cv2.cvtColor(img, self._COLOR_SPACE_IDS[self.name])
        self.channels = self._COLOR_SPACE_CHANNEL_NAMES[self.name]
