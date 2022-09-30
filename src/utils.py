import cv2
import numpy as np


def apply_flood_fill(img: np.array) -> np.array:
    # Copy the threshold-ed image
    img_flood_filled = img.copy()

    # Pad image to guarantee that all the background is flood-filled
    img_flood_filled = np.pad(img_flood_filled, 1, mode='constant', constant_values=0)

    # Mask used to flood filling
    # The size needs to be 2 pixel larger than the image
    h, w = img_flood_filled.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Flood-fill from the upper-left corner (point (0, 0))
    cv2.floodFill(img_flood_filled, mask, (0, 0), 255)

    # Down-sample the image to its original size
    img_flood_filled = img_flood_filled[1:-1, 1:-1]

    # Invert the flood-filled image
    img_copy_inv = ~img_flood_filled

    # Combine the original and inverted flood-filled image to obtain the foreground
    return img | img_copy_inv


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
