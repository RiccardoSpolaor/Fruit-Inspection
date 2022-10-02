from builtins import float
import cv2 as cv
import numpy as np
from enum import Enum
from time import time
from typing import List, TypedDict

from .graphics import plot_image_grid, get_highlighted_roi_by_mask


class ThresholdingMethod(Enum):
    MANUAL = 0
    OTSU = 1
    TWEAKED_OTSU = 2
    ADAPTIVE = 3


class _ThresholdingParameters(TypedDict):
    threshold: float
    tweak_factor: float
    block_size: int
    c: int


_THRESHOLDING_NAMES = {
    0: 'Manual Intensity Binarization',
    1: "Otsu's Algorithm",
    2: "Tweaked Otsu's Algorithm",
    3: "Adaptive Thresholding"
}


def _check_thresholding_parameters(method: ThresholdingMethod, **kwargs: _ThresholdingParameters) -> None:
    if method.value == ThresholdingMethod.MANUAL:
        assert kwargs['threshold'] is not None, \
            f'{_THRESHOLDING_NAMES[method.value]} needs parameters `threshold`.'
    if method.value == ThresholdingMethod.OTSU:
        assert kwargs['tweak_factor'] is not None, \
            f'{_THRESHOLDING_NAMES[method.value]} needs parameter `tweak_factor`.'
    if method.value == ThresholdingMethod.OTSU:
        assert kwargs['block_size'] is not None and kwargs['c'] is not None, \
            f'{_THRESHOLDING_NAMES[method.value]} needs parameters `block_size` and `c`.'


def get_fruit_mask(image: np.ndarray, method: ThresholdingMethod, **kwargs: _ThresholdingParameters) -> np.ndarray:
    _check_thresholding_parameters(method, **kwargs)
    _, mask = _THRESHOLDING_FUNCTIONS[method.value](image, **kwargs)
    return apply_flood_fill_to_mask(mask)


def manual_threshold(image: np.ndarray, threshold: float) -> (float, np.ndarray):
    threshold, mask = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
    return threshold, mask


def otsu_threshold(image: np.ndarray) -> (float, np.ndarray):
    threshold, mask = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return threshold, mask


def tweaked_otsu_threshold(image: np.ndarray, tweak_factor: float = .5) -> (float, np.ndarray):
    threshold, _ = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    desired_threshold = threshold * tweak_factor
    threshold, mask = cv.threshold(image, desired_threshold, 255, cv.THRESH_BINARY)
    return threshold, mask


def adaptive_threshold(image: np.ndarray, block_size: int, c: int) -> (None, np.ndarray):
    return None, cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block_size, c)


def adaptive_threshold_and_flood_fill_background(image: np.ndarray, block_size: int, c: int) -> (None, np.ndarray):
    _, mask = adaptive_threshold(image, block_size, c)
    mask = np.pad(mask, 1, mode='constant', constant_values=255)
    cv.floodFill(mask, None, (0, 0), 0)
    mask = mask[1:-1, 1:-1]
    return None, mask


_THRESHOLDING_FUNCTIONS = {
    0: manual_threshold,
    1: otsu_threshold,
    2: tweaked_otsu_threshold,
    3: adaptive_threshold_and_flood_fill_background
}


def apply_flood_fill_to_mask(image: np.array) -> np.array:
    # Copy the threshold-ed image
    img_flood_filled = image.copy()

    # Pad image to guarantee that all the background is flood-filled
    img_flood_filled = np.pad(img_flood_filled, 1, mode='constant', constant_values=0)

    # Mask used to flood filling
    # The size needs to be 2 pixel larger than the image
    h, w = img_flood_filled.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Flood-fill from the upper-left corner (point (0, 0))
    cv.floodFill(img_flood_filled, mask, (0, 0), 255)

    # Down-sample the image to its original size
    img_flood_filled = img_flood_filled[1:-1, 1:-1]

    # Invert the flood-filled image
    img_copy_inv = ~img_flood_filled

    # Combine the original and inverted flood-filled image to obtain the foreground
    return image + img_copy_inv


def plot_masking_process(gray_images: List[np.ndarray], display_images: List[np.ndarray], images_names: List[str],
                         method: ThresholdingMethod, **kwargs: _ThresholdingParameters) -> None:
    _check_thresholding_parameters(method, **kwargs)

    masks = []
    thresholds = []

    for image in gray_images:
        threshold, mask = _THRESHOLDING_FUNCTIONS[method.value](image, **kwargs)
        thresholds.append(threshold)
        masks.append(mask)

    flood_filled_masks = [apply_flood_fill_to_mask(i) for i in masks]

    highlighted_images = [get_highlighted_roi_by_mask(d, m) for d, m in zip(display_images, flood_filled_masks)]

    for m, t, ffm, h, n in zip(masks, thresholds, flood_filled_masks, highlighted_images, images_names):
        processed_images_names = [
            f'Binary mask {f" (threshold = {t})" if t is not None else ""}',
            'Flood-filled mask',
            'Outlined fruit'
        ]
        plot_image_grid([m, ffm, h], processed_images_names,
                        f'Outline of the fruits obtained through {_THRESHOLDING_NAMES[method.value]} for image {n}')


def mask_fruit_and_plot(gray_images: List[np.ndarray], display_images: List[np.ndarray], images_names: List[str],
                        method: ThresholdingMethod, title: str = None, **kwargs: _ThresholdingParameters) -> None:
    _check_thresholding_parameters(method, **kwargs)

    masks = []
    thresholds = []

    for image in gray_images:
        threshold, mask = _THRESHOLDING_FUNCTIONS[method.value](image, **kwargs)
        thresholds.append(threshold)
        masks.append(mask)

    masks = [apply_flood_fill_to_mask(i) for i in masks]

    highlighted_images = [get_highlighted_roi_by_mask(d, m) for d, m in zip(display_images, masks)]

    processed_images_names = [f'Image {n} {f" (threshold = {t})" if t is not None else ""}'
                              for n, t in zip(images_names, thresholds)]

    if title is None:
        title = f'Outline of the fruits obtained through {_THRESHOLDING_NAMES[method.value]}'

    plot_image_grid(highlighted_images, processed_images_names, title)


def plot_thresholding_on_light_and_dark_images(dark_images: List[np.ndarray], light_images: List[np.ndarray],
                                               images_names: List[str], method: ThresholdingMethod,
                                               **kwargs: _ThresholdingParameters) -> None:
    _check_thresholding_parameters(method, **kwargs)

    mask_fruit_and_plot(dark_images, dark_images, images_names, method,
                        f'Outline of the fruits obtained through {_THRESHOLDING_NAMES[method.value]} on darker images',
                        **kwargs)
    mask_fruit_and_plot(light_images, light_images, images_names, method,
                        f'Outline of the fruits obtained through {_THRESHOLDING_NAMES[method.value]} on lighter images',
                        **kwargs)


def get_thresholding_time(images: List[np.ndarray], method: ThresholdingMethod, repeats: int = 1_000,
                          **kwargs: _ThresholdingParameters) -> (float, float):
    _check_thresholding_parameters(method, **kwargs)

    s = time()

    for img in images * repeats:
        get_fruit_mask(img, method, **kwargs)

    total_time = time() - s
    mean_time = total_time / (len(images) * repeats)

    print(f'Total time to perform {_THRESHOLDING_NAMES[method.value]} on {repeats * len(images)} ',
          f'images = {total_time:.6f}')
    print(f'Mean time per instance to perform {_THRESHOLDING_NAMES[method.value]} = {mean_time:.6f}')

    return total_time, mean_time
