import cv2 as cv
import numpy as np
from typing import List

from .general import apply_flood_fill
from .graphics import plot_image_grid, get_highlighted_roi_by_mask


def manual_threshold(image: np.array, threshold: float) -> (float, np.array):
    threshold, mask = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
    return threshold, mask


def otsu_threshold(image: np.array) -> (float, np.array):
    threshold, mask = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return threshold, mask


def tweaked_otsu_threshold(image: np.array, tweak_factor: float = .5) -> (float, np.array):
    threshold, _ = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    desired_threshold = threshold * tweak_factor
    threshold, mask = cv.threshold(image, desired_threshold, 255, cv.THRESH_BINARY)
    return threshold, mask


def adaptive_threshold(image: np.array, kernel_size: int, c: int) -> (None, np.array):
    return None, cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, kernel_size, c)


def adaptive_threshold_and_flood_fill_background(image: np.array, kernel_size: int, c: int) -> (None, np.array):
    mask = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, kernel_size, c)
    mask = np.pad(mask, 1, mode='constant', constant_values=255)
    cv.floodFill(mask, None, (0, 0), 0)
    mask = mask[1:-1, 1:-1]
    return None, mask


def plot_masking_process(gray_images: List[np.array], display_images: List[np.array], images_names: List[str],
                         thresholding_function: callable, technique: str) -> None:
    masks = []
    thresholds = []

    for img in gray_images:
        threshold, mask = thresholding_function(img)
        thresholds.append(threshold)
        masks.append(mask)

    # processed_images_names = [f'{n}{f" (threshold = {t})" if threshold is not None else ""}'
    #                          for n, t in zip(images_names, thresholds)]

    # plot_image_grid(masks, processed_images_names, f'Binary masks obtained through {technique}')

    flood_filled_masks = [apply_flood_fill(i) for i in masks]

    # plot_image_grid(masks, processed_images_names, f'Binary masks obtained through {technique} and flood-fill')

    highlighted_images = [get_highlighted_roi_by_mask(d, m) for d, m in zip(display_images, flood_filled_masks)]

    # processed_images_names = [f'{n}' for n in images_names]

    for m, t, ffm, h, n in zip(masks, thresholds, flood_filled_masks, highlighted_images, images_names):
        processed_images_names = [
            f'Binary mask {f" (threshold = {t})" if t is not None else ""}',
            'Flood-filled mask',
            'Outlined fruit'
        ]
        plot_image_grid([m, ffm, h], processed_images_names,
                        f'Outline of the fruits obtained through {technique} for image {n}')

    # plot_image_grid(highlighted_images, processed_images_names, f'Outline of the fruits obtained through {technique}')


def compute_binary_masks_and_plot(gray_images: List[np.array], display_images: List[np.array], images_names: List[str],
                                  thresholding_function: callable, technique: str) -> None:
    masks = []
    thresholds = []

    for img in gray_images:
        threshold, mask = thresholding_function(img)
        thresholds.append(threshold)
        masks.append(mask)

    masks = [apply_flood_fill(i) for i in masks]

    highlighted_images = [get_highlighted_roi_by_mask(d, m) for d, m in zip(display_images, masks)]

    processed_images_names = [f'Image {n} {f" (threshold = {t})" if t is not None else ""}'
                              for n, t in zip(images_names, thresholds)]
    plot_image_grid(highlighted_images, processed_images_names,
                    f'Outline of the fruits obtained through {technique}')
