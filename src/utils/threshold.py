"""
Module about thresholding and segmentation functions.
"""
import cv2 as _cv
from enum import Enum as _Enum
import numpy as _np
from time import time as _time
from typing import Any as _Any, List as _List, Tuple as _Tuple, Union as _Union

from .general import plot_image_grid as _plot_image_grid, get_highlighted_roi_by_mask as _get_highlighted_roi_by_mask


class ThresholdingMethod(_Enum):
    """Methods to use for binary thresholding"""
    MANUAL = 0, "Manual Intensity binarization"
    OTSU = 1, "Otsu's Algorithm"
    TWEAKED_OTSU = 2, "Tweaked Otsu's Algorithm"
    ADAPTIVE = 3, "Local Adaptive Thresholding"


_THRESHOLDING_NAMES = {
    ThresholdingMethod.MANUAL: 'Manual Intensity Binarization',
    ThresholdingMethod.OTSU: "Otsu's Algorithm",
    ThresholdingMethod.TWEAKED_OTSU: "Tweaked Otsu's Algorithm",
    ThresholdingMethod.ADAPTIVE: "Adaptive Thresholding"
}


def _threshold_by_method(image: _np.ndarray, method: ThresholdingMethod,
                         **kwargs: _Any) -> _Tuple[_Union[float, None], _np.ndarray]:
    """
    Function that applies a binary thresholding function on an image given a specific method and eventual other named
    parameters (`**kwargs`).

    It returns in the given order the found threshold (just for global thresholding methods) and the segmentation mask
    of the image obtained by applying the threshold over itself.
    The "Local Adaptive" method returns `None` as a threshold since it does not find a global value for it.

    Parameters
    ----------
    image: ndarray
        The grayscale image from which the binary segmentation mask is obtained
    method: ThresholdingMethod
        Enum of the method to choose to apply the thresholding binarization of the image
    **kwargs: Any
        Keyword arguments that are passed to the thresholding functions described by method (`manual_threshold`,
        `otsu_threshold`, `tweaked_otsu_threshold`, `adaptive_threshold_and_flood_fill_background`).
        They include: `threshold` just for `MANUAL` method; `tweak_factor` just for `TWEAKED_OTSU` method; `block_size`
        and `c` just for `ADAPTIVE` method

    Returns
    -------
    threshold: Union[float, None]
        The binarization threshold found by the algorithm
    mask: ndarray
        The binary segmentation mask of `image`
    """
    if method == ThresholdingMethod.MANUAL:
        assert kwargs['threshold'] is not None, f'{_THRESHOLDING_NAMES[method]} needs parameter `threshold`.'
        return manual_threshold(image, kwargs['threshold'])
    elif method == ThresholdingMethod.OTSU:
        return otsu_threshold(image)
    elif method == ThresholdingMethod.TWEAKED_OTSU:
        assert kwargs['tweak_factor'] is not None, \
            f'{_THRESHOLDING_NAMES[method]} needs parameter `tweak_factor`.'
        return tweaked_otsu_threshold(image, kwargs['tweak_factor'])
    elif method == ThresholdingMethod.ADAPTIVE:
        assert kwargs['block_size'] is not None and kwargs['c'] is not None, \
            f'{_THRESHOLDING_NAMES[method]} needs parameters `block_size` and `c`.'
        return adaptive_threshold_and_flood_fill_background(image, kwargs['block_size'], kwargs['c'])
    else:
        raise KeyError('Method is not valid')


def get_fruit_segmentation_mask(image: _np.ndarray, method: ThresholdingMethod, **kwargs: _Any) -> _np.ndarray:
    """
    Function that returns a binary segmented mask on an image given a specific method and eventual other named
    parameters (`**kwargs`), Then flood-fills it in order to fill the possible black holes inside of the white mask.

    Parameters
    ----------
    image: ndarray
        The grayscale image from which the binary segmentation mask is obtained
    method: ThresholdingMethod
        Enum of the method to choose to apply the thresholding segmentation masking of the image
    **kwargs: Any
        Keyword arguments that are passed to the thresholding functions described by method (`manual_threshold`,
        `otsu_threshold`, `tweaked_otsu_threshold`, `adaptive_threshold_and_flood_fill_background`).
        They include: `threshold` just for `MANUAL` method; `tweak_factor` just for `TWEAKED_OTSU` method; `block_size`
        and `c` just for `ADAPTIVE` method

    Returns
    -------
    mask: ndarray
        The binary segmentation mask of `image` obtained according to a certain given or found threshold processed
        through flood-filling
    """
    _, mask = _threshold_by_method(image, method, **kwargs)
    return apply_flood_fill_to_segmentation_mask(mask)


def manual_threshold(image: _np.ndarray, threshold: float) -> _Tuple[float, _np.ndarray]:
    """
    Function that applies a "Global Intensity Binarization" thresholding function on an image given a specific manual
    threshold.

    It returns in the given order the given threshold and the segmentation mask of the image obtained by applying the
    threshold over itself.

    Parameters
    ----------
    image: ndarray
        The grayscale image from which the binary segmentation mask is obtained
    threshold: float
        The threshold to use to binarize the image

    Returns
    -------
    threshold: float
        The segmentation threshold given to the algorithm
    mask: ndarray
        The binary segmentation mask of `image` obtained according to `threshold`
    """
    return _cv.threshold(image, threshold, 255, _cv.THRESH_BINARY)


def otsu_threshold(image: _np.ndarray) -> _Tuple[float, _np.ndarray]:
    """
    Function that applies "Otsu's Algorithm" to find a global threshold and obtain the binary segmentation mask of an
    image according to it.

    It returns in the given order the found threshold and the segmentation mask of the image obtained by applying the
    threshold over itself.

    Parameters
    ----------
    image: ndarray
        The grayscale image from which the binary segmentation mask is obtained

    Returns
    -------
    threshold: float
        The segmentation threshold found by the algorithm
    mask: ndarray
        The binary segmentation mask of `image` obtained according to `threshold`
    """
    return _cv.threshold(image, 0, 255, _cv.THRESH_BINARY + _cv.THRESH_OTSU)


def tweaked_otsu_threshold(image: _np.ndarray, tweak_factor: float) -> _Tuple[float, _np.ndarray]:
    """
    Function that applies "Otsu's Algorithm" to find a global threshold and then multiplies it by a "tweak factor" in
    order to obtain a tweaked threshold and apply it to find the binary segmentation mask of an image according to it.

    It returns in the given order the found tweaked threshold and the segmentation mask of the image obtained by
    applying the threshold over itself.

    Parameters
    ----------
    image: ndarray
        The grayscale image from which the binary segmentation mask is obtained
    tweak_factor: float
        Value that will be multiplied to the value of the threshold obtained by "Otsu's Algorithm" in order to tweak it

    Returns
    -------
    threshold: float
        The segmentation threshold found by the algorithm
    mask: ndarray
        The binary mask of `image` obtained according to `threshold`
    """
    # Get threshold by Otsu
    threshold, _ = _cv.threshold(image, 0, 255, _cv.THRESH_BINARY + _cv.THRESH_OTSU)
    # Tweak the threshold
    desired_threshold = threshold * tweak_factor
    # Apply "Manual Intensity Binarization" using the new computed threshold
    return _cv.threshold(image, desired_threshold, 255, _cv.THRESH_BINARY)


def adaptive_threshold_and_flood_fill_background(image: _np.ndarray, block_size: int,
                                                 c: int) -> _Tuple[None, _np.ndarray]:
    """
    Function that applies a "Local Adaptive" thresholding method to obtain the binary segmentation mask of an image,
    then applies flood-fill in order to get a full black background.

    It returns in the given order `None`, since a global threshold is not computed, and the obtained segmentation mask
    of the image.

    Parameters
    ----------
    image: ndarray
        The grayscale image from which the binary segmentation mask is obtained
    block_size: int
        The size of the window of pixel neighbours used to compute the threshold of each pixel through "Local Adaptive"
        method
    c: int
        Constant subtracted from the mean computed by the "Local Adaptive" method for each pixel

    Returns
    -------
    _: None
        Since no global threshold can be found by the method, `None` is returned
    mask: ndarray
        The binary segmentation mask of `image` obtained after the "Local Adaptive" method and the flood-filling
    """
    mask = _cv.adaptiveThreshold(image, 255, _cv.ADAPTIVE_THRESH_MEAN_C, _cv.THRESH_BINARY, block_size, c)
    mask = _np.pad(mask, 1, mode='constant', constant_values=255)
    _cv.floodFill(mask, None, (0, 0), 0)
    mask = mask[1:-1, 1:-1]
    return None, mask


def apply_flood_fill_to_segmentation_mask(image: _np.ndarray) -> _np.ndarray:
    """
    Function that applies a flood-filling on a binary segmentation mask in order to remove possible small holes inside
    of it.

    Parameters
    ----------
    image: ndarray
        The binary segmentation mask to flood-fill

    Returns
    -------
    mask: ndarray
        The flood-filled segmentation mask
    """
    # Copy the threshold-ed image
    img_flood_filled = image.copy()

    # Pad image to guarantee that all the background is flood-filled
    img_flood_filled = _np.pad(img_flood_filled, 1, mode='constant', constant_values=0)

    # Mask used to flood filling
    # The size needs to be 2 pixel larger than the image
    h, w = img_flood_filled.shape[:2]
    mask = _np.zeros((h + 2, w + 2), _np.uint8)

    # Flood-fill from the upper-left corner (point (0, 0))
    _cv.floodFill(img_flood_filled, mask, (0, 0), 255)

    # Down-sample the image to its original size
    img_flood_filled = img_flood_filled[1:-1, 1:-1]

    # Invert the flood-filled image
    img_copy_inv = ~img_flood_filled

    # Combine the original and inverted flood-filled image to obtain the foreground
    return image + img_copy_inv


def plot_segmentation_process(gray_images: _List[_np.ndarray], display_images: _List[_np.ndarray],
                              images_names: _List[str], method: ThresholdingMethod, **kwargs: _Any) -> None:
    """
    Function that plots the binary segmentation masking process of a series of images given a specific thresholding
    method and eventual other named parameters (`**kwargs`)

    For each image it shows:
        - The segmentation mask obtained by the method.
        - The segmentation mask after flood-filling.
        - The segmentation mask applied on a version of the initial image used for display purposes (For instance the
        color version of the image)

    Parameters
    ----------
    gray_images: List[ndarray]
        The grayscale images for which the masking process is shown
    display_images: List[ndarray]
        Images corresponding to `gray_images`, used for displaying the computed binary segmentation masks over them.
    images_names: List[str]
        Names of the images from which the binary segmentation masks are computed
    method: ThresholdingMethod
        Enum of the method to choose to apply the thresholding segmentation masking of the images
    **kwargs: Any
        Keyword arguments that are passed to the thresholding functions described by method (`manual_threshold`,
        `otsu_threshold`, `tweaked_otsu_threshold`, `adaptive_threshold_and_flood_fill_background`).
        They include: `threshold` just for `MANUAL` method; `tweak_factor` just for `TWEAKED_OTSU` method; `block_size`
        and `c` just for `ADAPTIVE` method
    """
    masks = []
    thresholds = []

    # Compute the mask and th threshold of each image
    for image in gray_images:
        threshold, mask = _threshold_by_method(image, method, **kwargs)
        thresholds.append(threshold)
        masks.append(mask)

    # Flood fill the masks
    flood_filled_masks = [apply_flood_fill_to_segmentation_mask(i) for i in masks]

    # Highlight the masks over the display images
    highlighted_images = [_get_highlighted_roi_by_mask(d, m) for d, m in zip(display_images, flood_filled_masks)]

    # Plot the segmentation process of each image
    for m, t, ffm, h, n in zip(masks, thresholds, flood_filled_masks, highlighted_images, images_names):
        processed_images_names = [
            f'Binary segmentation mask {f" (threshold = {t})" if t is not None else ""}',
            'Flood-filled segmentation mask',
            'Outlined fruit'
        ]
        _plot_image_grid([m, ffm, h], processed_images_names,
                         f'Outline of the fruits obtained through {_THRESHOLDING_NAMES[method]} for image {n}')


# TODO: Remove Method
def segment_fruit_and_plot(gray_images: _List[_np.ndarray], display_images: _List[_np.ndarray],
                           images_names: _List[str], method: ThresholdingMethod, title: str = None,
                           **kwargs: _Any) -> None:
    masks = []
    thresholds = []

    for image in gray_images:
        threshold, mask = _threshold_by_method(image, method, **kwargs)
        thresholds.append(threshold)
        masks.append(mask)

    masks = [apply_flood_fill_to_segmentation_mask(i) for i in masks]

    highlighted_images = [_get_highlighted_roi_by_mask(d, m) for d, m in zip(display_images, masks)]

    processed_images_names = [f'Image {n} {f" (threshold = {t})" if t is not None else ""}'
                              for n, t in zip(images_names, thresholds)]

    if title is None:
        title = f'Outline of the fruits obtained through {_THRESHOLDING_NAMES[method]}'

    _plot_image_grid(highlighted_images, processed_images_names, title)


# TODO: Remove Method
def plot_thresholding_on_light_and_dark_images(dark_images: _List[_np.ndarray], light_images: _List[_np.ndarray],
                                               images_names: _List[str], method: ThresholdingMethod,
                                               **kwargs: _Any) -> None:
    segment_fruit_and_plot(dark_images, dark_images, images_names, method,
                           f'Outline of the fruits obtained through {_THRESHOLDING_NAMES[method]} on darker images',
                           **kwargs)
    segment_fruit_and_plot(light_images, light_images, images_names, method,
                           f'Outline of the fruits obtained through {_THRESHOLDING_NAMES[method]} on lighter images',
                           **kwargs)


def get_segmentation_time(images: _List[_np.ndarray], method: ThresholdingMethod, repeats: int = 1_000,
                          **kwargs: _Any) -> _Tuple[float, float]:
    """
    Function that computes the total time needed to apply segmentation and flood-filling on a series of images with a
    given number of repetitions. Moreover, the mean time per image is obtained by the function.

    Additionally, the functions prints the total and mean segmentation times.

    Parameters
    ----------
    images: List[ndarray]
        The grayscale images for which the time of the segmentation process is computed
    method: ThresholdingMethod
        Enum of the method to choose to apply the thresholding segmentation of the images
    repeats: int, optional
        Number of time the segmentation process is repeated for each image (default: 1,000)
    **kwargs: Any
        Keyword arguments that are passed to the thresholding functions described by method (`manual_threshold`,
        `otsu_threshold`, `tweaked_otsu_threshold`, `adaptive_threshold_and_flood_fill_background`).
        They include: `threshold` just for `MANUAL` method; `tweak_factor` just for `TWEAKED_OTSU` method; `block_size`
        and `c` just for `ADAPTIVE` method

    Returns
    -------
    total_time: float
        Total time needed for the segmentation process of the images with the given repetitions
    mean_time: float
        Mean time needed to segment an image
    """
    # Compute the initial time
    s = _time()

    # Mask the images with repetitions
    for img in images * repeats:
        get_fruit_segmentation_mask(img, method, **kwargs)

    # Get total time according to the initial time and mean time per image
    total_time = _time() - s
    mean_time = total_time / (len(images) * repeats)

    # Print the total and mean time
    print(f'Total time to perform {_THRESHOLDING_NAMES[method]} on {repeats * len(images)} ',
          f'images = {total_time:.6f}')
    print(f'Mean time per instance to perform {_THRESHOLDING_NAMES[method]} = {mean_time:.6f}')

    return total_time, mean_time
