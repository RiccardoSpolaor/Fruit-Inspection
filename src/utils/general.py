"""
Module about general plotting and image transformation functions.
"""
import cv2 as _cv
import numpy as _np
import matplotlib.pyplot as _plt
from matplotlib import ticker as _ticker
from typing import List as _List

from .colour import ColourSpace as _ColourSpace


def plot_image_grid(images: _List[_np.ndarray], images_names: _List[str] = None, title: str = None) -> None:
    """
    Function that plots a series of images as a row grid.

    Parameters
    ----------
    images: List[ndarray]
        List of images to plot as a row grid
    images_names: List[str], optional
        List of the names of the images to plot
    title: str, optional
        Title of the plot
    """

    assert images_names is None or len(images_names) == len(images), \
        '`images_names` must not be provided or it must have the same size as `images`.'

    fig = _plt.figure(figsize=(15, 5), constrained_layout=True)
    if title is not None:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle('', fontsize=16)
    for idx, img in enumerate(images):
        # Add an ax to the plot
        _plt.subplot(1, len(images), idx + 1)
        # Remove the numerical axes from the image
        _plt.axis('off')
        # If the image has three dimensions plot it as a color image, otherwise as a grayscale one
        if len(img.shape) == 3:
            _plt.imshow(_cv.cvtColor(img, _cv.COLOR_BGR2RGB))
        else:
            _plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        if images_names is not None:
            _plt.title(images_names[idx])
    _plt.show()


def plot_histogram_grid(images: _List[_np.ndarray], images_names: _List[str] = None, title: str = None) -> None:
    """
    Function that plots the histograms of a series of grayscale images as a row grid.

    Parameters
    ----------
    images: List[ndarray]
        List of grayscale images which histograms are plotted as a row grid
    images_names: List[str], optional
        List of the names of the images which histograms are plotted
    title: str, optional
        Title of the plot
    """
    assert images_names is None or len(images_names) == len(images), \
        '`images_names` must not be provided or it must have the same size as `images`.'

    fig = _plt.figure(figsize=(20, 13))
    if title is not None:
        fig.suptitle(title, fontsize=20)
    for idx, img in enumerate(images):
        # Add an ax to the plot
        _plt.subplot(2, len(images), idx + 1)
        # Obtain the gray-level histogram
        hist, _ = _np.histogram(img.flatten(), 256, [0, 256])
        # Plot the histogram
        _plt.stem(hist, use_line_collection=True)
        _plt.xlabel('gray levels', fontsize=13)
        _plt.ylabel('pixels', fontsize=13)
        if images_names is not None:
            _plt.title(images_names[idx])
    _plt.show()


def get_highlighted_roi_by_mask(image: _np.ndarray, mask: _np.ndarray, highlight_channel: str = 'green') -> _np.ndarray:
    """
    Function that highlights a Region of Interest provided by a mask over an image with a given BGR colour.

    Parameters
    ----------
    image: ndarray
        Image on which the mask is highlighted
    mask: ndarray
        Mask illustrating the Region of Interest to highlight over the mage
    highlight_channel: str, optional
        Colour of the highlighted mask: 'blue'; 'green' or 'red' (default: 'green')

    Returns
    -------
    highlighted_roi: ndarray
        Highlighted Region of Interest over the input image
    """
    channel_map = {'blue': 0, 'green': 1, 'red': 2}
    # Turn img into BGR image if img is a Grayscale image
    if len(image.shape) == 2:
        image = _cv.cvtColor(image, _cv.COLOR_GRAY2BGR)
    # Turn mask into BGR image if img is a BGR image
    mask = _cv.cvtColor(mask, _cv.COLOR_GRAY2BGR)
    # Force the bits of every channel except the selected one at 0
    mask[:, :, [i for i in range(3) if i != channel_map[highlight_channel]]] = 0
    # Highlight the unmasked ROI
    return _cv.addWeighted(mask, 0.3, image, 1, 0)


def plot_image_histogram_2d(image: _np.ndarray, image_name: str, colour_space: _ColourSpace, bins: int = 32,
                            tick_spacing: int = 5) -> None:
    """
    Function that plots the 2D histograms of an image considering all different couples of channels.

    Parameters
    ----------
    image: ndarray
        Image of which the 2D histograms are plotted
    image_name: str
        Name of the image of which the 2D histograms are plotted
    colour_space: ColourSpace
        Colour space considered to plot the2D histograms of the image
    bins: int, optional
        bins to use in the histogram (default: 32)
    tick_spacing: int, optional
        Tick spacing of the axes of the histograms (default: 5)
    """
    fig, axes = _plt.subplots(1, 3, figsize=(15, 5))

    channels_mapping = {idx: ch for idx, ch in enumerate(colour_space.channels)}

    # Turn the colored image into the defined colour space
    image = colour_space.bgr_to_colour_space(image)

    for idx, channels in enumerate([[0, 1], [0, 2], [1, 2]]):
        hist = _cv.calcHist([image], channels, None, [bins] * 2, [0, 256] * 2)

        channel_x = channels_mapping[channels[0]]
        channel_y = channels_mapping[channels[1]]

        ax = axes[idx]
        ax.set_xlim([0, bins - 1])
        ax.set_ylim([0, bins - 1])

        ax.set_xlabel(f'Channel {channel_x}')
        ax.set_ylabel(f'Channel {channel_y}')
        ax.set_title(f'2D Color Histogram for {channel_x} and '
                     f'{channel_y}')

        ax.yaxis.set_major_locator(_ticker.MultipleLocator(tick_spacing))
        ax.xaxis.set_major_locator(_ticker.MultipleLocator(tick_spacing))

        im = ax.imshow(hist)

    fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal')
    fig.suptitle(f'2D Colour Histograms of image {image_name} with {bins} bins in colour space {colour_space.name}',
                 fontsize=16)
    _plt.show()


def get_largest_blob_in_mask(mask: _np.ndarray) -> _np.ndarray:
    """
    Function to get the largest blob in a binary mask.

    Parameters
    ----------
    mask: ndarray
        The mask from which the largest blob is extracted

    Returns
    -------
    mask_with_largest_blob: ndarray
        The mask with solely the largest blob
    """
    contours, _ = _cv.findContours(mask, _cv.RETR_EXTERNAL, _cv.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=_cv.contourArea)
    out = _np.zeros(mask.shape, _np.uint8)
    return _cv.drawContours(out, [max_contour], -1, 255, _cv.FILLED)


def apply_mask_to_image(image: _np.ndarray, mask: _np.ndarray):
    """
    Function that applies a mask over an image

    Parameters
    ----------
    image: ndarray
        Image to apply the mask over
    mask: ndarray
        Mask to apply over the image

    Returns
    -------
    masked_image: ndarray
        The masked image
    """
    if len(image.shape) == 3:
        mask = _cv.cvtColor(mask, _cv.COLOR_GRAY2BGR)
    return image & mask
