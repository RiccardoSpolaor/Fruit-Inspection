"""
Module about colour related functions.
"""
import cv2 as _cv
import matplotlib.pyplot as _plt
from matplotlib import colors as _colors
import numpy as _np
from typing import List as _List


class ColourSpace:
    """
    Class of a colour space.

    Parameters
    ----------
    colour_space : str
        Name of the colour space from which the class is obtained

    Attributes
    ----------
    name: str
        Name of the colour space
    channels: List[str]
        Name of the channels of the colour space

    Methods
    -------
    bgr_to_colour_space(img: ndarray) -> ndarray
        Transform the image in the colour space instantiated by this class.
    """

    _COLOR_SPACE_IDS = {
        'BGR': None,
        'HSV': _cv.COLOR_BGR2HSV_FULL,
        'HLS': _cv.COLOR_BGR2HLS_FULL,
        'LUV': _cv.COLOR_BGR2Luv,
        'LAB': _cv.COLOR_BGR2LAB,
        'YCrCb': _cv.COLOR_BGR2YCrCb
    }

    _COLOR_SPACE_CHANNEL_NAMES = {
        'BGR': ['B', 'G', 'R'],
        'HSV': ['H', 'S', 'V'],
        'HLS': ['H', 'L', 'S'],
        'LUV': ['L', 'U', 'V'],
        'LAB': ['L', 'A', 'B'],
        'YCrCb': ['Y', 'Cr', 'Cb']
    }

    def __init__(self, colour_space: str) -> None:
        self.name = colour_space
        self.channels = self._COLOR_SPACE_CHANNEL_NAMES[self.name]

    def bgr_to_colour_space(self, image) -> _np.ndarray:
        """
        Transform the image in the colour space instantiated by this class.

        Parameters
        ----------
        image: ndarray
            The image to transform in the colour space instantiated by the class

        Returns
        -------
        transformed_image: ndarray
            The image transformed in the given colour space
        """
        if self._COLOR_SPACE_IDS[self.name] is None:
            return image
        else:
            return _cv.cvtColor(image, self._COLOR_SPACE_IDS[self.name])


def plot_colour_distribution_3d(images: _List[_np.ndarray], images_names: _List[str], colour_space: ColourSpace,
                                masks: _List[_np.ndarray] = None, title: str = None) -> None:
    """
    Function to plot the colour distribution of a series of images onsidering the channels

    Parameters
    ----------
    images: List[ndarray]
        Images of which the colour distribution is plotted
    images_names: List[str]
        Names of the images of which the colour distribution is plotted
    colour_space: ColourSpace
        Colour space considered to plot the colour distribution of each image
    masks: List[ndarray], optional
        Optional list of masks that exclude pixels while plotting the colour distribution. By default no masks are
        provided
    title: str, optional
        Optional title of the plot
    """
    fig = _plt.figure(figsize=(15, 5))

    for idx, colour_img in enumerate(images):
        # Turn the colored image into the defined colour space
        img = colour_space.bgr_to_colour_space(colour_img)

        # Get channels of the image and flatten them
        channels = _cv.split(img)
        channels = [ch.flatten() for ch in channels]

        # Get channel names
        channel_names = colour_space.channels

        # Get RGB color of every pixel
        pixel_colors = _cv.cvtColor(colour_img, _cv.COLOR_BGR2RGB).reshape((-1, 3))

        # Remove masked pixels
        if masks is not None:
            mask = masks[idx].flatten()
            channels = [ch[mask != 0] for ch in channels]
            pixel_colors = pixel_colors[mask != 0]

        # Normalize pixel colors
        norm = _colors.Normalize(vmin=-1., vmax=1.)
        norm.autoscale(pixel_colors)
        pixel_colors = norm(pixel_colors).tolist()

        # Plot a 3D scatter-plot if three channels are present
        ax = fig.add_subplot(1, len(images), idx + 1, projection='3d')
        ax.scatter(channels[0], channels[1], channels[2], facecolors=pixel_colors, marker=".")
        ax.set_xlabel(f'Channel {channel_names[0]}')
        ax.set_ylabel(f'Channel {channel_names[1]}')
        ax.set_zlabel(f'Channel {channel_names[2]}')
        ax.set_title(f'Color distribution for {images_names[idx]}')

    if title is None:
        fig.suptitle(f'Distribution of pixels in the {colour_space.name} colour space', fontsize=16)
    else:
        fig.suptitle(title, fontsize=16)
    # Remove extra space between the sub-images
    _plt.tight_layout()
    _plt.show()


def plot_colour_distribution_2d(image: _np.ndarray, image_name: str, colour_space: ColourSpace,
                                mask: _np.ndarray = None) -> None:
    """
    Function to plot the colour distributions of an image considering all different couples of channels

    Parameters
    ----------
    image: ndarray
        Image of which the colour distributions are plotted
    image_name: str
        Name of the image of which the colour distributions are plotted
    colour_space: ColourSpace
        Colour space considered to plot the colour distributions of the image
    mask: List[ndarray], optional
        Optional mask that exclude pixels while plotting the colour distributions. By default no mask is provided
    """
    fig, axes = _plt.subplots(1, 3, figsize=(15, 3))

    channels_mapping = {idx: ch for idx, ch in enumerate(colour_space.channels)}

    # Turn the colored image into the defined colour space
    img = colour_space.bgr_to_colour_space(image)

    for idx, channel_indices in enumerate([[0, 1], [0, 2], [1, 2]]):

        # Get channels of the image, flatten them and remove masked pixels
        channels = [img[:, :, ch] for ch in channel_indices]
        channels = [ch.flatten() for ch in channels]

        # Get RGB color of every pixel and remove the masked ones
        pixel_colors = _cv.cvtColor(image, _cv.COLOR_BGR2RGB).reshape((-1, 3))

        if mask is not None:
            # Flatten the mask
            mask = mask.flatten()
            # Remove masked pixels from the channels and pixel_colors arrays
            channels = [ch[mask != 0] for ch in channels]
            pixel_colors = pixel_colors[mask != 0]

        # Normalize pixel colors
        norm = _colors.Normalize(vmin=-1., vmax=1.)
        norm.autoscale(pixel_colors)
        pixel_colors = norm(pixel_colors).tolist()

        channel_x = channels_mapping[channel_indices[0]]

        channel_y = channels_mapping[channel_indices[1]]

        ax = axes[idx]

        ax.scatter(channels[0], channels[1], facecolors=pixel_colors, marker=".")

        ax.set_xlabel(f'Channel {channel_x}')
        ax.set_ylabel(f'Channel {channel_y}')
        ax.set_title(f'Color distribution for {channel_x} and {channel_y}')

    fig.suptitle(f'Distribution of pixels of image {image_name} in the {colour_space.name} colour space', fontsize=16,
                 y=1.1)
    _plt.show()
