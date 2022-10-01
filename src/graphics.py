import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
from typing import List

from utils import ColourSpace


def plot_image_grid(images: List[np.array], images_names: List[str] = None, title: str = None) -> None:
    assert images_names is None or len(images_names) == len(images), \
        '`images_names` must not be provided or it must have the same size as `images`.'

    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    if title is not None:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle('', fontsize=16)
    for idx, img in enumerate(images):
        # Add an ax to the plot
        plt.subplot(1, len(images), idx + 1)
        # Remove the numerical axes from the image
        plt.axis('off')
        # If the image has three dimensions plot it as a color image, otherwise as a grayscale one
        if len(img.shape) == 3:
            plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        if images_names is not None:
            plt.title(images_names[idx])
    plt.show()


'''def plot_image_grid(images: List[np.array], images_names: List[str] = None, title: str = None) -> None:
    assert images_names is None or len(images_names) == len(images), \
        '`images_names` must not be provided or it must have the same size as `images`.'

    is_sublist = any(isinstance(el, list) for el in images)

    if is_sublist:
        rows = len(images)
        cols = len(images[0])
    else:
        rows = 1
        cols = len(images)

    # fig, axes = plt.subplots(rows, cols, figsize=(10 * rows, 10))
    fig = plt.figure(figsize=(8, 8))
    spec = GridSpec(ncols=cols, nrows=rows, figure=fig)
    if title is not None:
        fig.suptitle(title, fontsize=18)

    axes = [plt.subplot(spec[i]) for i in range(rows * cols)]

    if is_sublist:
        images = [el for sublist in images for el in sublist]
        if images_names is not None:
            images_names = [el for sublist in images_names for el in sublist]

    for idx, img in enumerate(images):
        # Remove the numerical axes from the image
        axes[idx].axis('off')
        # If the image has three dimensions plot it as a color image, otherwise as a grayscale one
        if len(img.shape) == 3:
            axes[idx].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        else:
            axes[idx].imshow(img, cmap='gray', vmin=0, vmax=255)
        if images_names is not None:
            axes[idx].set_title(images_names[idx])
        axes[idx].set_adjustable('box')

    # Remove extra space between the sub-images
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.5, hspace=0.2, top=0.88)''';


def plot_histogram_grid(images: List[np.array], images_names: List[str] = None, title: str = None) -> None:
    assert images_names is None or len(images_names) == len(images), \
        '`images_names` must not be provided or it must have the same size as `images`.'

    fig = plt.figure(figsize=(20, 13))
    if title is not None:
        fig.suptitle(title, fontsize=20)
    for idx, img in enumerate(images):
        # Add an ax to the plot
        plt.subplot(2, len(images), idx + 1)
        # Obtain the gray-level histogram
        hist, _ = np.histogram(img.flatten(), 256, [0, 256])
        # Plot the histogram
        plt.stem(hist, use_line_collection=True)
        plt.xlabel('gray levels', fontsize=13)
        plt.ylabel('pixels', fontsize=13)
        if images_names is not None:
            plt.title(images_names[idx])
    plt.show()


def get_highlighted_roi_by_mask(img: np.array, mask: np.array, highlight_channel: str = 'green') -> np.array:
    channel_map = {'blue': 0, 'green': 1, 'red': 2}
    # Turn mask into BGR image
    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    # Force the bits of every channel except the selected one at 0
    mask[:, :, [i for i in range(3) if i != channel_map[highlight_channel]]] = 0
    # Highlight the unmasked ROI
    return cv.addWeighted(mask, 0.3, img, 1, 0)


def highlight_unmasked_region(img: np.array, mask: np.array, title: str, highlight_channel: str = 'green') -> None:
    # Get image with highlighted ROI
    highlighted_roi = get_highlighted_roi_by_mask(img, mask, highlight_channel)
    # Plot image with highlighted ROI
    plt.imshow(cv.cvtColor(highlighted_roi, cv.COLOR_BGR2RGB))
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()


def plot_colour_distribution_3d(images: List[np.array], images_names: List[str], colour_space: ColourSpace,
                                masks: List[np.array] = None, title: str = None) -> None:
    fig = plt.figure(figsize=(15, 5))

    for idx, colour_img in enumerate(images):
        # Turn the colored image into the defined colour space
        img = colour_space.bgr_to_color_space(colour_img)

        # Get channels of the image and flatten them
        channels = cv.split(img)
        channels = [ch.flatten() for ch in channels]

        # Get channel names
        channel_names = colour_space.channels

        # Get RGB color of every pixel
        pixel_colors = cv.cvtColor(colour_img, cv.COLOR_BGR2RGB).reshape((-1, 3))

        # Remove masked pixels
        if masks is not None:
            mask = masks[idx].flatten()
            channels = [ch[mask != 0] for ch in channels]
            pixel_colors = pixel_colors[mask != 0]

        # Normalize pixel colors
        norm = colors.Normalize(vmin=-1., vmax=1.)
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
    plt.tight_layout()
    plt.show()


def plot_colour_distribution_2d(image: np.array, image_name: str, colour_space: ColourSpace,
                                mask: np.array = None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 3))

    channels_mapping = {idx: ch for idx, ch in enumerate(colour_space.channels)}

    for idx, channel_indices in enumerate([[0, 1], [0, 2], [1, 2]]):
        # Turn the colored image into the defined colour space
        img = colour_space.bgr_to_color_space(image)

        # Get channels of the image, flatten them and remove masked pixels
        channels = [img[:, :, ch] for ch in channel_indices]
        channels = [ch.flatten() for ch in channels]

        # Get RGB color of every pixel and remove the masked ones
        pixel_colors = cv.cvtColor(image, cv.COLOR_BGR2RGB).reshape((-1, 3))

        if mask is not None:
            # Flatten the mask
            mask = mask.flatten()
            # Remove masked pixels from the channels and pixel_colors arrays
            channels = [ch[mask != 0] for ch in channels]
            pixel_colors = pixel_colors[mask != 0]

        # Normalize pixel colors
        norm = colors.Normalize(vmin=-1., vmax=1.)
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
    plt.show()


def plot_image_histogram_2d(image: np.array, image_name: str, colour_space: ColourSpace, bins: int = 32,
                            tick_spacing: int = 5) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    channels_mapping = {idx: ch for idx, ch in enumerate(colour_space.channels)}

    for idx, channels in enumerate([[0, 1], [0, 2], [1, 2]]):
        hist = cv.calcHist([image], channels, None, [bins] * 2, [0, 256] * 2)

        channel_x = channels_mapping[channels[0]]
        channel_y = channels_mapping[channels[1]]

        ax = axes[idx]
        ax.set_xlim([0, bins - 1])
        ax.set_ylim([0, bins - 1])

        ax.set_xlabel(f'Channel {channel_x}')
        ax.set_ylabel(f'Channel {channel_y}')
        ax.set_title(f'2D Color Histogram for {channel_x} and '
                     f'{channel_y}')

        ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        im = ax.imshow(hist)

    fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal')
    fig.suptitle(f'2D Colour Histograms of image {image_name} with {bins} bins in colour space {colour_space.name}',
                 fontsize=16)
    plt.show()
