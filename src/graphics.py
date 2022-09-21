import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def plot_image_grid(images: List[np.array], title: str, images_names: List[str] = None) -> None:
    assert images_names is None or len(images_names) == len(images), \
        '`images_names` must not be provided or it must have the same size as `images`.'

    fig = plt.figure(figsize=(20, 5))
    fig.suptitle(title, fontsize=20)
    for idx, img in enumerate(images):
        # Add an ax to the plot
        plt.subplot(1, len(images), idx + 1)
        # Remove the numerical axes from the image
        plt.axis('off')
        # If the image has three dimensions plot it as a color image, otherwise as a grayscale one
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        if images_names is not None:
            plt.title(images_names[idx])
    # Remove extra space between the sub-images
    plt.tight_layout()
    plt.show()


def plot_histogram_grid(images: List[np.array], title: str, images_names: List[str] = None) -> None:
    assert images_names is None or len(images_names) == len(images), \
        '`images_names` must not be provided or it must have the same size as `images`.'

    fig = plt.figure(figsize=(20, 13))
    fig.suptitle(title, fontsize=20)
    for idx, img in enumerate(images):
        # Add an ax to the plot
        plt.subplot(2, 3, idx + 1)
        # Obtain the gray-level histogram
        hist, _ = np.histogram(img.flatten(), 256, [0, 256])
        # Plot the histogram
        plt.stem(hist, use_line_collection=True)
        plt.xlabel('gray levels', fontsize=13)
        plt.ylabel('pixels', fontsize=13)
        if images_names is not None:
            plt.title(images_names[idx])
    plt.show()
