"""
Module about colour thresholding and segmentation functions.
"""
import cv2 as _cv
import numpy as _np
from scipy.spatial.distance import cdist as _cdist
from sklearn.feature_extraction.image import extract_patches_2d as _extract_patches_2d
from sklearn.mixture import GaussianMixture as _GaussianMixture
from typing import List as _List, Tuple as _Tuple

from .colour import ColourSpace as _ColourSpace
from .general import get_highlighted_roi_by_mask as _get_highlighted_roi_by_mask, plot_image_grid as _plot_image_grid


def get_k_means_segmented_image(image: _np.ndarray, channels: _Tuple[int, ...] = (0, 1, 2),
                                centers: int = 3) -> _np.ndarray:
    """
    Function that applies the K-Means algorithm to a given image in order to quantize the main colours and segment the
    image accordingly.

    It returns a grayscale segmented version of the image where the regions of the different colours are line-spaced
    with values between 0 and 255.

    Parameters
    ----------
    image: ndarray
        The colour image to segment
    channels: Tuple[int, ...]
        The colour channels to use for quantizing the image
    centers: int, optional
        The number of centers to use for applying th K-Means algorithm. It also corresponds to the number of main region
        of colours that will be found (default: 3)

    Returns
    -------
    segmented_image: ndarray
        The segmented grayscale version of `image`, where the region of different colours are line-spaced with values
        between 0 and 255
    """
    # Set the criteria and the flags
    criteria = (_cv.TERM_CRITERIA_EPS + _cv.TERM_CRITERIA_MAX_ITER, 10, 1.)
    flags = _cv.KMEANS_RANDOM_CENTERS

    # Transform the image in a flat float32 array
    z = _np.copy(image)
    z = z[:, :, channels]
    z = z.reshape(-1, len(channels))
    z = _np.float32(z)

    # Get the labels from K-Means
    _, labels, _ = _cv.kmeans(z, centers, None, criteria, 10, flags)

    # Line-space the values of the labels between 0 and 255 for clear distinction
    labels_map = list(_np.linspace(0, 255, num=centers, dtype=_np.uint8))
    new_labels = _np.copy(labels)
    for i, l in enumerate(_np.unique(labels)):
        new_labels[labels == l] = labels_map[i]

    # Get an image segmented according to K-Means
    segmented_image = new_labels.reshape(image.shape[0], image.shape[1])
    return segmented_image.astype(_np.uint8)


def get_gaussian_mixture_segmented_image(image: _np.ndarray, channels: _List[int], components: int = 3,
                                         seed: int = 42) -> _np.ndarray:
    """
    Function that applies the Gaussian Mixture algorithm to a given image in order to quantize the main colours and
    segment the image accordingly.

    It returns a grayscale segmented version of the image where the regions of the different colours are line-spaced
    with values between 0 and 255.

    Parameters
    ----------
    image: ndarray
        The colour image to segment
    channels: Tuple[int, ...]
        The colour channels to use for quantizing the image
    components: int, optional
        The number of components to find. It corresponds to the number of main region of colours that will be found
        (default: 3)
    seed: int, optional
        The seed to set for the random number generator of the algorithm (default: 42)

    Returns
    -------
    segmented_image: ndarray
        The segmented grayscale version of `image`, where the region of different colours are line-spaced with values
        between 0 and 255
    """
    # Get the main channels and reshape the image
    img_r = image[:, :, channels]
    img_r = img_r.reshape(-1, len(channels))

    # Get the Gaussian Mixture model and fit it on the image
    gm_model = _GaussianMixture(n_components=components, covariance_type='tied', random_state=seed).fit(img_r)

    # Predict the main colours of the image through the model
    gmm_labels = gm_model.predict(img_r)

    # Line-space the values of the labels between 0 and 255 for clear distinction
    labels = gmm_labels.reshape(image.shape[0], image.shape[1])
    labels_map = list(_np.linspace(0, 255, num=components, dtype=_np.uint8))
    new_labels = _np.copy(labels)
    for i, l in enumerate(_np.unique(labels)):
        new_labels[labels == l] = labels_map[i]

    # Get an image segmented according to the Gaussian Mixture model
    segmented_image = new_labels.reshape(image.shape[0], image.shape[1])
    return segmented_image.astype(_np.uint8)


def get_roi_samples(roi: _np.ndarray, num_samples: int, patch_size: int, seed: int = 42) -> _List[_np.ndarray]:
    """
    Function that automatically samples a given number of squared patches from an image with a Region of Interest. The
    Region of Interest of the image are the pixels different than 0.

    Parameters
    ----------
    roi: ndarray
        The image with the Region of Interest from which the patches are extracted
    num_samples: int
        The number of samples to obtain from the Region of Interest
    patch_size: int, optional
        The size of the samples on both dimensions
    seed: int, optional
        The seed to set for the random number generator of the algorithm (default: 42)

    Returns
    -------
    roi_patches: List[ndarray]
        The obtained list of samples
    """
    # Get all patches of size (`patch_size`, `patch_size`) from the image
    patches = list(_extract_patches_2d(roi, (patch_size, patch_size), random_state=seed))
    # Extract just the patches of the ROI (namely the ones where non 0 intensity pixels are present)
    roi_patches = [p for p in patches if _np.all(p)]
    # Get index of `num_samples` randomly chosen samples
    samples_idx = _np.random.choice(_np.arange(len(roi_patches)), num_samples, replace=False)
    # Get samples based on the obtained random indices
    return [roi_patches[i] for i in samples_idx]


def get_mean_and_inverse_covariance_matrix(samples: _List[_np.ndarray], colour_space: _ColourSpace,
                                           channels: _Tuple[int, ...] = (0, 1, 2)) -> _Tuple[_np.ndarray, _np.ndarray]:
    """
    Function that computes the mean colour and the inverse covariance colour matrix by a list of samples in a specific
    colour space and using the given channels.

    Parameters
    ----------
    samples: List[ndarray]
        The samples used to compute the mean colour and the inverse covariance colour matrix
    colour_space: ColourSpace
        The colour space the samples are turned to before computing the mean colour and inverse covariance colour matrix
        on them
    channels: Tuple[int, ...], optional
        The channels of the colour space that are used to compute the mean colour and inverse covariance colour matrix
        (default: all 3)

    Returns
    -------
    mean: ndarray
        The mean colour obtained by the samples
    inverse_covariance: ndarray
        The inverse covariance colour matrix obtained by the samples
    """
    colour_space_fun = colour_space.bgr_to_colour_space
    # Get the number of channels
    channel_num = len(channels)

    # Set an array of 0s of the shape of the covariance matrix
    covariance_tot = _np.zeros((channel_num, channel_num), dtype=_np.float32)
    # Set an array of 0s of the shape of a mean vector of the color of the samples
    mean_tot = _np.zeros((channel_num,), dtype=_np.float32)

    for s in samples:
        # Turn the sample patch in the selected colour space
        s_colour_space = colour_space_fun(s)[:, :, channels]
        # Reshape the sample patch
        s_colour_space = s_colour_space.reshape(-1, channel_num)
        # Obtain the covariance matrix and the mean for the patch
        cov, mean = _cv.calcCovarMatrix(s_colour_space, None, _cv.COVAR_NORMAL + _cv.COVAR_ROWS + _cv.COVAR_SCALE)
        # Add the obtained mean and the covariance to the ones of the previous patches
        covariance_tot = _np.add(covariance_tot, cov)
        mean_tot = _np.add(mean_tot, mean)

    # Divide the sum of means by the number of samples
    mean = mean_tot / len(samples)
    # Divide the sum of covariances by the number of samples
    covariance = covariance_tot / len(samples)

    # Invert covariance with SVD decomposition
    inv_cov = _cv.invert(covariance, _cv.DECOMP_SVD)[1]

    return mean, inv_cov


def get_mahalanobis_distance_segmented_image(image: _np.ndarray, mean: _np.ndarray,
                                             inverse_covariance_matrix: _np.ndarray, threshold: float,
                                             channels: _Tuple[int, ...] = (0, 1, 2)) -> _np.ndarray:
    """
    Function that computes the "Mahalanobis distance" of each pixel of an image from a certain colour according to its
    mean and its inverse covariance matrix. It then sets at 0 all pixels that have a distance farther or equal to a
    certain threshold and at 255 all pixels closer than threshold. It finally returns the obtained mask.

    Parameters
    ----------
    image: ndarray
        The image in a certain colour space from which the distance mask according to the "Mahalanobis distance" is
        computed
    mean: ndarray
        The mean of the colour from which the "Mahalanobis distance" is computed
    inverse_covariance_matrix: ndarray
        The inverse covariance matrix of the colour from which the "Mahalanobis distance" is computed
    threshold: float
        Threshold used to set to 0 all pixels that have a "Mahalanobis distance" farther or equal to it and at 255 all
        pixels with a distance lower than its value.
    channels: Tuple[int, ...], optional
        The channels of the colour space that are used to compute the "Mahalanobis distance" of the image (default: all
        3)

    Returns
    -------
    mask: ndarray
        Mask of the pixels of the image with "Mahalanobis distance" lower than threshold
    """
    # Get the number of channels of the image
    channel_num = len(channels)
    # Turn the mage in the selected colour space and get the requested channels
    image = image[:, :, channels]

    # Flatten the image and change the type to `float64`
    img_flattened = image.reshape(-1, channel_num)
    img_flattened = img_flattened.astype(_np.float64)

    # Get the mahalanobis distance of each pixel
    img_distances = _cdist(img_flattened, mean, metric='mahalanobis', VI=inverse_covariance_matrix)

    # Reshape the flattened distance image
    img_distances = img_distances.reshape(image.shape[0], image.shape[1])

    # Obtain a distance mask where pixels more distant than `threshold` are masked
    mask = _np.copy(img_distances).astype(_np.uint8)
    mask[img_distances >= threshold] = 0
    mask[img_distances < threshold] = 255
    return mask


def plot_mahalanobis_segmentation_comparisons(preprocessed_image: _np.ndarray, display_image: _np.ndarray,
                                              mean: _np.ndarray, inverse_covariance_matrix: _np.ndarray,
                                              thresholds: _List[float], title: str,
                                              channels: _Tuple[int, ...] = (0, 1, 2)) -> None:
    """
    Function that plots the comparison of the "Mahalanobis distance" colour segmentation of an image given different
    thresholds.

    Parameters
    ----------
    preprocessed_image: ndarray
        The image in a certain colour space from which the distance mask according to the "Mahalanobis distance" is
        computed
    display_image: ndarray
        The image used to display the distance mask highlighted over it
    mean: ndarray
        The mean of the colour from which the "Mahalanobis distance" is computed
    inverse_covariance_matrix: ndarray
        The inverse covariance matrix of the colour from which the "Mahalanobis distance" is computed
    thresholds: List[float]
        List of thresholds to compare.
    title: str
        Title of the plot
    channels: Tuple[int, ...], optional
        The channels of the colour space that are used to compute the "Mahalanobis distance" of the image (default: all
        3)
    """
    highlighted_rois = []

    for threshold in thresholds:
        mask = get_mahalanobis_distance_segmented_image(preprocessed_image, mean, inverse_covariance_matrix, threshold,
                                                        channels)
        highlighted_rois.append(_get_highlighted_roi_by_mask(display_image, mask))

    _plot_image_grid(highlighted_rois, [f'Detected pixels for threshold {t}' for t in thresholds], title)


def get_fruit_class(image: _np.ndarray, means: _List[_np.ndarray], inverse_covariance_matrices: _List[_np.ndarray],
                    channels: _Tuple[int, ...] = (1, 2, 3), threshold: float = 3,
                    display_image: _np.ndarray = None) -> int:
    """
    Function that computes the class of a fruit given the mean colour and inverse covariance colour matrix of each class
    in two lists and displays the found pixel for each class if a display image is given.

    The classes are ordinally sorted from 0 onwards and the corresponding colour means and inverse covariance colour
    matrices are obtained at the corresponding class index in the respective lists (e.g.: `means[0]` and
    `inverse_covariance_matrices[0]` correspond to the colour mean and inverse covariance colour matrix of the fruit
    with class 0)

    The number of classes is inferred by the length of the means and inverse covariance matrices lists (e.g.: if the two
    lists have length 3, there are 3 fruit classes, namely: 0; 1 and 2)

    The means and inverse covariance matrices should refer to the colour of the healthy part of a fruit.

    Parameters
    ----------
    image: ndarray
        The image of a fruit from which the class is computed
    means: List[ndarray]
        The mean of the colour from which the "Mahalanobis distance" is computed
    inverse_covariance_matrices: List[ndarray]
        The inverse covariance matrix of the colour from which the "Mahalanobis distance" is computed
    channels: Tuple[int, ...], optional
        The channels of the colour space that are used to compute the "Mahalanobis distance" of the image (default: all
        3)
    threshold: float, optional
        Threshold to compute the fruit class according to the colour distance from its healthy part. Pixels of the
        colour image having a Mahalanobis distance greater than it are not considered part of the corresponding healthy
        fruit region (default: 3)
    channels: Tuple[int, ...], optional
        The channels of the colour space that are used to compute the "Mahalanobis distance" of the image (default: all
        3)
    display_image: ndarray, optional
        The image used to display the found pixels of each class in `image` (default: None)

    Returns
    -------
    class: int
        The class of the fruit in the image
    """
    assert len(means) == len(inverse_covariance_matrices), \
        'The list of colour means and inverse covariance colour matrices must be of the same length'

    # Get the masked distance matrices of each class and keep the pixels with distance less than 3
    masks = [get_mahalanobis_distance_segmented_image(image, m, inv_cov, threshold, channels)
             for m, inv_cov in zip(means, inverse_covariance_matrices)]

    # Display the found pixels per class if the display image is provided
    if display_image is not None:
        _plot_image_grid([_get_highlighted_roi_by_mask(display_image, m) for m in masks],
                        [f'Pixels of class {idx}' for idx in range(len(masks))],
                        'Detected pixels for each class')

    # Count the number of found pixels per class
    counts = [_np.count_nonzero(m) for m in masks]

    # Return the class with more found pixels
    return int(_np.argmax(counts))
