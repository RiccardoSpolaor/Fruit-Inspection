import cv2 as cv
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.mixture import GaussianMixture
from typing import List, Tuple

from .graphics import ColourSpace, get_highlighted_roi_by_mask, plot_image_grid


def get_k_means_quantized_image(image: np.ndarray, channels: Tuple[int, ...] = (0, 1, 2), 
                                centers: int = 3) -> np.ndarray:
    # Set the criteria and the flags
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.)
    flags = cv.KMEANS_RANDOM_CENTERS

    # Transform the image in a flat float32 array
    z = np.copy(image)
    z = z[:, :, channels]
    z = z.reshape(-1, len(channels))
    z = np.float32(z)

    # Get the labels from K-Means
    _, labels, _ = cv.kmeans(z, centers, None, criteria, 10, flags)

    # Line-space the values of the labels between 0 and 255 for clear distinction
    labels_map = list(np.linspace(0, 255, num=centers, dtype=np.uint8))
    new_labels = np.copy(labels)
    for i, l in enumerate(np.unique(labels)):
        new_labels[labels == l] = labels_map[i]

    # Get an image segmented according to K-Means
    res = new_labels.reshape(image.shape[0], image.shape[1])
    return res.astype(np.uint8)


def get_gaussian_mixture_quantized_image(image: np.ndarray, channels: List[int], components: int = 3,
                                         seed: int = 42) -> np.ndarray:
    img_r = image[:, :, channels]
    img_r = img_r.reshape(-1, len(channels))

    gm_model = GaussianMixture(n_components=components, covariance_type='tied', random_state=seed).fit(img_r)

    gmm_labels = gm_model.predict(img_r)

    labels = gmm_labels.reshape(image.shape[0], image.shape[1])

    labels_map = list(np.linspace(0, 255, num=components, dtype=np.uint8))

    new_labels = np.copy(labels)

    for i, l in enumerate(np.unique(labels)):
        new_labels[labels == l] = labels_map[i]

    res = new_labels.reshape(image.shape[0], image.shape[1])
    return res.astype(np.uint8)


def get_samples(roi: np.ndarray, num_samples: int, patch_size: int, seed: int = 42) -> List[np.ndarray]:
    # Get all patches of size (`patch_size`, `patch_size`) from the image
    patches = list(extract_patches_2d(roi, (patch_size, patch_size), random_state=seed))
    # Extract just the patches of the ROI (namely the ones where non 0 intensity pixels are present)
    roi_patches = [p for p in patches if np.all(p)]
    # Get index of `num_samples` randomly chosen samples
    samples_idx = np.random.choice(np.arange(len(roi_patches)), num_samples, replace=False)
    # Get samples based on the obtained random indices
    return [roi_patches[i] for i in samples_idx]


def get_mean_and_inverse_covariance_matrix(samples: List[np.ndarray], colour_space: ColourSpace,
                                           channels: Tuple[int, ...] = (0, 1, 2)) -> (np.ndarray, np.ndarray):
    colour_space_fun = colour_space.bgr_to_color_space
    # Get the number of channels
    channel_num = len(channels)

    # Set an array of 0s of the shape of the covariance matrix
    covariance_tot = np.zeros((channel_num, channel_num), dtype=np.float32)
    # Set an array of 0s of the shape of a mean vector of the color of the samples
    mean_tot = np.zeros((channel_num,), dtype=np.float32)

    for s in samples:
        s_colour_space = colour_space_fun(s)
        # Turn the sample patch in the selected colour space
        s_colour_space = colour_space_fun(s)[:, :, channels]
        # Reshape the sample patch
        s_colour_space = s_colour_space.reshape(-1, channel_num)
        # Obtain the covariance matrix and the mean for the patch
        cov, mean = cv.calcCovarMatrix(s_colour_space, None, cv.COVAR_NORMAL + cv.COVAR_ROWS + cv.COVAR_SCALE)
        # Add the obtained mean and the covariance to the ones of the previous patches
        covariance_tot = np.add(covariance_tot, cov)
        mean_tot = np.add(mean_tot, mean)

    # Divide the sum of means by the number of samples
    mean = mean_tot / len(samples)
    # Divide the sum of covariances by the number of samples
    covariance = covariance_tot / len(samples)

    return mean, np.linalg.inv(covariance)


def get_mahalanobis_distance_img(img: np.ndarray, mean: np.ndarray, inverse_covariance_matrix: np.ndarray,
                                 threshold: float, channels: Tuple[int, ...] = (0, 1, 2)) -> np.ndarray:
    # Get the number of channels of the image
    channel_num = len(channels)
    # Turn the mage in the selected colour space and get the requested channels
    img = img[:, :, channels]

    # Flatten the image and change the type to `float64`
    img_flattened = img.reshape(-1, channel_num)
    img_flattened = img_flattened.astype(np.float64)

    img_distances = cdist(img_flattened, mean, metric='mahalanobis', VI=inverse_covariance_matrix)

    img_distances = img_distances.reshape(img.shape[0], img.shape[1])

    mask = np.copy(img_distances).astype(np.uint8)

    mask[img_distances >= threshold] = 0
    mask[img_distances < threshold] = 255

    return mask


def plot_mahalanobis_results(preprocessed_image: np.ndarray, display_image: np.ndarray, mean: np.ndarray,
                             inverse_covariance_matrix: np.ndarray, thresholds: List[float], title: str,
                             channels: Tuple[int, ...] = (0, 1, 2)) -> None:
    highlighted_rois = []

    for threshold in thresholds:
        mask = get_mahalanobis_distance_img(preprocessed_image, mean, inverse_covariance_matrix, threshold, channels)
        highlighted_rois.append(get_highlighted_roi_by_mask(display_image, mask))

    plot_image_grid(highlighted_rois, [f'Detected pixels for threshold {t}' for t in thresholds], title)


def get_fruit_class(img: np.ndarray, means: List[np.ndarray], inverse_covariance_matrices: List[np.ndarray],
                    channels: Tuple[int, ...] = (1, 2, 3), display_image: np.ndarray = None) -> int:
    masks = [get_mahalanobis_distance_img(img, m, inv_cov, 3, channels)
             for m, inv_cov in zip(means, inverse_covariance_matrices)]

    if display_image is not None:
        plot_image_grid([get_highlighted_roi_by_mask(display_image, m) for m in masks],
                        [f'Pixels of class {idx}' for idx in range(len(masks))],
                        'Detected pixels for each class')

    counts = [np.count_nonzero(m) for m in masks]

    return int(np.argmax(counts))
