"""
Example of Preprocessing pipeline for Oxford Pet dataset.

All required preprocessing is done beforehand, but you should
read this file to learn how preprocessing was done and see it
applied to an example image.

Functions used for preprocessing are:
    - skimage.transform.resize for image resizing
    - np.flip for horizontal flipping of the image
    - pca_color_augmentation (defined below)

Every preprocessed image has some postfix added to its name.
Postfix convention:
    "_f" - flipped image
    "_ca" - applied PCA color augmentation
Multiple postfixes can be combined, e.g. "_f_ca"
"""

import numpy as np

from skimage.io import imread, imsave
from skimage.transform import resize

from os.path import split, join

import sys
sys.path.extend(['../..'])

from utils.utils import get_args
from utils.config import process_config


def pca_color_augmentation(input_image, scale=0.2):
    """
    Apply PCA color augmentation to input image and
    return transformed image.
    """
    renorm_image = input_image.reshape(-1, 3)
    renorm_image = renorm_image.astype('float32')

    mean = np.mean(renorm_image, axis=0)
    std = np.std(renorm_image, axis=0)
    renorm_image = (renorm_image - mean) / std
    
    cov = np.cov(renorm_image, rowvar=False)
    
    lambdas, p = np.linalg.eig(cov)
    alphas = np.random.normal(0, scale, 3)
    
    delta = np.dot(p, alphas*lambdas)

    pca_augmentation = renorm_image + delta
    pca_color_image = pca_augmentation * std + mean
    pca_color_image = np.maximum(
        np.minimum(pca_color_image, 255), 0).astype('uint8')
    return pca_color_image.reshape(299, 299, 3)


def main(config):
    """
    Preprocessing pipeline for an example image.
    All functions and naming conventions used here
    are defined and described above.
    """
    dir, filename = split(config.path_to_example)

    original_image = imread(join(dir, filename))
    resized_image = resize(original_image, (299, 299),
        mode='reflect', preserve_range=True).astype('uint8')

    flipped_image = np.flip(resized_image, 1)
    color_augmented_image = pca_color_augmentation(resized_image)
    flipped_color_augmented_image = pca_color_augmentation(flipped_image)

    filename = filename.replace('.jpg', '')

    imsave(join(dir, filename + '.png'), resized_image)
    imsave(join(dir, filename + '_f.png'), flipped_image)
    imsave(join(dir, filename + '_ca.png'), color_augmented_image)
    imsave(join(dir, filename + '_f_ca.png'), flipped_color_augmented_image)


if __name__ == '__main__':
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
        main(config)

    except Exception as e:
        print('Missing or invalid arguments %s' % e)
