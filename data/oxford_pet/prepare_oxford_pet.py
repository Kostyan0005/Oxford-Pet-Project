"""
Description of Preprocessing pipeline for Oxford Pet dataset
(the dataset be found here: http://www.robots.ox.ac.uk/~vgg/data/pets/).

1) Create the train/dev/test split using original dataset.

Taking into account the fact that dev and test set distributions
should be represantative of the whole dataset, and that the size of the 
original dataset is not so big (7349 images), I decided that dev and test
datasets should be 10% of the whole dataset each (that is 735 images each),
and of course splitting is stratified.

As soon as all preprocessing described below takes place, respective
changes are made to the training set (which now contains ~ 34k images),
and then it is shuffled.


2) Preprocess Oxford Pet dataset for use in my application.

The steps that were done for preprocessing are:
    - reshape all images into shape (299, 299, 3)
    - flip every image horizontally (x2)
    - apply PCA color augmentation to all previous transformations (x4)
This results in a preprocessed dataset which is 4 times the dataset
I started with.

Images are placed in 3 different folders:
    - train_images/ (fully preprocessed)
    - dev_images/ (only resized)
    - test_images/ (only resized)
For your convenience (e.g. if your ever decide to change train/dev/test split,
train_images/ folder actually contains all preprocessed images).

Every preprocessed image has some postfix added to its name.
Postfix convention:
    "_f" - flipped image
    "_ca" - applied PCA color augmentation
Multiple postfixes can be combined, e.g. "_f_ca"

Functions used for preprocessing are:
    - skimage.transform.resize for image resizing
    - np.flip for horizontal flipping of the image
    - pca_color_augmentation (defined below)

As for image labels, every label is a class number of an image (from 0 to 36),
as there are 37 classes (pet breeds) in the dataset.


3) Futher data augmentation.

While working on this project, I discovered that there are some problematic
pet breeds which are difficult for the model (as well as for a person) to
distinguish, and so I found some additional images of those breeds on the 
Internet (using Google Images), and added them to the dataset with all required
preprocessing.


4) All preprocessing is done beforehand.

I decided that since preprocessing took a long time to complete,
I would supply this project with the dataset that is already preprocessed
and augmented, so no further processing is required.
"""

import numpy as np
from skimage.transform import resize


def pca_color_augmentation(original_image, scale=0.2):
    """
    Apply PCA color augmentation to original image and
    return transformed image of shape (299, 299, 3)
    """
    renorm_image = original_image.reshape(-1, 3)

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


def main():
    """
    An example of importing train/dev/test image names and labels.
    All preprocessed images are stored in
    data/oxford-iiit-pet/images_preprocessed/ directory.
    """

    train_filenames = np.load('train_filenames.npy')
    dev_filenames = np.load('dev_filenames.npy')
    test_filenames = np.load('test_filenames.npy')
    train_labels = np.load('train_labels.npy')
    dev_labels = np.load('dev_labels.npy')
    test_labels = np.load('test_labels.npy')
