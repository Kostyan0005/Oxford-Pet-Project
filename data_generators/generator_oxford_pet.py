"""
Input pipeline for Oxford Pet dataset.
"""

import sys
sys.path.extend(['..'])

import numpy as np
import tensorflow as tf

from utils.utils import get_args
from utils.config import process_config


class OxfordPetTfLoader:
    def __init__(self, config):
        self.config = config

        self.train_filenames = np.load(self.config.data_dir + 'train_filenames.npy')
        self.train_filenames = self.config.data_dir + 'train_images/' \
            + self.train_filenames + '.png'
        self.dev_filenames = np.load(self.config.data_dir + 'dev_filenames.npy')
        self.dev_filenames = self.config.data_dir + 'dev_images/' \
            + self.dev_filenames + '.png'
        self.test_filenames = np.load(self.config.data_dir + 'test_filenames.npy')
        self.test_filenames = self.config.data_dir + 'test_images/' \
            + self.test_filenames + '.png'
        self.train_labels = np.load(self.config.data_dir + 'train_labels.npy')
        self.dev_labels = np.load(self.config.data_dir + 'dev_labels.npy')
        self.test_labels = np.load(self.config.data_dir + 'test_labels.npy')

        # Check lens
        assert self.train_filenames.shape[0] == self.train_labels.shape[0], \
        "Train filenames and labels should have same length"
        assert self.dev_filenames.shape[0]  == self.dev_labels.shape[0], \
        "Dev filenames and labels should have same length"
        assert self.test_filenames.shape[0]  == self.test_labels.shape[0], \
        "Test filenames and labels should have same length"

        # Define datasets sizes
        self.train_size = self.train_filenames.shape[0]
        self.dev_size = self.dev_filenames.shape[0]
        self.test_size = self.test_filenames.shape[0]

        # Define number of iterations per epoch
        self.num_iterations_train = (self.train_size + self.config.batch_size - 1) \
            // self.config.batch_size
        self.num_iterations_dev  = (self.dev_size  + self.config.batch_size - 1) \
            // self.config.batch_size
        self.num_iterations_test  = (self.test_size  + self.config.batch_size - 1) \
            // self.config.batch_size

        self.features_placeholder = None
        self.labels_placeholder = None

        self.dataset = None
        self.iterator = None
        self.init_iterator_op = None
        self.next_batch = None

        self._build_dataset_api()

    @staticmethod
    def _parse_function(filename, label, size):
        """Obtain the image from the filename (for both training and validation).

        The following operations are applied:
            - Decode the image from png format
            - Convert to float and to range [0, 1]
        """
        image_string = tf.read_file(filename)

        image_decoded = tf.image.decode_png(image_string, channels=3)

        # This will convert to float values in [0, 1]
        image = tf.image.convert_image_dtype(image_decoded, tf.float32)

        if image.get_shape().as_list() != [size, size, 3]:
            image = tf.image.resize_images(image, [size, size])

        return image, label


    def _build_dataset_api(self):
        with tf.device('/cpu:0'):
            self.features_placeholder = tf.placeholder(tf.string, [None, ])
            self.labels_placeholder = tf.placeholder(tf.int64, [None, ])
            self.mode_placeholder = tf.placeholder(tf.string, shape=())

            # Create a Dataset serving batches of images and labels
            # We don't repeat for multiple epochs because we always train and evaluate for one epoch
            parse_fn = lambda f, l: self._parse_function(f, l, self.config.image_size)

            self.dataset = (tf.data.Dataset.from_tensor_slices(
                    (self.features_placeholder, self.labels_placeholder)
                )
                .map(parse_fn, num_parallel_calls=self.config.num_parallel_calls)
                .batch(self.config.batch_size)
                .prefetch(1)  # make sure you always have one batch ready to serve
            )

            # Create reinitializable iterator from dataset
            self.iterator = self.dataset.make_initializable_iterator()

            self.iterator_init_op = self.iterator.initializer

            self.next_batch = self.iterator.get_next()

    # There are 3 possible modes: 'train', 'dev', 'test'
    def initialize(self, sess, mode='train'):
        if mode == 'train':
            sess.run(self.iterator_init_op, feed_dict={
                self.features_placeholder: self.train_filenames,
                self.labels_placeholder: self.train_labels,
                self.mode_placeholder: mode})
        elif mode == 'dev':
            sess.run(self.iterator_init_op, feed_dict={
                self.features_placeholder: self.dev_filenames,
                self.labels_placeholder: self.dev_labels,
                self.mode_placeholder: mode})
        elif mode == 'test':
            sess.run(self.iterator_init_op, feed_dict={
                self.features_placeholder: self.test_filenames,
                self.labels_placeholder: self.test_labels,
                self.mode_placeholder: mode})


    def get_inputs(self):
        return self.next_batch


def main(config):
    """
    Function to test from console, accepts config object as an argument
    """
    tf.reset_default_graph()
    sess = tf.Session()

    data_loader = OxfordPetTfLoader(config)

    images, labels = data_loader.get_inputs()

    print('Train')
    data_loader.initialize(sess, mode='train')

    out_im, out_l = sess.run([images, labels])

    print(out_im.shape, out_im.dtype)
    print(out_l.shape, out_l.dtype)

    print('Dev')
    data_loader.initialize(sess, mode='dev')

    out_im, out_l = sess.run([images, labels])

    print(out_im.shape, out_im.dtype)
    print(out_l.shape, out_l.dtype)

    print('Test')
    data_loader.initialize(sess, mode='test')

    out_im, out_l = sess.run([images, labels])

    print(out_im.shape, out_im.dtype)
    print(out_l.shape, out_l.dtype)


if __name__ == '__main__':
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
        main(config)

    except Exception as e:
        print('Missing or invalid arguments %s' % e)
