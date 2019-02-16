"""
Model for the Oxford Pet dataset.
"""

import tensorflow as tf

from base.base_model import BaseModel


class OxfordPetModel(BaseModel):
    def __init__(self, data_loader, config):
        super().__init__(config)
        # Get the data_generators to make the joint of the inputs in the graph
        self.data_loader = data_loader

        # define some important variables
        self.x = None
        self.y = None
        self.is_training = None
        self.out_argmax = None

        self.loss = None
        self.acc = None
        self.optimizer = None
        self.train_step = None

        self.build_model()
        self.init_saver()

    def build_model(self):

        """
        Helper Variables
        """
        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        self.global_step_inc = self.global_step_tensor.assign(self.global_step_tensor + 1)
        self.global_epoch_tensor = tf.Variable(0, trainable=False, name='global_epoch')
        self.global_epoch_inc = self.global_epoch_tensor.assign(self.global_epoch_tensor + 1)

        """
        Inputs to the network
        """
        with tf.variable_scope('inputs'):
            self.x, self.y = self.data_loader.get_inputs()

            assert self.x.get_shape().as_list() == [None, self.config.image_size, self.config.image_size, 3]

            self.is_training = tf.placeholder(tf.bool, name='Training_flag')

        tf.add_to_collection('inputs', self.x)
        tf.add_to_collection('inputs', self.y)
        tf.add_to_collection('inputs', self.is_training)

        """
        Network Architecture
        """
        with tf.variable_scope('network'):
            with tf.variable_scope('xception'):
                # begin with pre-trained Xception network
                self.xception = tf.keras.applications.xception.Xception(
                    include_top=False, pooling='avg')

                # train only the Exit Flow block of Xception network
                self.xception.trainable = False

            with tf.variable_scope('out'):
                self.xception_output = self.xception(self.x)
                self.dense = tf.layers.Dense(self.config.num_classes)
                self.out = self.dense(self.xception_output)
                tf.add_to_collection('out', self.out)

        """
        Some operators for the training process
        """
        with tf.variable_scope('predictions'):
            self.predictions = tf.argmax(self.out, 1, name='predictions')
            tf.add_to_collection('predictions', self.predictions)

        with tf.variable_scope('metrics'):
            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.out)
            self.acc = tf.reduce_mean(
                tf.cast(tf.equal(self.y, self.predictions), tf.float32))

        with tf.variable_scope('train_step'):
            self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

        tf.add_to_collection('train', self.train_step)
        tf.add_to_collection('train', self.loss)
        tf.add_to_collection('train', self.acc)

    def init_saver(self):
        """
        initialize the tensorflow saver that will be used in saving the checkpoints.
        :return:
        """
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep, save_relative_paths=True)

