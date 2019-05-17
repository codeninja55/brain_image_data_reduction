# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# brain_image_data_reduction
# autoencoder_keras.py
# 
# Attributions: 
# [1] 
# ----------------------------------------------------------------------------------------------------------------------

__author__ = 'Andrew Che <@codeninja55>'
__copyright__ = 'Copyright (C) 2019, Andrew Che <@codeninja55>'
__credits__ = ['']
__license__ = '{license}'
__version__ = '{mayor}.{minor}.{rel}'
__maintainer__ = 'Andrew Che'
__email__ = 'andrew@neuraldev.io'
__status__ = '{dev_status}'
__date__ = '2019.05.15'

"""autoencoder_keras.py: 

{Description}
"""

from keras.optimizers import SGD
from keras.layers import Dense, Input
from keras.models import Model


class Autoencoder(object):

    def __init__(self, input_dim, hidden_dim, epoch=250, learning_rate=0.001):
        self.alpha = learning_rate
        self.epoch = epoch

        input_layer = Input(shape=(input_dim,))
        # mapping_layer = Dense(input_dim)(input_layer)
        # print(mapping_layer)
        demapping_layer = Input(shape=(hidden_dim,))
        hidden_layer = Dense(hidden_dim, activation='tanh')(input_layer)

        output_layer = Dense(input_dim, activation='tanh')(hidden_layer)

        self._autoencoder_model = Model(input_layer, output_layer)
        self._encoder_model = Model(input_layer, hidden_layer)
        tmp_decoder_layer = self._autoencoder_model.layers[-1]
        self._decoder_model = Model(demapping_layer, tmp_decoder_layer(demapping_layer))

        sgd = SGD(lr=self.alpha)
        adam = SGD(lr=self.alpha)

        self._autoencoder_model.compile(
            optimizer=adam,
            loss='binary_crossentropy'
        )

    def train(self, input_train, input_test, batch_size=32):
        """
        Train the autoencoder.
        :param input_train: Numpy array of training data (if the model has a single input), or list of Numpy arrays
        (if the model has multiple inputs). If input layers in the model are named, you can also pass a dictionary
        mapping input names to Numpy arrays. x can be None (default) if feeding from framework-native tensors
        (e.g. TensorFlow data tensors).
        :param input_test: Numpy array of target (label) data (if the model has a single output), or list of Numpy
        arrays (if the model has multiple outputs). If output layers in the model are named, you can also pass a
        dictionary mapping output names to Numpy arrays. y can be None (default) if feeding from framework-native
        tensors (e.g. TensorFlow data tensors).
        :param batch_size: Number of samples per gradient update. If unspecified, batch_size will default to 32.
        :return:
        None
        """
        self._autoencoder_model.fit(
            x=input_train,
            y=input_train,
            epochs=self.epoch,
            batch_size=batch_size,
            shuffle=True,
            verbose=1,
            validation_data=(
                input_test,
                input_test
            )
        )

    def summary(self):
        return self._autoencoder_model.summary()

    def get_encoded_image(self, image):
        encoded_image = self._encoder_model.predict(image, batch_size=16, verbose=1)
        return encoded_image

    def get_decoded_image(self, encoded_imgs):
        decoded_image = self._decoder_model.predict(encoded_imgs)
        return decoded_image
