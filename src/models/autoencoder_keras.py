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

from keras.optimizers import SGD, Adam, Nadam, RMSprop, Adadelta
from keras.layers import Dense, Input, LeakyReLU, ReLU, Activation
from keras.models import Model
from keras import regularizers
from keras.utils import plot_model
from keras.engine.topology import Layer


class Identity(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = kwargs['input_shape'][0]
        super(Identity, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.output_dim


class Autoencoder(object):

    def __init__(self, input_dim, hidden_dim, epoch=250, learning_rate=0.001):
        self.alpha = learning_rate
        self.epoch = epoch

        # Input layer (requirement for Keras functional API to define input dimensions)
        input_layer = Input(shape=(input_dim,), name='Input_Layer')

        # Mapping layer
        mapping_layer = Activation('relu', name='Mapping_Layer')(input_layer)
        # mapping_layer = ReLU(input_shape=(hidden_dim,), name='Mapping_Layer')(input_layer)
        # mapping_layer = Identity(lambda x: x, input_shape=(input_dim,))(input_layer)

        # Bottleneck layer
        hidden_layer = Dense(
            units=hidden_dim,
            activation='relu',
            activity_regularizer=regularizers.l1(10e-5),
            name='Bottleneck_Layer')(mapping_layer)
        # hidden_activation = LeakyReLU(alpha=self.alpha, name='Hidden_Activation')(hidden_layer)

        # Demapping layer - used in decoder model
        hidden_input = Input(shape=(hidden_dim,), name='Demapping_Input_Layer')

        # Output layer
        output_layer = Dense(input_dim, activation='sigmoid', name='Output_Layer')(hidden_layer)

        # Autoassociative model used to train the network
        self._autoencoder_model = Model(input_layer, output_layer)
        # Encoder model used to get the encoded data out
        self._encoder_model = Model(input_layer, hidden_layer)
        # tmp_decoder_layer = self._autoencoder_model.layers[-1]
        # self._decoder_model = Model(hidden_input, tmp_decoder_layer(hidden_input))

        sgd = SGD(lr=self.alpha)
        nadam = Nadam(lr=self.alpha)
        adam = Adam(lr=self.alpha)
        adadelta = Adadelta(lr=self.alpha)

        self._autoencoder_model.compile(
            optimizer=adadelta,
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
            # validation_split=0.33
        )

    def summary(self):
        return self._autoencoder_model.summary()

    def plot_model(self, to_file='autoencoder_graph.png'):
        plot_model(self._autoencoder_model, to_file=to_file)

    def get_encoded_image(self, image):
        encoded_image = self._encoder_model.predict(image, batch_size=16, verbose=1)
        return encoded_image

    # def get_decoded_image(self, encoded_imgs):
    #     decoded_image = self._decoder_model.predict(encoded_imgs)
    #     return decoded_image
