# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# brain_image_data_reduction
# autoencoder_conv.py
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
__date__ = '2019.05.17'

"""autoencoder_conv.py: 

{Description}
"""

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model


class Autoencoder(object):

    def __init__(self):
        # Encoding
        input_layer = Input(shape=(28, 28, 1))
        encoding_conv_layer_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
        encoding_pooling_layer_1 = MaxPooling2D((2, 2), padding='same')(encoding_conv_layer_1)
        encoding_conv_layer_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoding_pooling_layer_1)
        encoding_pooling_layer_2 = MaxPooling2D((2, 2), padding='same')(encoding_conv_layer_2)
        encoding_conv_layer_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoding_pooling_layer_2)
        code_layer = MaxPooling2D((2, 2), padding='same')(encoding_conv_layer_3)

        # Decoding
        decodging_conv_layer_1 = Conv2D(8, (3, 3), activation='relu', padding='same')(code_layer)
        decodging_upsampling_layer_1 = UpSampling2D((2, 2))(decodging_conv_layer_1)
        decodging_conv_layer_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(decodging_upsampling_layer_1)
        decodging_upsampling_layer_2 = UpSampling2D((2, 2))(decodging_conv_layer_2)
        decodging_conv_layer_3 = Conv2D(16, (3, 3), activation='relu')(decodging_upsampling_layer_2)
        decodging_upsampling_layer_3 = UpSampling2D((2, 2))(decodging_conv_layer_3)
        output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decodging_upsampling_layer_3)

        self._model = Model(input_layer, output_layer)
        self._model.compile(optimizer='adadelta', loss='binary_crossentropy')

    def train(self, input_train, input_test, batch_size, epochs):
        self._model.fit(input_train,
                        input_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(
                            input_test,
                            input_test))

    def get_decoded_image(self, encoded_imgs):
        decoded_image = self._model.predict(encoded_imgs)
        return decoded_image


