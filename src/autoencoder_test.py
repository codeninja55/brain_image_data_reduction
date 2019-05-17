# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# brain_image_data_reduction
# autoencoder_test.py
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
__date__ = '2019.05.16'

"""autoencoder_test.py: 

{Description}
"""

# %%
# ===== # IMPORTS # ===== #
import os
import sys

from sklearn import datasets

import numpy as np
import matplotlib.pyplot as plt

# nb_dir = os.path.split(os.getcwd())[0]
# models_dir = os.path.abspath(os.path.join(nb_dir, 'models'))
# if nb_dir not in sys.path:
#     sys.path.append(nb_dir)
# if models_dir not in sys.path:
#     sys.path.append(models_dir)

from tensorflow.keras.datasets import fashion_mnist
from models.autoencoder_keras import Autoencoder

# %%
# ===== # IMPORT DATA # ===== #
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print('Train dataset shape: ', x_train.shape)
print('Train labels shape: ', y_train.shape)
print('Test dataset shape: ', x_test.shape)


# %%
# ===== # PREPROCESS # ===== #
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print('Train dataset reshaped: ', x_train[1].shape)
print('Test dataset reshaped: ', x_test.shape)


# %%
# ===== # KERAS IMPLEMENTATION # ===== #
autoencoder = Autoencoder(x_train.shape[1], 32)
autoencoder.train(x_train, x_test, 256)
encoded_imgs = autoencoder.get_encoded_image(x_test)
decoded_imgs = autoencoder.get_decoded_image(encoded_imgs)


# %%
# ===== # KERAS EVALUATION # ===== #
plt.figure(figsize=(20, 4))
for i in range(10):
    # Original
    subplot = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    subplot.get_xaxis().set_visible(False)
    subplot.get_yaxis().set_visible(False)

    # Reconstruction
    subplot = plt.subplot(2, 10, i + 11)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    subplot.get_xaxis().set_visible(False)
    subplot.get_yaxis().set_visible(False)
plt.show()

