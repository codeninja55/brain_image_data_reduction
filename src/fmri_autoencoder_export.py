# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# brain_image_data_reduction
# fmri_autoencoder_test.py
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

"""fmri_autoencoder_test.py: 

{Description}
"""

import sys
import os

import scipy
import numpy as np
from collections import Counter

# TO NOT USE GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras import backend as K
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from models.autoencoder_keras import Autoencoder
from models.autoassociative import Autoassociative
from models.classifier import fmri_classifier_run
from utils.data import load_fmri_dict


# %% # ===== # LOADING DATA # ===== #

raw_data = load_fmri_dict()
# raw_data_x, raw_data_y = load_smaller_fmri()
# print('Raw data shape - x: {}, y: {}'.format(raw_data_x[0].shape, raw_data_y[0].shape))

# %% # ===== # PREPROCESSING # ===== #
RANDOM_SEED = 14000604

for subject in raw_data:
    np.random.seed(RANDOM_SEED)

    # Optimise by reducing float size for TF on GPU
    X = np.array(subject['data'], dtype=np.float32)
    y = np.array(subject['labels']).ravel()

    # %% # ===== # DATA ENCODING # ===== #

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y)

    print('Dimensionality Reduction')
    bottleneck_dim = 600
    print('Bottleneck Dimensions: {}'.format(bottleneck_dim))

    autoencoder = Autoencoder(X.shape[1], bottleneck_dim, epoch=500, learning_rate=1e-4)
    # autoencoder.plot_model()
    autoencoder.summary()
    # prompt_continue('Do you want to continue with the autoassociative encoding of this model?')

    autoencoder.train(X, X, batch_size=16)

    export_X = autoencoder.get_encoded_image(X)

    np.savetxt('data/encoded_fmri_X-' + subject['subject'] + '.csv', export_X, delimiter=',')
    np.savetxt('data/encoded_fmri_y-' + subject['subject'] + '.csv', y, delimiter=',')

    K.clear_session()
    del autoencoder
    import gc

    gc.collect()
