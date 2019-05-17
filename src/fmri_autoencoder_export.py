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

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import adam, SGD
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from models.autoencoder_keras import Autoencoder
from utils.data import load_fmri
from utils.metrics import plot_confusion_matrix, plot_history
from utils.helpers import prompt_continue

# %%
PWD_PATH = os.path.abspath(os.getcwd())
DATA_PATH = os.path.abspath(os.path.join(PWD_PATH, 'data'))
if PWD_PATH not in sys.path:
    sys.path.append(PWD_PATH)
    sys.path.append(DATA_PATH)

# %% # ===== # LOADING DATA # ===== #

raw_data_x, raw_data_y = load_fmri()
print('Raw data shape - x: {}, y: {}'.format(raw_data_x[0].shape, raw_data_y[0].shape))

# %% # ===== # PREPROCESSING # ===== #
RANDOM_SEED = 3000

np.random.seed(RANDOM_SEED)
# concatenate all subjects randomly except 1
subject_choices = [x for x in np.arange(6)]
rand_idx = np.random.randint(0, len(subject_choices))
test_subject = subject_choices.pop(rand_idx)

# concatenate all the random choices of subjects to train on
X_train_full = raw_data_x.pop(0)
y_full = raw_data_y.pop(0)
for x_arr, y_arr in zip(raw_data_x, raw_data_y):
    X_train_full = np.append(X_train_full, x_arr, axis=0)
    y_full = np.append(y_full, y_arr, axis=0)

X_train_full = X_train_full.astype('float32')
print(y_full.shape)
y_full = y_full.reshape(-1)

print('X train dataset full shape: {}'.format(X_train_full.shape))

# %% # ===== # DATA ENCODING # ===== #
SCALING = False

le = LabelEncoder()

# %% # ===== # DATA ENCODING # ===== #

print('Dimensionality Reduction')
# bottleneck_dim = int(X.shape[1] / 8)
bottleneck_dim = 200
print('Bottleneck Dimensions: {}'.format(bottleneck_dim))

autoencoder = Autoencoder(X_train_full.shape[1], bottleneck_dim)

autoencoder.summary()
prompt_continue('Do you want to continue with the autoassociative encoding of this model?')

autoencoder.train(X_train_full, X_train_full, batch_size=16)
encoded_X_train = autoencoder.get_encoded_image(X_train_full)

# %% %% # ===== # EXPORT # ===== #
np.savetxt('encoded_fmri_X.csv', encoded_X_train, delimiter=',')
np.savetxt('encoded_fmri_y.csv', y_full, delimiter=',')

# %% TESTING
test_X_load = np.loadtxt('encoded_fmri_X.csv', delimiter=',')
np.testing.assert_array_equal(test_X_load, encoded_X_train)
test_y_load = np.loadtxt('encoded_fmri_y.csv', delimiter=',')
np.testing.assert_array_equal(test_y_load, y_full)
