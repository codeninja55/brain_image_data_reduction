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
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import adam, SGD
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from models.autoencoder_keras import Autoencoder
from models.autoassociative import Autoassociative
from models.classifier import fmri_classifier_run
from utils.data import load_fmri, load_smaller_fmri
from utils.metrics import plot_confusion_matrix, plot_history
from utils.helpers import prompt_continue
from fmri_classifier_test import base_classifier_test, get_base_classifier_data


# %% # ===== # LOADING DATA # ===== #

raw_data_x, raw_data_y = load_fmri()
# raw_data_x, raw_data_y = load_smaller_fmri()
print('Raw data shape - x: {}, y: {}'.format(raw_data_x[0].shape, raw_data_y[0].shape))

# %% # ===== # PREPROCESSING # ===== #
# RANDOM_SEED = 14000604

# np.random.seed(RANDOM_SEED)
# concatenate all subjects randomly except 1
subject_choices = [x for x in np.arange(6)]
rand_idx = np.random.randint(0, len(subject_choices))
test_subject = subject_choices.pop(rand_idx)

# concatenate all the random choices of subjects to train on
X_train_full = raw_data_x[0]
y_train_full = raw_data_y[0]
for i in subject_choices[1:len(subject_choices)]:
    X_train_full = np.append(X_train_full, raw_data_x[i], axis=0)
    y_train_full = np.append(y_train_full, raw_data_y[i], axis=0)

y_train_full = y_train_full.reshape(-1)
X_test = raw_data_x[test_subject]
y_test = raw_data_y[test_subject].reshape(-1)

X_full = raw_data_x.pop(0)
y_full = raw_data_y.pop(0)
for x_arr, y_arr in zip(raw_data_x, raw_data_y):
    X_full = np.append(X_full, x_arr, axis=0)
    y_full = np.append(y_full, y_arr, axis=0)

print('Dtype: {}'.format(X_full.dtype))

del raw_data_x, raw_data_y

# %% # ===== # DATA ENCODING # ===== #

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train_full)
y_test_encoded = le.fit_transform(y_test)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_encoded,
    train_size=0.7,
    # random_state=RANDOM_SEED
)

print('Training set shape - x: {}, y: {}'.format(X_train_full.shape, y_train_full.shape))
print('Training eval shape - x: {}, y: {}'.format(X_val.shape, y_val.shape))
print('Testing data shape - x: {}, y: {}'.format(X_test.shape, y_test_encoded.shape))

# print('Training set class distribution:\n 1: {} \n 2: {}'.format(*Counter(y_train).values()))
# print('Evaluation set class distribution:\n 1: {} \n 2: {}'.format(*Counter(y_val).values()))
# print('Testing set class distribution:\n 1: {} \n 2: {}'.format(*Counter(y_test).values()))

# %% # ===== # DATA ENCODING # ===== #
print('Dimensionality Reduction')
bottleneck_dim = 200
print('Bottleneck Dimensions: {}'.format(bottleneck_dim))

# autoencoder = Autoencoder(X_full.shape[1], bottleneck_dim, epoch=750, learning_rate=1e-4)
# # autoencoder.plot_model()
# autoencoder.summary()
# # prompt_continue('Do you want to continue with the autoassociative encoding of this model?')
#
# autoencoder.train(X_train_full, X_test, batch_size=8)
#
# export_X = autoencoder.get_encoded_image(X_full)
# encoded_X_train = autoencoder.get_encoded_image(X_full)
# encoded_X_val = autoencoder.get_encoded_image(X_val)
# encoded_X_test = autoencoder.get_encoded_image(X_test)
#
# print('Encoded training set shape - x: {}'.format(encoded_X_train.shape))
# print('Encoded training eval shape - x: {}'.format(encoded_X_val.shape))
# print('Encoded testing data shape - x: {}'.format(encoded_X_test.shape))

# %% %% # ===== # EXPORT # ===== #
# np.savetxt('data/encoded_fmri_X.csv', export_X, delimiter=',')
# np.savetxt('data/encoded_fmri_y.csv', y_full, delimiter=',')
# np.savetxt('data/raw_fmri_X.csv', X_train_full, delimiter=',')

# %% AUTOASSOCIATIVE TEST

autoassociative = Autoassociative(X_train_full.shape[1], bottleneck_dim, epoch=500, learning_rate=1e-3)
autoassociative.summary()

autoassociative.train(X_train_full, X_test, batch_size=1)

export_X_2 = autoassociative.get_encoded_image(X_full)
encoded_X_train_2 = autoassociative.get_encoded_image(X_train)
encoded_X_val_2 = autoassociative.get_encoded_image(X_val)
encoded_X_test_2 = autoassociative.get_encoded_image(X_test)

# %% Clean up
# import gc
# del autoassociative
K.clear_session()
# gc.collect()

# %%

np.savetxt('data/autoassociated_fmri_X.csv', export_X_2, delimiter=',')
np.savetxt('data/autoassociated_fmri_y.csv', y_full, delimiter=',')

# %%

# fmri_classifier_run(
#     input_data=export_X_2,
#     input_labels=y_full,
#     validation_data=encoded_X_val_2,
#     validation_labels=y_val,
#     test_data=encoded_X_test_2,
#     test_labels=y_test_encoded,
#     verbose=1,
#     evaluate=True
# )

# fmri_classifier_run(
#     input_data=encoded_X_train,
#     input_labels=y_train,
#     validation_data=encoded_X_val,
#     validation_labels=y_val,
#     test_data=encoded_X_test,
#     test_labels=y_test_encoded,
#     verbose=1,
#     evaluate=True
# )

# base_clf.evaluate(base_X_test, base_y_test, verbose=0)
