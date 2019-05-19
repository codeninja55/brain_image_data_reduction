# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# brain_image_data_reduction
# fmri_test.py
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

"""fmri_test.py: 

{Description}
"""

# %%
import sys
import os
from typing import Tuple

import keras
import numpy as np
from collections import Counter
import scipy.io as sio
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from utils.data import load_fmri
from models.classifier import FMRIClassifier, fmri_classifier_run
from utils.helpers import prompt_continue

RANDOM_SEED = 3000


def get_base_classifier_data(scaling=True):
    # %% # ===== # LOAD DATA # ===== #
    raw_data_x, raw_data_y = load_fmri()
    print('Raw data shape - x: {}, y: {}'.format(raw_data_x[0].shape, raw_data_y[0].shape))

    # %% # ===== # PREPROCESSING # ===== #
    np.random.seed(RANDOM_SEED)
    # concatenate all subjects randomly except 1
    subject_choices = [x for x in np.arange(6)]
    rand_idx = np.random.randint(0, len(subject_choices))
    test_subject = subject_choices.pop(rand_idx)

    # concatenate all the random choices of subjects to train on
    X = raw_data_x[0]
    y = raw_data_y[0]
    for i in subject_choices[1:len(subject_choices)]:
        X = np.append(X, raw_data_x[i], axis=0)
        y = np.append(y, raw_data_y[i], axis=0)

    y = y.reshape(-1)
    X_test = raw_data_x[test_subject]
    y_test = raw_data_y[test_subject].reshape(-1)

    X = X.astype(np.float16)
    X_test = X_test.astype(np.float16)

    # %%
    # from keras.utils import to_categorical
    # y_train_encoded = to_categorical(y)
    # y_test_encoded = to_categorical(y_test)
    #
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y)
    y_test_encoded = le.fit_transform(y_test)

    if scaling:
        # Normalization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X)
        X_test_scaled = scaler.fit_transform(X_test)

        X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train_encoded,
                                                          train_size=0.7, random_state=RANDOM_SEED)
        X_test = X_test_scaled
        y_test = y_test_encoded
        print('Training and Test data sets have been scaled.')
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y_train_encoded, train_size=0.7, random_state=RANDOM_SEED)
        y_test = y_test_encoded

    print('Training set shape - x: {}, y: {}'.format(X_train.shape, y_train.shape))
    print('Training eval shape - x: {}, y: {}'.format(X_val.shape, y_val.shape))
    print('Testing data shape - x: {}, y: {}'.format(X_test.shape, y_test.shape))

    # print('Training set class distribution:\n 1: {} \n 2: {}'.format(*Counter(y_train).values()))
    # print('Evaluation set class distribution:\n 1: {} \n 2: {}'.format(*Counter(y_val).values()))
    # print('Testing set class distribution:\n 1: {} \n 2: {}'.format(*Counter(y_test).values()))

    del X, y
    del raw_data_x, raw_data_y

    return X_train, y_train, X_val, y_val, X_test, y_test


def base_classifier_test(scaling=True, verbose=0):
    # %% # ===== # TRAINING AND CLASSIFICATION # ===== #

    X_train, y_train, X_val, y_val, X_test, y_test = get_base_classifier_data(scaling=scaling)

    fmri_classifier_run(
        input_data=X_train,
        input_labels=y_train,
        validation_data=X_val,
        validation_labels=y_val,
        test_data=X_test,
        test_labels=y_test,
        verbose=verbose
    )
