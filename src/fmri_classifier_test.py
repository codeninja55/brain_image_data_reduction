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
from utils.metrics import plot_confusion_matrix, plot_history
from utils.helpers import prompt_continue

# %% # ===== # LOAD DATA # ===== #
PWD_PATH = os.path.abspath(os.getcwd())
DATA_PATH = os.path.abspath(os.path.join(PWD_PATH, 'data'))
if PWD_PATH not in sys.path:
    sys.path.append(PWD_PATH)
    sys.path.append(DATA_PATH)

RANDOM_SEED = 3000

# %% # ===== # LOADING DATA # ===== #

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

X = X.astype('float32')
X_test = X_test.astype('float32')

# %%
SCALING = True

# Normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.fit_transform(X_test)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_test = le.fit_transform(y_test)


if SCALING:
    X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_encoded,
                                                      train_size=0.8, random_state=RANDOM_SEED)
    X_test = X_test_scaled
    print('Training and Test data sets have been scaled.')
else:
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, train_size=0.8, random_state=RANDOM_SEED)

print('Training set shape - x: {}, y: {}'.format(X_train.shape, y_train.shape))
print('Training eval shape - x: {}, y: {}'.format(X_val.shape, y_val.shape))
print('Testing data shape - x: {}, y: {}'.format(X_test.shape, y_test.shape))

print('Training set class distribution:\n 1: {} \n 2: {}'.format(*Counter(y_train).values()))
print('Evaluation set class distribution:\n 1: {} \n 2: {}'.format(*Counter(y_val).values()))
print('Testing set class distribution:\n 1: {} \n 2: {}'.format(*Counter(y_test).values()))


del X, y
# %% # ===== # TRAINING AND CLASSIFICATION # ===== #

model = keras.Sequential()

# First FCN hidden layer
model.add(keras.layers.Dense(
    units=50,
    input_dim=X_train.shape[1],
    activation='tanh',
    kernel_initializer='random_normal'
))
# model.add(keras.layers.Dropout(0.2))

# Second FCN hidden layer
model.add(keras.layers.Dense(10, activation='tanh'))
# model.add(keras.layers.Dropout(0.2))

# Output layer
model.add(keras.layers.Dense(1, activation='sigmoid'))

sgd = keras.optimizers.SGD(lr=1e-5, nesterov=False)
adam = keras.optimizers.adam(lr=.0001)

model.compile(
    loss='binary_crossentropy',
    metrics=['accuracy'],
    optimizer=adam
)

print(model.summary())

prompt_continue()

classifier = model.fit(
    X_train,
    y_train,
    epochs=1000,
    verbose=1,
    shuffle=True,
    batch_size=8,
    validation_data=(X_val, y_val)
)

# %% KERAS EVALUATION

pred_y = model.predict(X_test)
pred_y = (pred_y > 0.5)

score = model.evaluate(X_test, y_test, verbose=1)
print('Loss: {}'.format(score[0]))
print('Accuracy: {}'.format(score[1]))

print(classification_report(y_test, pred_y))

plot_history(classifier)
cm = confusion_matrix(y_test, pred_y)
print(cm)
plot_confusion_matrix(cm, classes=le.inverse_transform([1, 2]))

