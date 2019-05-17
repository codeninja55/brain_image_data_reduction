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


# %% # ===== # DATA ENCODING # ===== #
SCALING = False

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_test = le.fit_transform(y_test)

if SCALING:
    # Normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.fit_transform(X_test)

    X_train = X_train_scaled
    # X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_encoded,
    #                                                   train_size=0.8, random_state=RANDOM_SEED)
    X_test = X_test_scaled
    y_train = y_encoded
    print('Training and Test data sets have been scaled.')
else:
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, train_size=0.8, random_state=RANDOM_SEED)

print('Training set shape - x: {}, y: {}'.format(X_train.shape, y_train.shape))
# print('Training eval shape - x: {}, y: {}'.format(X_val.shape, y_val.shape))
print('Testing data shape - x: {}, y: {}'.format(X_test.shape, y_test.shape))

print('Training set class distribution:\n 1: {} \n 2: {}'.format(*Counter(y_train).values()))
# print('Evaluation set class distribution:\n 1: {} \n 2: {}'.format(*Counter(y_val).values()))
print('Testing set class distribution:\n 1: {} \n 2: {}'.format(*Counter(y_test).values()))

# %% # ===== # DATA ENCODING # ===== #

prompt_continue()

print('Dimensionality Reduction')
# bottleneck_dim = int(X.shape[1] / 8)
bottleneck_dim = 500
print('Bottleneck Dimensions: '.format(bottleneck_dim))

autoencoder = Autoencoder(X_train.shape[1], bottleneck_dim)

autoencoder.summary()
prompt_continue('Do you want to continue with the autoassociative encoding of this model?')

autoencoder.train(X_train, X_test, batch_size=16)
encoded_X_train = autoencoder.get_encoded_image(X_train)
# encoded_X_val = autoencoder.get_encoded_image(X_val)
encoded_X_test = autoencoder.get_encoded_image(X_test)

print('Encoded training set shape - x: {}, y: {}'.format(encoded_X_train.shape, y_train.shape))
# print('Encoded training eval shape - x: {}, y: {}'.format(encoded_X_val.shape, y_val.shape))
print('Encoded testing data shape - x: {}, y: {}'.format(encoded_X_test.shape, y_test.shape))

# %% # ===== # TRAINING AND CLASSIFICATION # ===== #

model = Sequential()

# First FCN hidden layer
model.add(Dense(
    units=50,
    input_dim=encoded_X_train.shape[1],
    activation='tanh',
    kernel_initializer='random_normal'
))
# model.add(keras.layers.Dropout(0.2))

# Second FCN hidden layer
model.add(Dense(10, activation='tanh'))
# model.add(keras.layers.Dropout(0.2))

# Output layer
model.add(Dense(1, activation='sigmoid'))

# sgd = SGD(lr=1e-5, nesterov=False)
adam = adam(lr=.0001)

model.compile(
    loss='binary_crossentropy',
    metrics=['accuracy'],
    optimizer=adam
)

print(model.summary())

prompt_continue()

classifier = model.fit(
    encoded_X_train,
    y_train,
    epochs=1000,
    verbose=1,
    shuffle=True,
    batch_size=8,
    validation_data=(encoded_X_test, y_val)
)

# %% KERAS EVALUATION

pred_y = model.predict(encoded_X_test)
pred_y = (pred_y > 0.5)

score = model.evaluate(encoded_X_test, y_test, verbose=1)
print('Loss: {}'.format(score[0]))
print('Accuracy: {}'.format(score[1]))

print(classification_report(y_test, pred_y))

plot_history(classifier)
cm = confusion_matrix(y_test, pred_y)
print(cm)
plot_confusion_matrix(cm, classes=le.inverse_transform([1, 2]))

