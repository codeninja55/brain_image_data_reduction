# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# brain_image_data_reduction
# classifier.py
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

"""classifier.py: 

{Description}
"""

#%%
import os
import sys

from pathlib import Path
import scipy.io as sio
import random
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn import preprocessing

#%%


def load_fmri(f, inp, cls):
    PWD_PATH = os.path.abspath(os.getcwd())
    FILE_PATH = Path(PWD_PATH) / 'src' / 'data' / f
    raw_mat = sio.loadmat(FILE_PATH.absolute())

    num_rois = 25
    rois = ['CALC', 'LIPL', 'LT', 'LTRIA', 'LOPER', 'LIPS', 'LDLPFC']
    vox_ind = [
        raw_mat["meta"][0]["rois"][0]["columns"][0][j][0]
        for j in range(num_rois - 1)
        if raw_mat["meta"][0]["rois"][0]["name"][0][j][0] in rois
    ]

    voxels = []
    for i in range(len(vox_ind)):
        for j in range(len(vox_ind[i])):
            voxels.append(vox_ind[i][j])

    stop = 8
    frames = [i for i in range(stop)]

    num_trials = 54
    data = []
    classification = []

    for ki, k in enumerate(frames):
        data.append([])
        classification.append([])
        offset_i = 0

        for i in range(num_trials):
            if len(raw_mat["data"][i][0]) != num_trials + 1:
                offset_i += 1
                continue

            data[ki].append([])
            classification[ki].append([])
            offset_j = 0

            for ji, j in enumerate(voxels):
                data[ki][i - offset_i].append([])
                data[ki][i - offset_i][ji - offset_j] = raw_mat["data"][i][0][k][j - 1]
                classification[ki][i - offset_i].append(
                    1 if raw_mat["info"][0][i]["firstStimulus"][0] == "P" else 0
                )

    frame_start = 0
    frame_end = 8
    lastinp = 0
    for i in range(frame_start, frame_end):
        for j in range(len(data[i])):
            inp.append(data[i][j])
            if len(data[i][j]) != lastinp:
                print(len(data[i][j]))
                lastinp = len(data[i][j])
            cls.append(classification[i][j])
        return inp, cls


numTestingTrials = 50
inputfiles = ['data-starplus-04847-v7.mat']  # , 'data-starplus-04799-v7.mat']
inp = []
cls = []
for f in inputfiles:
    print('loading ', f)
    inp, cls = load_fmri(f, inp, cls)

#%%
c = list(zip(inp, cls))
random.shuffle(c)
training_data, training_classification = zip(*c[numTestingTrials:])
testing_data, testing_classification = zip(*c[:numTestingTrials])

training_data_n = preprocessing.scale(training_data)
testing_data_n = preprocessing.scale(testing_data)
print('training and classifying...')
clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(50, 10), random_state=1, max_iter=1000,
                    learning_rate_init=.0005)

clf.fit(training_data_n, np.array(training_classification).reshape(-1))

predict = np.array(clf.predict(testing_data_n))
actual = np.array(testing_classification).reshape(-1)
error = np.mean(predict != actual)
print('Error&nbsp;&nbsp;&nbsp;&nbsp; : ', error)
print('prediction: ', predict)
print('actual&nbsp;&nbsp;&nbsp; : ', actual)
print('f1Score&nbsp;&nbsp; : ', f1_score(actual, predict, average='weighted'))
