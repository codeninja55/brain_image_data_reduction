# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# brain_image_data_reduction
# data.py
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

"""data.py: 

{Description}
"""
import os
import sys
from typing import Tuple
from pathlib import Path

import numpy as np
import scipy.io as sio

PWD_PATH = os.path.abspath(os.getcwd())
print('PWD:', os.getcwd())
DATA_PATH = os.path.abspath(os.path.join(PWD_PATH, 'data'))
if PWD_PATH not in sys.path:
    sys.path.append(PWD_PATH)
    sys.path.append(DATA_PATH)


INPUT_FILES = [
    'data-starplus-04799-4000.mat',
    'data-starplus-04820-4000.mat',
    'data-starplus-04847-4000.mat',
    'data-starplus-05675-4000.mat',
    'data-starplus-05680-4000.mat',
    'data-starplus-05710-4000.mat',
]

ROIS = ['CALC', 'LIPL', 'LT', 'LTRIA', 'LOPER', 'LIPS', 'LDLPFC']


def load_fmri() -> Tuple:
    samples = []
    labels = []

    for f in INPUT_FILES:
        mat_path = Path(DATA_PATH) / f
        raw = sio.loadmat(mat_path.absolute())
        # raw = sio.loadmat(f)
        samples.append(np.array(raw['examples']))
        labels.append(np.array(raw['labels']))

    return samples, labels


