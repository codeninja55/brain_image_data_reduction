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
__date__ = '2019.05.18'

"""classifier.py: 

{Description}
"""

import pprint
import tensorflow as tf
import keras.backend as K
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam, Nadam, Adadelta, RMSprop
from sklearn.metrics import classification_report, confusion_matrix
import keras.utils.generic_utils

from utils.helpers import prompt_continue
from utils.metrics import plot_confusion_matrix, plot_history

run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)


class FMRIClassifier(object):

    def __init__(self, input_dim, epoch=250, optimizer='', learning_rate=0.0001, decay=.0):
        self.history = None
        self.alpha = learning_rate
        self.epoch = epoch

        self.model = Sequential(name='fMRI Classifier')

        # First FCN hidden layer
        self.model.add(Dense(
            units=100,
            input_dim=input_dim,
            activation='relu',
            kernel_initializer='random_normal'
        ))

        # Second FCN hidden layer
        self.model.add(Dense(
            units=20,
            activation='relu',
            kernel_initializer='random_normal'
        ))

        # Output layer
        self.model.add(Dense(
            units=1,
            activation='sigmoid',
            kernel_initializer='random_normal'
        ))

        # Optimizer
        _decay = decay
        _optimizer = SGD(lr=self.alpha, nesterov=False, momentum=.0, decay=_decay)
        if optimizer == 'adam':
            _optimizer = Adam(lr=self.alpha, decay=_decay)
        elif optimizer == 'nadam':
            _optimizer = Nadam(lr=self.alpha)
        elif optimizer == 'adadelta':
            _optimizer = Adadelta(lr=self.alpha, decay=_decay)
        elif optimizer == 'rmsprop':
            _optimizer = RMSprop(lr=self.alpha, decay=_decay)

        # Compile the model
        self.model.compile(
            loss='binary_crossentropy',  # ['categorical_crossentropy',
                                         #  'mean_squared_error',
                                         #  'kullback_leibler_divergence']
            optimizer=_optimizer,
            metrics=['accuracy', 'mse'],
            options=run_opts
        )

    def train(self, x, y, batch_size=32, shuffle=False, validation_data=(None, None), verbose=0):
        self.history = self.model.fit(
            x=x,
            y=y,
            epochs=self.epoch,
            shuffle=shuffle,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=verbose
        )
        return self.history

    def evaluate(self, test_data, test_labels, verbose=1):
        pred_y = self.model.predict(test_data)
        pred_y = (pred_y > 0.5)
        score = self.model.evaluate(x=test_data, y=test_labels, verbose=verbose)
        print('\n\nfMRI Classifier Evaluation Results:')
        print('Loss: {}'.format(score[0]))
        print('Accuracy: {}\n'.format(score[1]))
        report = classification_report(y_true=test_labels, y_pred=pred_y, output_dict=False)
        print(report)

        if self.history:
            plot_history(self.history)

        cm = confusion_matrix(test_labels, pred_y)
        print(cm)

        # plot_confusion_matrix(cm, classes=le.inverse_transform([1, 2]))

    def summary(self):
        print('\n\nfMRI Classifier Model Summary')
        configs = self.model.get_config()
        configs['optimizer'] = str(self.model.optimizer.__str__())
        configs['optimizer_configs'] = self.model.optimizer.get_config()
        configs['learning_rate'] = str(K.eval(self.model.optimizer.lr))
        configs['loss'] = str(self.model.loss)
        configs['metrics'] = self.model.metrics_names
        pprint.pprint(configs)
        return self.model.summary()


def fmri_classifier_run(input_data, input_labels, validation_data,
                        validation_labels, test_data, test_labels,
                        verbose=0, evaluate=True):
    clf = FMRIClassifier(
        input_dim=input_data.shape[1],
        epoch=1000,
        optimizer='nadam',
        learning_rate=0.001
    )
    clf.summary()
    # prompt_continue()
    clf.train(
        x=input_data,
        y=input_labels,
        batch_size=16,
        validation_data=(validation_data, validation_labels),
        verbose=verbose,
        shuffle=True
    )

    if evaluate:
        clf.evaluate(test_data=test_data, test_labels=test_labels, verbose=verbose)
        return None
    else:
        return clf

