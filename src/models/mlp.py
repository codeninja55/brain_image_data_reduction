# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# brain_image_data_reduction
# mlp.py
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

"""mlp.py: 

{Description}
"""

import numpy as np


class Layer:
    """
    The building block with each layer capable of performing two things:
    - Process input to get output: output: layer.forward(input)
    - Propagate gradients through itself: grad_input = layer.backprop(input, grad_output)
    Some layers have learnable parameters which they update during backpropagation.
    """
    def __init__(self):
        # here we can initialise layer parameters (if any) and auxiliary stuff.
        # A dummy layer does nothing
        pass

    def forward(self, inpt):
        """Takes the input data  of shape (batch, input_units), returns output data (batch, output_units)."""
        return inpt

    def backprop(self, inpt, output_gradient):
        """Performs a backpropagation step through the layer, with respect to the given input."""
        # To compute loss gradients w.r.t input, we need to apply chain rule
        # Chain rule: d(loss) / d(x) = d(loss) / d(layer) * d(layer) / d(x)
        # We have already received d(loss) / d(layer) as input, so you only need to multiply it
        # If our layer has parameters (e.g. dense layer), we also need to update them here using d(loss) / d(layer)
        num_units = inpt.shape[1]
        d_layer_d_input = np.eye(num_units)
        return np.dot(output_gradient, d_layer_d_input)  # chain rule


class ReLU(Layer):
    def __init__(self):
        # ReLU layer simply applies elementwise rectified linear unit to all inputs
        super().__init__()

    def forward(self, inpt):
        # apply elementwise ReLU to (batch, input_units) matrix
        relu_forward = np.maximum(0, inpt)
        return relu_forward

    def backprop(self, inpt, output_gradient):
        # compute gradient of loss w.r.t ReLU input
        relu_gradient = inpt > 0
        return output_gradient * relu_gradient


class Sigmoidal(Layer):
    # TODO:
    pass


class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        """
        A dense layer is a layer which performs a learned affine transformation.
        Function: f(x) = <W * x> + b
        :param input_units:
        :param output_units:
        :param learning_rate:
        """
        super().__init__()
        self.alpha = learning_rate
        self.weights = np.random.normal(loc=0.0,
                                        scale=np.sqrt(2 / (input_units + output_units)),
                                        size=(input_units, output_units))
        self.biases = np.zeros(output_units)

    def forward(self, inpt):
        """
        Perform an affine transformation.
        Function: f(x) = <W * x> + b
        :param inpt:
        :return:
        """
        # input shape: [batch, input_units]
        # output shape: [batch, output_units]
        return np.dot(inpt, self.weights) + self.biases

    def backprop(self, inpt, output_gradient):
        """
        Compute d(f) / d(x) = d(f) / d(dense) * d(dense) / d(x) where d(dense) / d(x) = weights.T
        :param inpt:
        :param output_gradient:
        :return:
        """
        input_gradient = np.dot(output_gradient, self.weights.T)

        # compute gradient w.r.t. weights and biases
        weights_gradient = np.dot(inpt.T, output_gradient)
        bias_gradient = output_gradient.mean(axis=0) * inpt.shape[0]
        assert weights_gradient.shape == self.weights.shape and bias_gradient.shape == self.biases.shape

        # Here we perform a stochastic gradient descent step
        self.weights = self.weights - self.alpha * weights_gradient
        self.biases = self.biases - self.alpha * bias_gradient

        return input_gradient


class BottleNeck(Layer):
    # TODO:
    def __init__(self):
        super().__init__()

    def forward(self, inpt):
        return super().forward(inpt)

    def backprop(self, inpt, output_gradient):
        return super().backprop(inpt, output_gradient)


class MLP:
    def __init__(self):
        self.network = []

    def add(self, layer):
        self.network.append(layer)

    # @staticmethod
    # def sigmoid(x):
    #     return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax_crossentropy_with_logits(logits, reference_answers):
        # compute crossentropy from logits[batch, n_classes] and ids of correct answers
        logits_for_answers = logits[np.arange(len(logits)), reference_answers]
        x_entropy = - logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))
        return x_entropy

    @staticmethod
    def grad_softmax_crossentropy_with_logits(logits, reference_answers):
        # compute crossentropy gradient from logits[batch, n_classes] and ids of correct answers
        ones_for_answers = np.zeros_like(logits)
        ones_for_answers[np.arange(len(logits)), reference_answers] = 1
        softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        return (- ones_for_answers + softmax) / logits.shape[0]

    def forward(self, X):
        """
        Compute activations of all network layers by applying them sequentially.
        :param X:
        :return: list of activations for each layer
        """
        #
        # return
        activations = []
        _X = X

        for layer in self.network:
            activations.append(layer.forward(_X))
            # updating input to last layer output
            _X = activations[-1]

        assert len(activations) == len(self.network)
        return activations

    def predict(self, X):
        """
        Compute network predictions.
        :param X:
        :return: indices of largest logit probability
        """
        logits = self.forward(X)[-1]
        return logits.argmax(axis=-1)

    def train(self, X, y):
        """
        Train the network on a given batch of X and y.
        :param X:
        :param y:
        :return:
        """
        # We first need to run forward to get all layer activations.
        # Then we can run layer.backward going from last to first layer.
        # After we have called backward for all layers, all Dense layers have already made one gradient step.

        # get the layer activations
        layer_activations = self.forward(X)
        layer_inputs = [X] + layer_activations  # layer_input[i] is an input for network[i]
        logits = layer_activations[-1]

        # compute the loss and the initial gradient
        loss = self.softmax_crossentropy_with_logits(logits, y)
        loss_gradient = self.grad_softmax_crossentropy_with_logits(logits, y)

        # propagate gradients through the network
        # reverse propagation as this is backprop
        for layer_idx in range(len(self.network))[::-1]:
            layer = self.network[layer_idx]
            loss_gradient = layer.backprop(layer_inputs[layer_idx],
                                           loss_gradient)  # gradient w.r.t. input, also weight updates

        return np.mean(loss)

