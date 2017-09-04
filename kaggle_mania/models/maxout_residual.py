""" Maxout Network

TODO: Track Uncertainty
- potentially bias training based on most uncertain examples until an equilibrium is found
"""

import sys
import h5py
import pandas as pd
from sklearn.svm import SVC
import math
from sklearn.metrics import *
from sklearn.cross_validation import KFold

import theano
theano.config.floatX = 'float32'
import theano.tensor as T
import numpy as np
import lasagne
import time
import os, sys
import click

from lasagne.layers import FeaturePoolLayer, NonlinearityLayer, DenseLayer, ElemwiseSumLayer, ConcatLayer
from lasagne.nonlinearities import rectify, softmax, linear
from lasagne.objectives import aggregate, categorical_crossentropy
from lasagne.init import HeNormal

from maxout import Maxout


class MaxoutResidual(Maxout):
    """ Maxout Residual Network """
    def add_residual_dense_maxout_block(self, network, num_nodes=240, dropout=0.5):
        network = lasagne.layers.DropoutLayer(network, p=self.dropout)
        identity = network
        network = DenseLayer(network,nonlinearity=rectify,num_units=self.num_nodes,W=HeNormal(gain=.01))
        network = FeaturePoolLayer(incoming=network, pool_size=2,axis=1, pool_function=theano.tensor.max)
        return NonlinearityLayer(ElemwiseSumLayer([identity, network.input_layer]), nonlinearity=rectify)


    def get_network(self):
        network = lasagne.layers.InputLayer(shape=(None, self.num_features),input_var=self.input_var)
        network = lasagne.layers.DropoutLayer(network, p=self.dropout)
        network = DenseLayer(network,nonlinearity=rectify,num_units=self.num_nodes,W=HeNormal(gain=.01))
        for _ in xrange(0, self.num_layers):
            network = self.add_residual_dense_maxout_block(network, self.num_nodes, self.dropout)
        return lasagne.layers.DenseLayer(network, num_units=2,
                            nonlinearity=lasagne.nonlinearities.softmax)
