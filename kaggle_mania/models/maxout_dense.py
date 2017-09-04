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

from lasagne.layers import FeaturePoolLayer, NonlinearityLayer, batch_norm, DenseLayer, ElemwiseSumLayer, ConcatLayer
from lasagne.nonlinearities import rectify, softmax
from lasagne.init import HeNormal

from maxout import Maxout


class MaxoutDense(Maxout):
    """ Maxout Dense Network """
    def add_dense_maxout_block(self, network, num_nodes=240, dropout=0.5):
        network = lasagne.layers.DropoutLayer(network, p=self.dropout)
        network = DenseLayer(network,nonlinearity=rectify,num_units=self.num_nodes)
        maxout = FeaturePoolLayer(incoming=network, pool_size=2,axis=1, pool_function=theano.tensor.max)
        return ConcatLayer([network, maxout], axis=1)

    def get_network(self):
        network = lasagne.layers.InputLayer(shape=(None, self.num_features),input_var=self.input_var)
        for i in xrange(0, self.num_layers):
            network = DenseLayer(network,nonlinearity=rectify,num_units=self.num_nodes)
            if i != 0:
                network = FeaturePoolLayer(incoming=network, pool_size=2,axis=1, pool_function=theano.tensor.mean)
            for _ in xrange(0, 1):
                network = DenseLayer(network,nonlinearity=rectify,num_units=self.num_nodes)
                layers = [network]
                for _ in xrange(0, 4):
                    network = batch_norm(self.add_dense_maxout_block(network, self.num_nodes, self.dropout))
                    layers.append(network)
                    network = ConcatLayer(layers, axis=1)
        maxout = FeaturePoolLayer(incoming=network, pool_size=2,axis=1, pool_function=theano.tensor.mean)
        return lasagne.layers.DenseLayer(network, num_units=2,nonlinearity=lasagne.nonlinearities.softmax)
