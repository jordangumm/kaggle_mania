""" Maxout Network

TODO: Track Uncertainty
- potentially bias training based on most uncertain examples until an equilibrium is found
"""

import lime
import sys
import h5py
import pandas as pd
import cPickle as pickle
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.cross_validation import KFold

import theano
theano.config.floatX = 'float32'
import theano.tensor as T
import numpy as np
import lasagne
import time
import math
import os, sys
import click
import copy

from lasagne.layers import FeaturePoolLayer, batch_norm
from lasagne.nonlinearities import softmax, linear, sigmoid, elu
from lasagne.objectives import aggregate, categorical_crossentropy
from lasagne.init import HeNormal
from lasagne.init import Glorot, Normal
from lasagne.updates import norm_constraint


class Maxout():
    """ Maxout Network """

    def __init__(self, num_features, num_layers, num_nodes, dropout, learning_rate, weight_decay, verbose=False):
        self.verbose = verbose
        self.input_var = T.matrix('inputs')
        self.target_var = T.ivector('targets')

        self.num_features = num_features
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.network = self.build_network()

        self.prediction = lasagne.layers.get_output(self.network,
                                               deterministic=True)
        self.predict_function = theano.function([self.input_var], self.prediction,
                                                        allow_input_downcast=True)

        self.loss = categorical_crossentropy(self.prediction, self.target_var)
        self.loss = aggregate(self.loss, mode='mean')

	if not os.path.exists('output/models/'):
	    os.mkdir('output/models')

        # L2 regularization with weight decay
        weightsl2 = lasagne.regularization.regularize_network_params(self.network,
                                                    lasagne.regularization.l2)
        weightsl1 = lasagne.regularization.regularize_network_params(self.network,
                                                    lasagne.regularization.l1)
        self.loss += weight_decay*weightsl2 #+ 1e-5*weightsl1

        # ADAM training
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.adagrad(self.loss, params, learning_rate=learning_rate)
        #updates = lasagne.updates.adam(self.loss, params)
        #updates = lasagne.updates.nesterov_momentum(self.loss, params,
        #                learning_rate=learning_rate, momentum=momentum)

        self.train = theano.function([self.input_var, self.target_var],
                                            self.loss, updates=updates)

        self.create_test_function()
        self.create_bayes_test_function()


    def create_test_function(self):
        """ Create Test Function
        """
        test_prediction = lasagne.layers.get_output(self.network,deterministic=True)
        test_loss = categorical_crossentropy(test_prediction, self.target_var).mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), self.target_var), dtype=theano.config.floatX)
        self.test = theano.function([self.input_var, self.target_var],[test_loss, test_acc])


    def create_bayes_test_function(self):
        """ Create Bayes Test Function
        """
        test_prediction = lasagne.layers.get_output(self.network,deterministic=False)
        test_loss = categorical_crossentropy(test_prediction, self.target_var).mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), self.target_var), dtype=theano.config.floatX)
        self.bayes_test = theano.function([self.input_var, self.target_var],[test_loss, test_acc])
        self.bayes_predict = theano.function([self.input_var], test_prediction, allow_input_downcast=True)


    def add_maxout_layer(self, network, num_nodes=240):
        network = lasagne.layers.DropoutLayer(network, p=self.dropout)
        network = lasagne.layers.DenseLayer(network, nonlinearity=None, num_units=num_nodes*2, W=Glorot(Normal))
        return lasagne.layers.FeaturePoolLayer(incoming=network, pool_size=2,
                                    axis=1, pool_function=theano.tensor.max)


    def build_network(self):
        network = lasagne.layers.InputLayer(shape=(None, self.num_features),
                                            input_var=self.input_var)
        network = lasagne.layers.DropoutLayer(network, p=self.dropout)
        for _ in xrange(0, self.num_layers):
            network = self.add_maxout_layer(network, self.num_nodes)
        return lasagne.layers.DenseLayer(network, num_units=2,nonlinearity=softmax)


    def save_network(self, model_fp):
        net_info = {'network': self.network, 'params': lasagne.layers.get_all_param_values(self.network)}
        pickle.dump(net_info, open(model_fp, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


    def load_network(self, model_fp):
        net = pickle.load(open(model_fp,'rb'))
        all_params = net['params']
        lasagne.layers.set_all_param_values(self.network, all_params)


    def predict_proba(self, X):
        """ Returns log loss of X """
        return self.predict_function(X)


    def predict_bayes_proba(self, X):
        """ Returns predictions over X """
        preds = []
        for x in X:
            samples = []
            for _ in xrange(100):
                samples.append(self.bayes_predict([x])[0][1])
            preds.append(np.mean(samples))
        return preds


    def get_bayes_validation_metrics(self, test_X, test_y):
        """ Average non-deterministic feed forward passes from all examples """
        val_loss = 0.0
        val_acc = 0.0
        for batch in self.iterate_minibatches(test_X, test_y, 1, shuffle=False):
            inputs, targets = batch
            mc_iters = 20
            sampled_acc = 0.0
            sampled_loss = 0.0
            for _ in xrange(mc_iters):
                err, acc = self.bayes_test(inputs, targets)
                sampled_acc += acc
                sampled_loss += err
            val_acc += sampled_acc / mc_iters
            val_loss += sampled_loss / mc_iters
        val_acc = val_acc / len(test_y)
        val_loss = val_loss / len(test_y)
        return val_acc, val_loss


    def get_validation_metrics(self, test_X, test_y):
        """ Average deterministic feed forward batch passes """
        val_loss = 0.0
        val_acc = 0.0
        for batch in self.iterate_minibatches(test_X, test_y, 1, shuffle=False):
            inputs, targets = batch
            err, acc = self.test(inputs, targets)
            val_acc += acc
            val_loss += err
        val_acc = val_acc / len(test_y)
        val_loss = val_loss / len(test_y)
        return val_acc, val_loss


    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]


    def fit(self, train_X, train_y, val_X, val_y, features, batch_size=10,
                                    num_epochs=99999, early_stop_rounds=3):
        """ Train Maxout Network

        Returns list of predictions for test_X
        """
        season_evals = []

        best_val_loss = 1000.0
        best_bayes_loss = 1000.0
        since_best = 0 # for early stopping
        all_bayes_loss_epochs = []

        for epoch_num, epoch in enumerate(range(num_epochs)):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.iterate_minibatches(train_X, train_y, batch_size, shuffle=True):
                inputs, targets = batch
                err = self.train(inputs, targets)
                train_err += err
                train_batches += 1

            bayes_val_acc, bayes_val_loss = self.get_bayes_validation_metrics(val_X, val_y)
            val_acc, val_loss = self.get_validation_metrics(val_X, val_y)

            if bayes_val_loss < best_bayes_loss:
                self.save_network('output/models/model.pkl')
                best_bayes_loss = bayes_val_loss
                best_val_loss = val_loss # track raw validation loss with bayes
                since_best = 0

            if self.verbose:
                # print the results for this epoch:
                print("Epoch {} of {} took {:.3f}s".format(
                                epoch + 1, num_epochs, time.time() - start_time))
                print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
                print("  validation loss:\t\t{:.6f}".format(val_loss))
                print("  validation accu:\t\t{:.6f}".format(val_acc))

                print("  bayes val loss:\t\t{:.6f}".format(bayes_val_loss))
                print("  bayes val accu:\t\t{:.6f}".format(bayes_val_acc))

            since_best += 1
            if since_best > early_stop_rounds:
                break

        self.network = self.load_network('output/models/model.pkl')
        if self.verbose:
            print 'best val loss: {}'.format(best_val_loss)
        return self.predict_proba(val_X)
