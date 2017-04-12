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

from lasagne.layers import FeaturePoolLayer
from lasagne.nonlinearities import rectify, softmax


class Maxout():
    """ Maxout Network """

    def __init__(self, num_features, num_layers, num_nodes, dropout_p, learning_rate, momentum):
        self.input_var = T.matrix('inputs')
        self.target_var = T.ivector('targets')

        self.num_features = num_features
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.dropout_p = dropout_p
        self.network = self.get_network()

        self.prediction = lasagne.layers.get_output(self.network, deterministic=True)
        self.predict_function = theano.function([self.input_var], self.prediction)
        self.calc_loss = lasagne.objectives.categorical_crossentropy(self.prediction, self.target_var).mean()

        self.bayes_prediction = lasagne.layers.get_output(self.network,deterministic=False)
        self.calc_bayes_loss = lasagne.objectives.categorical_crossentropy(self.bayes_prediction, self.target_var)

        def get_uncertainty_loss(num_samples):
            """ Integrate bayesian loss into training step

            Now the error gradient reflects uncertainty!
            """
            final_loss = 0.0
            for _ in xrange(0,num_samples):
                final_loss += self.calc_bayes_loss.mean()
            return final_loss / float(num_samples)

        self.bayes_loss = get_uncertainty_loss(100)

        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.adam(self.bayes_loss, params)

        self.train = theano.function([self.input_var, self.target_var],
                                            self.bayes_loss, updates=updates)
        self.create_test_functions()


    def create_test_functions(self):
        """ Create Test Functions
        """
        test_acc = T.mean(T.eq(T.argmax(self.prediction, axis=1),
                                    self.target_var), dtype=theano.config.floatX)
        self.test = theano.function([self.input_var, self.target_var],
                                    [self.calc_loss, test_acc])

        bayes_test_acc = T.mean(T.eq(T.argmax(self.bayes_prediction, axis=1),
                                    self.target_var), dtype=theano.config.floatX)
        self.bayes_test = theano.function([self.input_var, self.target_var],
                                    [self.bayes_loss, bayes_test_acc])


    def add_maxout_layer(self, network, num_nodes=240):
        network = lasagne.layers.DenseLayer(network, nonlinearity=rectify, num_units=num_nodes)
        return lasagne.layers.FeaturePoolLayer(incoming=network, pool_size=2,
                                    axis=1, pool_function=theano.tensor.max)


    def get_network(self):
        network = lasagne.layers.InputLayer(shape=(None, self.num_features),
                                            input_var=self.input_var)
        network = lasagne.layers.DropoutLayer(network, p=self.dropout_p)
        for _ in xrange(0, self.num_layers):
            network = self.add_maxout_layer(network, self.num_nodes)
        #network = lasagne.layers.FeaturePoolLayer(incoming=network, pool_size=4,
        #                                axis=1, pool_function=theano.tensor.mean)
        return lasagne.layers.DenseLayer(network, num_units=2,
                            nonlinearity=lasagne.nonlinearities.softmax)

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

    def get_bayes_validation_metrics(self, test_X, test_y, batch_size):
        # And a full pass over the validation data:
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        for batch in self.iterate_minibatches(test_X, test_y, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = self.bayes_test(inputs, targets)
            val_loss += err
            val_acc += acc
            val_batches += 1
        val_acc = val_acc / val_batches
        val_loss = val_loss / val_batches
        return val_acc, val_loss

    def get_validation_metrics(self, test_X, test_y, batch_size):
        # And a full pass over the validation data:
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        for batch in self.iterate_minibatches(test_X, test_y, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = self.test(inputs, targets)
            val_loss += err
            val_acc += acc
            val_batches += 1

        val_acc = val_acc / val_batches
        val_loss = val_loss / val_batches
        return val_acc, val_loss

    def train_model(self, train_X, train_y, test_X, test_y, features, batch_size=10, num_epochs=999,
                              early_stop_rounds=3, eval_type='log_loss'):
        """ Train Maxout Network

        Returns tuple of scores to minimize
        """
        self.network = self.get_network()
        season_evals = []

        best_val_loss = 100.0
        best_val_acc = 0.0
        best_bayes_loss = 100.0
        best_bayes_acc = 0.0
        since_best = 0 # for early stopping
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

            bayes_val_acc, bayes_val_loss = self.get_bayes_validation_metrics(test_X, test_y, batch_size)
            val_acc, val_loss = self.get_validation_metrics(test_X, test_y, batch_size)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                since_best = 0
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                since_best = 0

            since_best += 1

            # Then we print the results for this epoch:
            #print("Epoch {} of {} took {:.3f}s".format(
            #                    epoch + 1, num_epochs, time.time() - start_time))
            #print("  training loss:\t{:.6f}".format(train_err / train_batches))
            #print("  val loss:\t\t{:.6f}".format(val_loss))
            #print("  val accuracy:\t\t{:.2f} %".format(val_acc * 100))
            #print("  bayes val loss:\t\t{:.6f}".format(bayes_val_loss))
            #print("  bayes val accuracy:\t\t{:.2f} %".format(bayes_val_acc * 100))

            if since_best > early_stop_rounds:
                break
        #print 'best loss: {}'.format(best_val_loss)
        #print 'best acc: {}'.format(best_val_acc)
        return best_val_loss



def train_cross_validation(df, features, batch_size=10, num_epochs=999,
                num_layers=2, num_nodes=10,dropout=0.5,learning_rate=0.002,
                momentum=0.5, early_stop_rounds=3, eval_type='log_loss'):
    """ Train Maxout Networks in CV style

    Returns tuple of scores to minimize
    """

    models = []
    seasons = (2013,2014,2015,2016) # kaggle years minus 2016
    for i, season in enumerate(seasons):

        test_season = season
        tmp_df = df[df['season'] == test_season]
        for s in xrange(test_season-3, test_season): # only using previous 3 seasons
            tmp_df = tmp_df.append(df[df['season'] == s])

        test_df = tmp_df[tmp_df['season'] == test_season]
        train_df = tmp_df[tmp_df['season'] != test_season]

        #for f in features:
        #    test_df[f] = test_df[f].rank(pct=True)
        #    train_df[f] = train_df[f].rank(pct=True)

        print 'test season: {}'.format(test_df['season'].unique()[0])
        print 'train seasons: {}-{}'.format(min(train_df['season'].unique()), max(train_df['season'].unique()))
        print 'num features: {}'.format(len(features))

        test_X = np.array(test_df[features], dtype=np.float32)
        test_y = np.array(test_df['won'], dtype=np.int32)

        train_X = np.array(train_df[features], dtype=np.float32)
        train_y = np.array(train_df['won'], dtype=np.int32)

        for i in xrange(0,5):
            maxout_trainer = Maxout(num_features=len(features),
                                    num_layers=num_layers,
                                    num_nodes=100,
                                    dropout_p=dropout,
                                    learning_rate=learning_rate,
                                    momentum=momentum)
            score = maxout_trainer.train_model(train_X=train_X,
                                                train_y=train_y,
                                                test_X=test_X,
                                                test_y=test_y,
                                                features=features,
                                                early_stop_rounds=early_stop_rounds,
                                                eval_type=eval_type)

            print 'test {}: {}'.format(eval_type, score)
        print



@click.command()
@click.argument('num_nodes', type=click.INT)
@click.argument('num_layers', type=click.INT)
@click.option('-dropout', type=click.FLOAT, default=0.5)
@click.option('-learning_rate', type=click.FLOAT, default=0.001)
@click.option('-momentum', type=click.FLOAT, default=0.5)
@click.option('-eval_type', type=click.STRING, default='log_loss')
@click.option('-batch_size', type=click.INT, default=20)
@click.option('-early_stop', type=click.INT, default=20)
@click.option('-norm_tech', default='scale')
def run(num_nodes, num_layers, dropout, learning_rate, momentum, eval_type, batch_size, early_stop, norm_tech):
    for i, s in enumerate((2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016)):
        if i == 0:
            df = pd.read_csv('../data/games/{}_tourney_diff_games.csv'.format(s))
            df['season'] = s
        else:
            tmp = pd.read_csv('../data/games/{}_tourney_diff_games.csv'.format(s))
            tmp['season'] = s
            df = df.append(tmp)

    features = df.keys().tolist()
    features.remove('season')
    features.remove('won')

    train_cross_validation(df=df, features=features, batch_size=batch_size, num_epochs=999,
                        num_layers=num_layers, num_nodes=num_nodes, dropout=dropout,
                        learning_rate=learning_rate, momentum=momentum,
                        early_stop_rounds=early_stop, eval_type=eval_type)


if __name__ == "__main__":
    run()