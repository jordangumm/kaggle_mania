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
from lasagne.nonlinearities import rectify, softmax, linear
from lasagne.objectives import aggregate, squared_error
from lasagne.init import HeNormal


class Maxout():
    """ Maxout Network """

    def __init__(self, num_features, num_layers, num_nodes, dropout_p, learning_rate, momentum, verbose=False):
        self.verbose = verbose
        self.input_var = T.matrix('inputs')
        self.target_var = T.ivector('targets')

        self.num_features = num_features
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.dropout_p = dropout_p
        self.network = self.get_network()

        self.prediction = lasagne.layers.get_output(self.network,
                                                    deterministic=True)
        self.predict_function = theano.function([self.input_var], self.prediction)

        self.loss = squared_error(self.prediction, self.target_var)
        self.loss = aggregate(self.loss, mode='mean')

        # L2 regularization (weight decay)
        weightsl2 = lasagne.regularization.regularize_network_params(self.network,
                                                    lasagne.regularization.l2)
        self.loss #+= 1e-4*weightsl2

        # ADAM training
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        #updates = lasagne.updates.adagrad(self.loss, params, learning_rate=learning_rate)
        updates = lasagne.updates.adam(self.loss, params)
        #updates = lasagne.updates.nesterov_momentum(self.loss, params,
        #                learning_rate=learning_rate, momentum=momentum)

        self.train = theano.function([self.input_var, self.target_var],
                                            self.loss, updates=updates)

        self.create_test_functions()


    def create_test_functions(self):
        """ Create Test Functions
        """
        test_prediction = lasagne.layers.get_output(self.network,deterministic=True)
        test_loss = squared_error(test_prediction, self.target_var).mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), self.target_var), dtype=theano.config.floatX)
        self.test = theano.function([self.input_var, self.target_var],[test_loss, test_acc])


    def add_maxout_layer(self, network, num_nodes=240):
        network = lasagne.layers.DenseLayer(network, nonlinearity=rectify, num_units=num_nodes, W=HeNormal(gain=.01))
        return lasagne.layers.FeaturePoolLayer(incoming=network, pool_size=2,
                                    axis=1, pool_function=theano.tensor.max)


    def get_network(self):
        network = lasagne.layers.InputLayer(shape=(None, self.num_features),
                                            input_var=self.input_var)
        network = lasagne.layers.DropoutLayer(network, p=0.2)
        for _ in xrange(0, self.num_layers):
            network = self.add_maxout_layer(network, self.num_nodes)
        return lasagne.layers.DenseLayer(network, num_units=1,nonlinearity=linear)


    def train_model(self, train_X, train_y, val_X, val_y, test_X, features, batch_size=10, num_epochs=999,
                              early_stop_rounds=3, eval_type='log_loss'):
        """ Train Maxout Network

        Returns list of predictions for test_X
        """
        self.network = self.get_network()
        season_evals = []
        def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
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

        best_val_loss = 1000.0
        best_bayes_loss = 1000.0
        since_best = 0 # for early stopping
        all_bayes_loss_epochs = []
        for epoch_num, epoch in enumerate(range(num_epochs)):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(train_X, train_y, batch_size, shuffle=True):
                inputs, targets = batch
                err = self.train(inputs, targets)
                train_err += err
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_batches = 0
            for batch in iterate_minibatches(val_X, val_y, batch_size, shuffle=False):
                inputs, targets = batch
                err, acc = self.test(inputs, targets)
                val_err += err
                val_batches += 1

            val_loss = val_err / val_batches

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                since_best = 0

            since_best += 1

            if self.verbose:
                # print the results for this epoch:
                print("Epoch {} of {} took {:.3f}s".format(
                                epoch + 1, num_epochs, time.time() - start_time))
                print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
                print("  validation loss:\t\t{:.6f}".format(val_loss))

            if since_best > early_stop_rounds:
                break

        print 'best val loss: {}'.format(best_val_loss)

        return self.predict_function(test_X)



def train_bagging(df, features, verbose, batch_size=10, num_epochs=999,
                num_layers=2, num_nodes=10,dropout=0.5,learning_rate=0.002,
                momentum=0.5, early_stop_rounds=3, eval_type='log_loss'):
    """ Train Maxout Networks in boosted aggregation style

    Returns tuple of scores to minimize
    """
    games = pd.read_csv('../data/original/TourneyDetailedResults.csv')
    teams = pd.read_csv('../data/original/Teams.csv')
    seeds = pd.read_csv('../data/original/TourneySeeds.csv')

    models = []
    seasons = (2013,2014,2015,2016) # kaggle years minus 2016
    for i, season in enumerate(seasons):

        test_season = season
        tmp_df = df[df['season'] == test_season]
        for s in xrange(2003, test_season): # only using previous 3 seasons
            tmp_df = tmp_df.append(df[df['season'] == s])

        test_df = tmp_df[tmp_df['season'] == test_season]
        train_df = tmp_df[tmp_df['season'] != test_season]

        print 'test season: {}'.format(test_df['season'].unique()[0])
        print 'train seasons: {}-{}'.format(min(train_df['season'].unique()), max(train_df['season'].unique()))
        print 'num features: {}'.format(len(features))

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

        test_X = np.array(scaler.fit_transform(test_df[features]), dtype=np.float32)
        test_y = np.array(test_df['won'], dtype=np.int32)

        pred_outputs = []
        [pred_outputs.append([]) for _ in xrange(len(test_y))]

        # Run boosted aggregation!
        for i in xrange(0,25):
            sampling_df = train_df.sample(n=len(train_df), replace=True)

            train_X = np.array(scaler.fit_transform(sampling_df[features]), dtype=np.float32)
            train_y = np.array(sampling_df['won'], dtype=np.int32)

            val_X = np.array(scaler.fit_transform(train_df[features]), dtype=np.float32)
            val_y = np.array(train_df['won'], dtype=np.int32)


            maxout_trainer = Maxout(num_features=len(features),
                                    num_layers=num_layers,
                                    num_nodes=100,
                                    dropout_p=dropout,
                                    learning_rate=learning_rate,
                                    momentum=momentum,
                                    verbose=verbose)
            predictions = maxout_trainer.train_model(train_X=train_X,
                                                train_y=train_y,
                                                val_X=val_X,
                                                val_y=val_y,
                                                test_X=test_X,
                                                features=features,
                                                early_stop_rounds=early_stop_rounds,
                                                eval_type=eval_type)
            [pred_outputs[i].append(pred) for i, pred in enumerate(predictions)]

        import math
        tourney_games = games[games['Season'] == test_season]
        bagg_output = open('../output/{}_maxout_tourney_score_predictions.csv'.format(season), 'w+')
        bagg_output.write('wteam,lteam,wscore_var,wscore_mean,wscore_truth,lscore_var,lscore_mean,lscore_truth\n')
        for i, pred in enumerate(pred_outputs):
            if i%2==0:
                wteam = tourney_games.iloc[[int(math.floor(i/2))]]['Wteam'].unique()[0]
                lteam = tourney_games.iloc[[int(math.floor(i/2))]]['Lteam'].unique()[0]
                wteam_name = teams[teams['Team_Id'] == wteam]['Team_Name'].unique()[0]
                lteam_name = teams[teams['Team_Id'] == lteam]['Team_Name'].unique()[0]
                wteam_seed = seeds[seeds['Team'] == wteam]['Seed'].unique()[0]
                lteam_seed = seeds[seeds['Team'] == lteam]['Seed'].unique()[0]
                bagg_output.write('{}.{},{}.{},'.format(wteam_seed, wteam_name,
                                                            lteam_seed, lteam_name))
                bagg_output.write('{},{},{},'.format(np.var(pred),np.mean(pred),test_y[i]))
            else:
                bagg_output.write('{},{},{}\n'.format(np.var(pred),np.mean(pred),test_y[i]))

        bagg_output.close()

        if verbose:
            sys.exit('only one iteration for verbose debugging')



@click.command()
@click.argument('num_nodes', type=click.INT)
@click.argument('num_layers', type=click.INT)
@click.option('-dropout', type=click.FLOAT, default=0.5)
@click.option('-learning_rate', type=click.FLOAT, default=0.001)
@click.option('-momentum', type=click.FLOAT, default=0.5)
@click.option('-eval_type', type=click.STRING, default='log_loss')
@click.option('-batch_size', type=click.INT, default=20)
@click.option('-early_stop', type=click.INT, default=20)
@click.option('-verbose', type=click.BOOL, default=False)
def run(num_nodes, num_layers, dropout, learning_rate, momentum, eval_type, batch_size, early_stop, verbose):
    for i, s in enumerate((2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016)):
        if i == 0:
            df = pd.read_csv('../data/games/{}_tourney_diff_games_score.csv'.format(s))
            df['season'] = s
        else:
            tmp = pd.read_csv('../data/games/{}_tourney_diff_games_score.csv'.format(s))
            tmp['season'] = s
            df = df.append(tmp)

    features = df.keys().tolist()
    features.remove('season')
    features.remove('won')

    train_bagging(df=df, features=features, batch_size=batch_size, num_epochs=999,
                        num_layers=num_layers, num_nodes=num_nodes, dropout=dropout,
                        learning_rate=learning_rate, momentum=momentum,
                        early_stop_rounds=early_stop, eval_type=eval_type,
                        verbose=verbose)


if __name__ == "__main__":
    run()
