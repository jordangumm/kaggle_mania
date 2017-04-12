""" Maxout Network

TODO: Track Uncertainty
- potentially bias training based on most uncertain examples until an equilibrium is found
"""

import lime
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
import copy

from lasagne.layers import FeaturePoolLayer, batch_norm
from lasagne.nonlinearities import rectify, softmax, linear, sigmoid, elu
from lasagne.objectives import aggregate, categorical_crossentropy
from lasagne.init import HeNormal
from lasagne.init import Glorot, Normal
from lasagne.updates import norm_constraint

from sklearn.pipeline import make_pipeline


class Maxout():
    """ Maxout Network """

    def __init__(self, num_features, num_layers, num_nodes, dropout, learning_rate, momentum, verbose=False):
        self.verbose = verbose
        self.input_var = T.matrix('inputs')
        self.target_var = T.ivector('targets')

        self.num_features = num_features
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.network = self.get_network()
        self.final_network = copy.copy(self.network)

        self.final_prediction = lasagne.layers.get_output(self.final_network,
                                                          deterministic=True)
        self.final_predict = theano.function([self.input_var], self.final_prediction,
                                                           allow_input_downcast=True)


        self.prediction = lasagne.layers.get_output(self.network,
                                               deterministic=True)
        self.predict_function = theano.function([self.input_var], self.prediction,
                                                        allow_input_downcast=True)

        self.loss = categorical_crossentropy(self.prediction, self.target_var)
        self.loss = aggregate(self.loss, mode='mean')

        # L2 regularization (weight decay)
        weightsl2 = lasagne.regularization.regularize_network_params(self.network,
                                                    lasagne.regularization.l2)
        weightsl1 = lasagne.regularization.regularize_network_params(self.network,
                                                    lasagne.regularization.l1)
        self.loss += 1e-1*weightsl2 #+ 1e-5*weightsl1

        # ADAM training
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.adagrad(self.loss, params, learning_rate=learning_rate)
        #updates = lasagne.updates.adam(self.loss, params)
        #updates = lasagne.updates.nesterov_momentum(self.loss, params,
        #                learning_rate=learning_rate, momentum=momentum)

        self.train = theano.function([self.input_var, self.target_var],
                                            self.loss, updates=updates)

        self.create_test_functions()


    def create_test_functions(self):
        """ Create Test Functions
        """
        test_prediction = lasagne.layers.get_output(self.network,deterministic=True)
        test_loss = categorical_crossentropy(test_prediction, self.target_var).mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), self.target_var), dtype=theano.config.floatX)
        self.test = theano.function([self.input_var, self.target_var],[test_loss, test_acc])


    def add_maxout_layer(self, network, num_nodes=240):
        network = lasagne.layers.DropoutLayer(network, p=self.dropout)
        network = lasagne.layers.DenseLayer(network, nonlinearity=None, num_units=num_nodes, W=Glorot(Normal))
        return lasagne.layers.FeaturePoolLayer(incoming=network, pool_size=4,
                                    axis=1, pool_function=theano.tensor.max)


    def get_network(self):
        network = lasagne.layers.InputLayer(shape=(None, self.num_features),
                                            input_var=self.input_var)
        network = lasagne.layers.DropoutLayer(network, p=self.dropout)
        for _ in xrange(0, self.num_layers):
            network = self.add_maxout_layer(network, self.num_nodes)
        return lasagne.layers.DenseLayer(network, num_units=2,nonlinearity=softmax)


    def predict_proba(self, test_X):
        return self.final_predict(test_X)


    def fit(self, train_X, train_y, val_X, val_y, test_X, features, batch_size=10,
                    num_epochs=99999, early_stop_rounds=3, eval_type='log_loss'):
        """ Train Maxout Network

        Returns list of predictions for test_X
        """
        #self.network = self.get_network()
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

        try:
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
                    self.final_network = copy.copy(self.network)
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
        except:
            pass

        print 'best val loss: {}'.format(best_val_loss)

        return self.predict_function(test_X)


def train_bagging(df, features, verbose, batch_size, num_epochs,
                    num_layers, num_nodes,dropout,learning_rate,
                    momentum, early_stop_rounds, num_baggs, eval_type='log_loss'):
    """ Train Maxout Networks in boosted aggregation style

    Returns tuple of scores to minimize
    """
    games = pd.read_csv('../data/original/TourneyDetailedResults.csv')
    teams = pd.read_csv('../data/original/Teams.csv')
    seeds = pd.read_csv('../data/original/TourneySeeds.csv')

    def normalize(data):
        for key in data.keys():
            if not key in features: continue
            data[key] -= data[key].mean()
            data[key] /= data[key].std()
        return data

    models = []
    seasons = (2013,2014,2015,2016) # kaggle years minus 2016
    for season in seasons:

        test_season = season
        tmp_df = df[df['season'] == test_season]
        for s in xrange(2003, test_season): # only using previous 3 seasons
            tmp_df = tmp_df.append(df[df['season'] == s])

        test_df = normalize(tmp_df[tmp_df['season'] == test_season])
        train_df = normalize(tmp_df[tmp_df['season'] != test_season])

        batch_size = int(len(test_df)*.1) #################################### SET BATCH SIZE!!!

        print 'test season: {}'.format(test_df['season'].unique()[0])
        print 'train seasons: {}-{}'.format(min(train_df['season'].unique()), max(train_df['season'].unique()))
        print 'num features: {}'.format(len(features))

        test_X = np.array(test_df[features], dtype=np.float32)
        test_y = np.array(test_df['won'], dtype=np.int32)

        pred_outputs = []
        [pred_outputs.append([]) for _ in xrange(len(test_y))]

        # Run boosted aggregation!
        for bag_iteration in xrange(0,num_baggs):
            sampling_df = train_df.sample(n=len(train_df), replace=True, random_state=bag_iteration*94)

            train_X = np.array(sampling_df[features], dtype=np.float32)
            train_y = np.array(sampling_df['won'], dtype=np.int32)

            val_X = np.array(train_df[features], dtype=np.float32)
            val_y = np.array(train_df['won'], dtype=np.int32)

            maxout_trainer = Maxout(num_features=len(features),
                                    num_layers=num_layers,
                                    num_nodes=num_nodes,
                                    dropout=dropout,
                                    learning_rate=learning_rate,
                                    momentum=momentum,
                                    verbose=verbose)
            iter_preds = maxout_trainer.fit(train_X=train_X,
                                train_y=train_y,
                                val_X=val_X,
                                val_y=val_y,
                                test_X=test_X,
                                features=features,
                                early_stop_rounds=early_stop_rounds,
                                eval_type=eval_type,
                                batch_size=batch_size)

            #from lime.lime_tabular import LimeTabularExplainer
            #explainer = LimeTabularExplainer(train_X, feature_names=features, class_names=['lost', 'won'], discretize_continuous=True)
            #for game_num, test_example in enumerate(test_X):
            #    exp = explainer.explain_instance(test_example, maxout_trainer.predict_proba, num_features=len(features))
            #    exp.save_to_file('../output/{}/{}_explanation.html'.format(season, game_num))

            [pred_outputs[pred_num].append(pred[1]) for pred_num, pred in enumerate(iter_preds)]

        #print pred_outputs
        import math
        tourney_games = games[games['Season'] == test_season]
        tourney_seeds = seeds[seeds['Season'] == test_season]
        bagg_output = open('../output/{}_maxout_tourney_game_predictions.csv'.format(season), 'w+')
        bagg_output.write('team_one,team_two,team_one_pred_mean,team_one_truth,team_two_pred_mean,team_two_truth\n')
        predictions = []
        for pred_num, pred in enumerate(pred_outputs):
            if pred_num%2==0:
                wteam = tourney_games.iloc[[int(math.floor(pred_num/2))]]['Wteam'].unique()[0]
                lteam = tourney_games.iloc[[int(math.floor(pred_num/2))]]['Lteam'].unique()[0]
                wteam_name = teams[teams['Team_Id'] == wteam]['Team_Name'].unique()[0]
                lteam_name = teams[teams['Team_Id'] == lteam]['Team_Name'].unique()[0]
                wteam_seed = tourney_seeds[tourney_seeds['Team'] == wteam]['Seed'].unique()[0]
                lteam_seed = tourney_seeds[tourney_seeds['Team'] == lteam]['Seed'].unique()[0]
                bagg_output.write('{}.{}.{},{}.{}.{},'.format(wteam_seed, wteam_name, wteam,
                                                            lteam_seed, lteam_name, lteam))
                bagg_output.write('{},{},'.format(np.mean(pred),test_y[pred_num]))
                predictions.append(np.mean(pred))
            else:
                bagg_output.write('{},{}\n'.format(np.mean(pred),test_y[pred_num]))
                predictions.append(np.mean(pred))
        bagg_output.close()

        from sklearn.metrics import log_loss

        predictions_df = pd.read_csv('../output/{}_maxout_tourney_game_predictions.csv'.format(season))
        truth_df = pd.read_csv('../data/original/TourneyCompactResults.csv')
        truth_df = truth_df[truth_df['Season'] == season]

        final_predictions = []
        final_y = []
        for i, pred in predictions_df.iterrows():
            if pred['team_one'] > pred['team_two']:
                final_predictions.append(pred['team_one_pred_mean'] / (pred['team_one_pred_mean'] + pred['team_two_pred_mean']))
                final_y.append(pred['team_one_truth'])
            else:
                final_predictions.append(pred['team_two_pred_mean'] / (pred['team_two_pred_mean'] + pred['team_one_pred_mean']))
                final_y.append(pred['team_two_truth'])

        print '{} final log loss: {}'.format(season, log_loss(final_y, final_predictions))

        if verbose:
            sys.exit('only one iteration for verbose debugging')



@click.command()
@click.argument('num_nodes', type=click.INT)
@click.argument('num_layers', type=click.INT)
@click.option('-dropout', type=click.FLOAT, default=0.50)
@click.option('-learning_rate', type=click.FLOAT, default=0.4)
@click.option('-momentum', type=click.FLOAT, default=0.5)
@click.option('-eval_type', type=click.STRING, default='log_loss')
@click.option('-batch_size', type=click.INT, default=1)
@click.option('-early_stop', type=click.INT, default=200)
@click.option('-verbose', type=click.BOOL, default=False)
@click.option('-max_epochs', type=click.INT, default=9999)
@click.option('-num_baggs', type=click.INT, default=1)
def run(num_nodes, num_layers, dropout, learning_rate, momentum, eval_type, batch_size, early_stop, verbose, max_epochs, num_baggs):
    for i, s in enumerate(xrange(2003,2017)):
        if i == 0:
            df = pd.read_csv('../data/games/{}_tourney_games.csv'.format(s))
            df['season'] = s
        else:
            tmp = pd.read_csv('../data/games/{}_tourney_games.csv'.format(s))
            tmp['season'] = s
            df = df.append(tmp)

    df = df.fillna(0.0)

    features = df.keys().tolist()
    features.remove('season')
    #features.remove('team_name')
    features.remove('won')

    """ Features removed due to LIME inspection """
    features.remove('seed')
    features.remove('_seed')

    train_bagging(df=df, features=features, batch_size=batch_size, num_epochs=max_epochs,
                    num_layers=num_layers, num_nodes=num_nodes, dropout=dropout,
                    learning_rate=learning_rate, momentum=momentum,
                    early_stop_rounds=early_stop, eval_type=eval_type,
                    verbose=verbose, num_baggs=num_baggs)


if __name__ == "__main__":
    run()
