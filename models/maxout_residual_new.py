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

from maxout_new import Maxout


class MaxoutResidual(Maxout):
    """ Maxout Residual Network """
    def add_residual_dense_maxout_block(self, network, num_nodes=240, dropout=0.5):
        identity = network
        network = DenseLayer(network,nonlinearity=rectify,num_units=self.num_nodes,W=HeNormal(gain=.01))
        network = FeaturePoolLayer(incoming=network, pool_size=2,axis=1, pool_function=theano.tensor.max)
        return NonlinearityLayer(ElemwiseSumLayer([identity, network.input_layer]), nonlinearity=rectify)


    def get_network(self):
        network = lasagne.layers.InputLayer(shape=(None, self.num_features),input_var=self.input_var)
        network = DenseLayer(network,nonlinearity=rectify,num_units=self.num_nodes,W=HeNormal(gain=.01))
        for _ in xrange(0, self.num_layers):
            network = self.add_residual_dense_maxout_block(network, self.num_nodes, self.dropout)
        return lasagne.layers.DenseLayer(network, num_units=2,
                            nonlinearity=lasagne.nonlinearities.softmax)





def train_bagging(df, features, verbose, batch_size=10, num_epochs=999,
                num_layers=2, num_nodes=10,dropout=0.5,learning_rate=0.002,
                momentum=0.5, early_stop_rounds=3, eval_type='log_loss', num_baggs=10):
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


        test_X = np.array(test_df[features], dtype=np.float32)
        test_y = np.array(test_df['won'], dtype=np.int32)

        pred_outputs = []
        [pred_outputs.append([]) for _ in xrange(len(test_y))]

        # Run boosted aggregation!
        for i in xrange(0,num_baggs):
            sampling_df = train_df.sample(n=len(train_df), replace=True, random_state=i*94)

            train_X = np.array(sampling_df[features], dtype=np.float32)
            train_y = np.array(sampling_df['won'], dtype=np.int32)

            val_X = np.array(train_df[features], dtype=np.float32)
            val_y = np.array(train_df['won'], dtype=np.int32)

            maxout_trainer = MaxoutResidual(num_features=len(features),
                                    num_layers=num_layers,
                                    num_nodes=num_nodes,
                                    dropout=dropout,
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
            [pred_outputs[i].append(pred[1]) for i, pred in enumerate(predictions)]

        import math
        tourney_games = games[games['Season'] == test_season]
        tourney_seeds = seeds[seeds['Season'] == test_season]
        bagg_output = open('../output/{}_maxout_tourney_game_predictions.csv'.format(season), 'w+')
        bagg_output.write('team_one,team_two,team_one_pred_mean,team_one_truth,team_two_pred_mean,team_two_truth\n')
        predictions = []
        for i, pred in enumerate(pred_outputs):
            if i%2==0:
                wteam = tourney_games.iloc[[int(math.floor(i/2))]]['Wteam'].unique()[0]
                lteam = tourney_games.iloc[[int(math.floor(i/2))]]['Lteam'].unique()[0]
                wteam_name = teams[teams['Team_Id'] == wteam]['Team_Name'].unique()[0]
                lteam_name = teams[teams['Team_Id'] == lteam]['Team_Name'].unique()[0]
                wteam_seed = tourney_seeds[tourney_seeds['Team'] == wteam]['Seed'].unique()[0]
                lteam_seed = tourney_seeds[tourney_seeds['Team'] == lteam]['Seed'].unique()[0]
                bagg_output.write('{}.{},{}.{},'.format(wteam_seed, wteam_name,
                                                            lteam_seed, lteam_name))
                bagg_output.write('{},{},'.format(np.mean(pred),test_y[i]))
                predictions.append(np.mean(pred))
            else:
                bagg_output.write('{},{}\n'.format(np.mean(pred),test_y[i]))
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
                final_predictions.append((pred['team_one_pred_mean'] + 1.0-pred['team_two_pred_mean']) / 2.0)
                final_y.append(pred['team_one_truth'])
            else:
                final_predictions.append((pred['team_two_pred_mean'] + 1.0-pred['team_one_pred_mean']) / 2.0)
                final_y.append(pred['team_two_truth'])

        print '{} final log loss: {}'.format(season, log_loss(final_y, final_predictions))

        if verbose:
            sys.exit('only one iteration for verbose debugging')



@click.command()
@click.argument('num_nodes', type=click.INT)
@click.argument('num_layers', type=click.INT)
@click.option('-dropout', type=click.FLOAT, default=0.9)
@click.option('-learning_rate', type=click.FLOAT, default=0.001)
@click.option('-momentum', type=click.FLOAT, default=0.5)
@click.option('-eval_type', type=click.STRING, default='log_loss')
@click.option('-batch_size', type=click.INT, default=1)
@click.option('-early_stop', type=click.INT, default=2)
@click.option('-verbose', type=click.BOOL, default=False)
@click.option('-max_epochs', type=click.INT, default=9999)
@click.option('-num_baggs', type=click.INT, default=3)
def run(num_nodes, num_layers, dropout, learning_rate, momentum, eval_type, batch_size, early_stop, verbose, max_epochs, num_baggs):
    for i, s in enumerate(xrange(2003,2017)):
        if i == 0:
            df = pd.read_csv('../data/games/{}_tourney_games.csv'.format(s))
            # maybe integrate previously predicted attributes?
            #extra = pd.read_csv('../output/{}_maxout_tourney_game_predictions.csv'.format(s))
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
    features.remove('pythag')

    train_bagging(df=df, features=features, batch_size=batch_size, num_epochs=max_epochs,
                        num_layers=num_layers, num_nodes=num_nodes, dropout=dropout,
                        learning_rate=learning_rate, momentum=momentum,
                        early_stop_rounds=early_stop, eval_type=eval_type,
                        verbose=verbose, num_baggs=num_baggs)


if __name__ == "__main__":
    run()
