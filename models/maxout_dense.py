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

from maxout_new import Maxout


class MaxoutDense(Maxout):
    """ Maxout Dense Network """
    def add_dense_maxout_block(self, network, num_nodes=240, dropout_p=0.5):
        network = lasagne.layers.DropoutLayer(network, p=self.dropout_p)
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
                    network = batch_norm(self.add_dense_maxout_block(network, self.num_nodes, self.dropout_p))
                    layers.append(network)
                    network = ConcatLayer(layers, axis=1)
        maxout = FeaturePoolLayer(incoming=network, pool_size=2,axis=1, pool_function=theano.tensor.mean)
        return lasagne.layers.DenseLayer(network, num_units=2,nonlinearity=lasagne.nonlinearities.softmax)




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
        for i in xrange(0,6):
            sampling_df = train_df.sample(n=len(train_df), replace=True, random_state=i*94)

            train_X = np.array(scaler.fit_transform(sampling_df[features]), dtype=np.float32)
            train_y = np.array(sampling_df['won'], dtype=np.int32)

            val_X = np.array(scaler.fit_transform(train_df[features]), dtype=np.float32)
            val_y = np.array(train_df['won'], dtype=np.int32)


            maxout_trainer = MaxoutDense(num_features=len(features),
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
            [pred_outputs[i].append(pred[1]) for i, pred in enumerate(predictions)]

        import math
        tourney_games = games[games['Season'] == test_season]
        tourney_seeds = seeds[seeds['Season'] == test_season]
        bagg_output = open('../output/{}_maxout_tourney_game_predictions.csv'.format(season), 'w+')
        bagg_output.write('wteam,lteam,wpred_var,wpred_mean,w_truth,lpred_var,lpred_mean,l_truth\n')
        final_predictions = []
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
                bagg_output.write('{},{},{},'.format(np.var(pred),np.mean(pred),test_y[i]))
                final_predictions.append(np.mean(pred))
            else:
                bagg_output.write('{},{},{}\n'.format(np.var(pred),np.mean(pred),test_y[i]))
                final_predictions.append(np.mean(pred))
        bagg_output.close()

        from sklearn.metrics import log_loss

        print '{} final log loss: {}'.format(season, log_loss(test_y, final_predictions))

        if verbose:
            sys.exit('only one iteration for verbose debugging')



@click.command()
@click.argument('num_nodes', type=click.INT)
@click.argument('num_layers', type=click.INT)
@click.option('-dropout', type=click.FLOAT, default=0.2)
@click.option('-learning_rate', type=click.FLOAT, default=0.001)
@click.option('-momentum', type=click.FLOAT, default=0.5)
@click.option('-eval_type', type=click.STRING, default='log_loss')
@click.option('-batch_size', type=click.INT, default=200)
@click.option('-early_stop', type=click.INT, default=3)
@click.option('-verbose', type=click.BOOL, default=False)
@click.option('-max_epochs', type=click.INT, default=9999)
def run(num_nodes, num_layers, dropout, learning_rate, momentum, eval_type, batch_size, early_stop, verbose, max_epochs):
    for i, s in enumerate((2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016)):
        if i == 0:
            df = pd.read_csv('../data/games/{}_tourney_diff_games.csv'.format(s))
            # maybe integrate previously predicted attributes?
            #extra = pd.read_csv('../output/{}_maxout_tourney_game_predictions.csv'.format(s))
            df['season'] = s

        else:
            tmp = pd.read_csv('../data/games/{}_tourney_diff_games.csv'.format(s))
            tmp['season'] = s
            df = df.append(tmp)

    features = df.keys().tolist()
    features.remove('season')
    #features.remove('team_name')
    features.remove('won')

    train_bagging(df=df, features=features, batch_size=batch_size, num_epochs=max_epochs,
                        num_layers=num_layers, num_nodes=num_nodes, dropout=dropout,
                        learning_rate=learning_rate, momentum=momentum,
                        early_stop_rounds=early_stop, eval_type=eval_type,
                        verbose=verbose)


if __name__ == "__main__":
    run()
