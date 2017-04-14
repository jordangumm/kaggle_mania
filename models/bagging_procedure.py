""" Bagging Procedure

"""

import sys

import pandas as pd
import numpy as np
import click

from maxout import Maxout
from maxout_residual import MaxoutResidual
from maxout_dense import MaxoutDense


def train_with_bagging(df, features, verbose, batch_size, num_epochs,
                    num_layers, num_nodes,dropout,learning_rate,
                    momentum, early_stop_rounds, num_baggs,
                    maxout_type='maxout', eval_type='log_loss'):
    """ Train Maxout Networks in boosted aggregation style

    Returns tuple of scores to minimize
    """
    games = pd.read_csv('../data/original/TourneyDetailedResults.csv')
    teams = pd.read_csv('../data/original/Teams.csv')
    seeds = pd.read_csv('../data/original/TourneySeeds.csv')

    maxout_class = None
    if maxout_type == 'maxout':
        maxout_class = Maxout
    elif maxout_type == 'maxout_residual':
        maxout_class = MaxoutResidual
    elif maxout_type == 'maxout_dense':
        maxout_class = MaxoutDense
    else:
        sys.exit("Unknown maxout_type {}".format(maxout_type))

    def normalize(data):
        for key in data.keys():
            if not key in features: continue
            mean = data[key].mean()
            std = data[key].std()
            data.loc[:, key] = data[key].apply(lambda x: x - mean / std)
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

            maxout_trainer = maxout_class(num_features=len(features),
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
@click.option('-dropout', type=click.FLOAT, default=0.5)
@click.option('-learning_rate', type=click.FLOAT, default=0.001)
@click.option('-momentum', type=click.FLOAT, default=0.5)
@click.option('-eval_type', type=click.STRING, default='log_loss')
@click.option('-batch_size', type=click.INT, default=1)
@click.option('-early_stop', type=click.INT, default=7)
@click.option('-verbose', type=click.BOOL, default=False)
@click.option('-max_epochs', type=click.INT, default=9999)
@click.option('-num_baggs', type=click.INT, default=1)
@click.option('-maxout_type', type=click.STRING, default='maxout')
def run(num_nodes, num_layers, dropout, learning_rate, momentum,
        eval_type, batch_size, early_stop, verbose, max_epochs,
                                        num_baggs, maxout_type):
    for i, s in enumerate(xrange(2003,2017)):
        if i == 0:
            df = pd.read_csv('../data/games/{}_tourney_diff_games.csv'.format(s))
            df['season'] = s
        else:
            tmp = pd.read_csv('../data/games/{}_tourney_diff_games.csv'.format(s))
            tmp['season'] = s
            df = df.append(tmp)

    df = df.fillna(0.0)

    features = df.keys().tolist()
    features.remove('season')
    #features.remove('team_name')
    features.remove('won')

    """ Features removed due to LIME inspection """
    features.remove('seed')
    #features.remove('_seed')

    train_with_bagging(df=df, features=features, batch_size=batch_size, num_epochs=max_epochs,
                num_layers=num_layers, num_nodes=num_nodes, dropout=dropout,
                learning_rate=learning_rate, momentum=momentum,
                early_stop_rounds=early_stop, eval_type=eval_type,
                verbose=verbose, num_baggs=num_baggs, maxout_type=maxout_type)


if __name__ == "__main__":
    run()
