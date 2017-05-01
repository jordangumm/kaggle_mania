""" Bagging Procedure

"""

import sys

import pandas as pd
import numpy as np
import click

from models.maxout import Maxout
from models.maxout_residual import MaxoutResidual
from models.maxout_dense import MaxoutDense

from sklearn.metrics import log_loss
from sklearn.model_selection import KFold


def train_with_bagging(train_df, features, verbose, batch_size, num_epochs,
                                num_layers, num_nodes, dropout_p, learning_rate,
                                early_stop_rounds, num_baggs, weight_decay,
                                maxout_type='maxout', eval_type='log_loss'):
    """ Train Maxout Networks in boosted aggregation style

    Returns tuple of scores to minimize
    """
    test_season = train_df['season'].max()+1
    if maxout_type == 'maxout':
        maxout_class = Maxout
    elif maxout_type == 'maxout_residual':
        maxout_class = MaxoutResidual
    elif maxout_type == 'maxout_dense':
        maxout_class = MaxoutDense
    else:
        sys.exit("Unknown maxout_type {}".format(maxout_type))

    print 'test season: {}'.format(test_season)
    print 'train seasons: {}-{}'.format(min(train_df['season'].unique()), max(train_df['season'].unique()))
    print 'num features: {}'.format(len(features))

    kf = KFold(n_splits=10)
    kf.get_n_splits(train_df)

    holdout_losses = []
    bayes_holdout_losses = []
    holdout_num = 0
    for train_index, test_index in kf.split(train_df):

        # bagging uses original training set as validation set
        X = train_df[features]
        y = train_df['won']

        val_df = train_df.iloc[train_index]
        test_df = train_df.iloc[test_index]

        test_X = np.array(test_df[features], dtype=np.float32)
        test_y = np.array(test_df['won'], dtype=np.int32)

        pred_outputs = []
        [pred_outputs.append([]) for _ in xrange(len(test_df))]

        bayes_pred_outputs = []
        [bayes_pred_outputs.append([]) for _ in xrange(len(test_df))]

        # Run boosted aggregation!
        for bag_iteration in xrange(0,num_baggs):
            sampling_df = val_df.sample(n=len(val_df), replace=True, random_state=bag_iteration*94)

            train_X = np.array(sampling_df[features], dtype=np.float32)
            train_y = np.array(sampling_df['won'], dtype=np.int32)

            val_X = np.array(val_df[features], dtype=np.float32)
            val_y = np.array(val_df['won'], dtype=np.int32)

            maxout_trainer = maxout_class(num_features=len(features),
                                    num_layers=num_layers,
                                    num_nodes=num_nodes,
                                    dropout=dropout_p,
                                    learning_rate=learning_rate,
                                    verbose=verbose,
                                    weight_decay=weight_decay)
            maxout_trainer.fit(train_X=train_X,
                                train_y=train_y,
                                val_X=val_X,
                                val_y=val_y,
                                features=features,
                                early_stop_rounds=early_stop_rounds,
                                batch_size=batch_size)

            iter_preds = maxout_trainer.predict_proba(test_X)
            bayes_iter_preds = maxout_trainer.predict_bayes_proba(test_X)

            #print 'holdout loss: {}\tholdout bayes loss: {}'.format(log_loss(test_y, iter_preds),
            #                                                      log_loss(test_y, bayes_iter_preds))

            #from lime.lime_tabular import LimeTabularExplainer
            #explainer = LimeTabularExplainer(train_X, feature_names=features, class_names=['lost', 'won'], discretize_continuous=True)
            #for game_num, test_example in enumerate(test_X):
            #    exp = explainer.explain_instance(test_example, maxout_trainer.predict_proba, num_features=len(features))
            #    exp.save_to_file('../output/{}/{}_explanation.html'.format(season, game_num))

            [pred_outputs[pred_num].append(pred[1]) for pred_num, pred in enumerate(iter_preds)]
            [bayes_pred_outputs[pred_num].append(pred) for pred_num, pred in enumerate(bayes_iter_preds)]


        final_preds = [np.mean(p) for p in pred_outputs]
        final_bayes_preds = [np.mean(p) for p in bayes_pred_outputs]
        print 'holdout {} loss:\t{}'.format(holdout_num, log_loss(test_y, final_preds))
        print 'holdout {} bayes loss:\t{}'.format(holdout_num, log_loss(test_y, final_bayes_preds))
        print
        holdout_losses.append(log_loss(test_y, final_preds))
        bayes_holdout_losses.append(log_loss(test_y, final_bayes_preds))
        holdout_num += 1

    if verbose:
        sys.exit('only one iteration for verbose debugging')
    return np.mean(holdout_losses), np.mean(bayes_holdout_losses)


@click.command()
@click.argument('num_nodes', type=click.INT)
@click.argument('num_layers', type=click.INT)
@click.option('-dropout', type=click.FLOAT, default=0.263817749011)
@click.option('-learning_rate', type=click.FLOAT, default=0.0331640958066)
@click.option('-eval_type', type=click.STRING, default='log_loss')
@click.option('-batch_size', type=click.INT, default=3)
@click.option('-early_stop', type=click.INT, default=7)
@click.option('-verbose', type=click.BOOL, default=False)
@click.option('-max_epochs', type=click.INT, default=9999)
@click.option('-num_baggs', type=click.INT, default=1)
@click.option('-maxout_type', type=click.STRING, default='maxout')
@click.option('-test_season', type=click.INT, default=2008)
def run(num_nodes, num_layers, dropout, learning_rate,
        eval_type, batch_size, early_stop, verbose, max_epochs,
                            num_baggs, maxout_type, test_season):
    for i, s in enumerate(xrange(2010,2017)):
        if i == 0:
            df = pd.read_csv('data/games/{}_tourney_diff_games.csv'.format(s))
            df['season'] = s
        else:
            tmp = pd.read_csv('data/games/{}_tourney_diff_games.csv'.format(s))
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

    def normalize(df):
        df[features] = (df[features]-df[features].mean())/df[features].std()
        #for key in df.keys():
        #    if not key in features: continue
        #    df[key] = df[key]/df[key].loc[df.abs().idxmax()].astype(np.float64)
        #    df[key]=(df-df.mean())/df.std()

            #mean = data[key].mean()
            #std = data[key].std()
            #data.loc[:, key] = data.loc[:, key].apply(lambda x: x - mean / std)
            #data.loc[:, key] = data.loc[:, key].apply(lambda x: (x - data[key].min()) / (data[key].max() - data[key].min()))
        return df

    train_df = normalize(df[df['season'] < test_season])
    test_df = normalize(df[df['season'] == test_season])

    #train_df = df[df['season'] < test_season]
    #test_df = df[df['season'] == test_season]

    train_with_bagging(train_df=train_df, test_df=test_df, features=features,
                batch_size=batch_size, num_epochs=max_epochs,
                num_layers=num_layers, num_nodes=num_nodes, dropout_p=dropout,
                learning_rate=learning_rate, weight_decay=0.0586784183643,
                early_stop_rounds=early_stop, eval_type=eval_type,
                verbose=verbose, num_baggs=num_baggs, maxout_type=maxout_type)


if __name__ == "__main__":
    run()
