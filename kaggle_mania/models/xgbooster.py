""" XGBoost Feature Selection

"""

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.metrics import *
import click


def get_optimal_rounds(dtrain, param):
    """ Return int representing best rounds to avoid overfitting

    Use cross validation xgboost module with early stopping

    Arguments:
    param - dictionary of parameters to pass to cv
    """
    num_round = 1000
    bst = xgb.cv(param, dtrain, num_round, nfold=10,
                metrics={'logloss', 'auc'}, seed=0,
                callbacks=[xgb.callback.print_evaluation(show_stdv=True),
                    xgb.callback.early_stop(10)])
    return len(bst)-1


def train_xgboost(df, features):
    for season in xrange(2013,2017):
        test_season = season

        tmp_df = df[df['season'] <= test_season]

        test_df = tmp_df[tmp_df['season'] == test_season]

        train_df = tmp_df[tmp_df['season'] != test_season]

        print 'test season: {}'.format(test_df['season'].unique()[0])
        print 'train seasons: {}-{}'.format(min(train_df['season'].unique()), max(train_df['season'].unique()))
        print 'num features: {}'.format(len(features))

        test_X = np.array(test_df[features], dtype=np.float32)
        test_y = np.array(test_df['won'], dtype=np.int32)

        train_X = np.array(train_df[features], dtype=np.float32)
        train_y = np.array(train_df['won'], dtype=np.int32)


        dtrain = xgb.DMatrix(train_X, train_y, feature_names=features)
        dtest = xgb.DMatrix(test_X, test_y, feature_names=features)


        param = {'max_depth':100, 'eta':0.01, 'silent':1, 'objective':'binary:logistic'}
        num_round = get_optimal_rounds(dtrain=dtrain, param=param)

        evallist  = [(dtrain,'train')]
        param['nthread'] = 4
        param['eval_metric'] = 'logloss'
        plst = param.items()
        bst = xgb.train(plst, dtrain, num_round, evallist)
        train_predictions = bst.predict(dtrain)
        test_predictions = bst.predict(dtest)


        print("  train loss:\t\t{:.6f}".format(log_loss(train_y, train_predictions)))
        print("  train accuracy:\t{:.6f}".format(accuracy_score(train_y, [round(x) for x in train_predictions])))
        print("  train auc:\t\t{:.6f}".format(roc_auc_score(train_y, train_predictions)))
        print
        print("  test loss:\t\t{:.6f}".format(log_loss(test_y, test_predictions)))
        print("  test accuracy:\t{:.6f}".format(accuracy_score(test_y, [round(x) for x in test_predictions])))
        print("  test auc:\t\t{:.6f}".format(roc_auc_score(test_y, test_predictions)))
        print


if __name__ == "__main__":
    for i, s in enumerate(xrange(2003,2017)):
        if i == 0:
            df = pd.read_csv('../data/games/{}_tourney_diff_games.csv'.format(s))
            # maybe integrate previously predicted attributes?
            #extra = pd.read_csv('../output/{}_maxout_tourney_game_predictions.csv'.format(s))
            df['season'] = s
        else:
            tmp = pd.read_csv('../data/games/{}_tourney_diff_games.csv'.format(s))
            tmp['season'] = s
            df = df.append(tmp)

    df = df.fillna(0.0)

    features = df.keys().tolist()
    features.remove('season')
    features.remove('won')
    features.remove('pythag')

    df[features] = df[features].rank(pct=True)

    print train_xgboost(df=df, features=features)
