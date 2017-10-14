#!/usr/bin/env python
""" Generate Final Games Files for Model Use """

# conda execute
# env:
#  - python >=2
#  - beautifulsoup4
#  - pandas
#  - tqdm
# run_with: python2


import pandas as pd
import click
import os, sys
import numpy as np

from tqdm import tqdm

GAMES = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'data/raw/TourneyDetailedResults.csv'))


def min_max_rpi_normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        if feature_name in ('kaggle_id', 'team_one', 'team_two', 'season', 'rpi', 'team_name', 'won'): continue
        result[feature_name] = result[feature_name] / result['rpi']
        max_value = result[feature_name].max()
        min_value = result[feature_name].min()

        result[feature_name] = result[feature_name].apply(lambda x:  (x - min_value) / (max_value - min_value))
    return result



def create_classification_games(stats, games, season, game_type):
    output = open('data/games/{}_{}_games.csv'.format(season, game_type), 'w+')
    features = [f for f in stats.keys() if f not in ['team_name', 'season', 'kaggle_id']]
    header = [f for f in stats.keys() if f not in ['team_name', 'season', 'kaggle_id']]
    [header.append('_{}'.format(f)) for f in stats.keys() if f not in ['team_name', 'season', 'kaggle_id']]
    output.write('won,')
    output.write(','.join(header))
    output.write('\n')

    stats = convert_stats_to_ranks(stats)

    diff_output = open('data/games/{}_{}_diff_games.csv'.format(season, game_type), 'w+')
    diff_output.write('won,')
    header = [f for f in stats.keys() if f not in ['team_name', 'season', 'kaggle_id']]
    diff_output.write(','.join([s for s in header]))
    diff_output.write('\n')

    for game in tqdm(games[games['Season'] == season].iterrows()):
        game = game[1]
        wteam = stats[(stats['kaggle_id'] == game['Wteam']) & (stats['season'] == season)]
        lteam = stats[(stats['kaggle_id'] == game['Lteam']) & (stats['season'] == season)]

        try:
            wkaggle = wteam['kaggle_id'].unique()[0]
            lkaggle = lteam['kaggle_id'].unique()[0]
        except:
            continue

        wstats = wteam[features].values[0]
        lstats = lteam[features].values[0]

        if wkaggle > lkaggle:

            output.write('1,')
            output.write(','.join(str(x) for x in wstats))
            output.write(',')
            output.write(','.join(str(x) for x in lstats))
            output.write('\n')
        else:

            output.write('0,')
            output.write(','.join(str(x) for x in lstats))
            output.write(',')
            output.write(','.join(str(x) for x in wstats))
            output.write('\n')

        #if wkaggle > lkaggle:

        #    diff_stats = wstats-lstats
        #    diff_output.write('1,')
        #    diff_output.write(','.join(str(x) for x in diff_stats))
        #    diff_output.write('\n')
        #else:

        #    diff_stats = lstats-wstats
        #    diff_output.write('0,')
        #    diff_output.write(','.join(str(x) for x in diff_stats))
        #    diff_output.write('\n')

    output.close()


def create_base_game_examples(stats, output_fp):
    """ """
    output = []
    for i, season in enumerate(tqdm(stats['season'].unique())):

        season_stats = stats[stats['season'] == season]
        season_games = GAMES[GAMES['Season'] == season]

        for k, game in season_games.iterrows():
            game_entry = {}

            if int(game['Wteam']) > int(game['Lteam']):
                game_entry['won'] = 1
                team_one = season_stats[season_stats['kaggle_id'] == game['Wteam']]
                team_two = season_stats[season_stats['kaggle_id'] == game['Lteam']]
            else:
                game_entry['won'] = 0
                team_one = season_stats[season_stats['kaggle_id'] == game['Lteam']]
                team_two = season_stats[season_stats['kaggle_id'] == game['Wteam']]

            game_entry['season'] = season
            game_entry['team_one'] = team_one['team_name'].unique()[0]
            game_entry['team_one_kaggle'] = team_one['kaggle_id'].unique()[0]
            game_entry['team_two'] = team_two['team_name'].unique()[0]
            game_entry['team_two_kaggle'] = team_two['kaggle_id'].unique()[0]

            # Per minute stats
            for stat_name in season_stats.columns:
                if stat_name in ('season', 'team_name', 'kaggle_id', 'minutes_played'): continue
                game_entry[stat_name] = team_one[stat_name].unique()[0] / team_one['minutes_played'].unique()[0]
                game_entry['opp_{}'.format(stat_name)] = team_two[stat_name].unique()[0] / team_one['minutes_played'].unique()[0]

            if type(output) == type(None):
                output = pd.DataFrame(game_entry, index=[i+k])
            else:
                output = output.append(pd.DataFrame(game_entry, index=[i+k]))

        output[output['season'] == season] = min_max_rpi_normalize(output[output['season'] == season])
    del output['rpi']

    output.to_csv(output_fp, index=None)


def create_advanced_game_examples():
    pass


@click.command()
@click.argument('base_input_fp', default='data/intermediate/base_team_stats.csv')
@click.argument('advanced_input_fp', default='data/intermediate/advanced_team_stats.csv')
@click.argument('output_dp', default='data/final')
def run(base_input_fp, advanced_input_fp, output_dp):
    """ Generate final games for model training and testing

    Depends on correctly generated base and advanced statistics CSV files
    """
    if not os.path.exists(output_dp):
        os.mkdir(output_dp)

    if not os.path.exists(base_input_fp): sys.exit('Need to generate base input data!')

    stats = pd.read_csv(base_input_fp)

    create_base_game_examples(stats=stats, output_fp=os.path.join(output_dp, 'base_games.csv'))


    if not os.path.exists(advanced_input_fp): sys.exit('Need to generate advanced input data!')


if __name__ == '__main__':
    run()
