""" Generate Final Games Files for Model Use """

import pandas as pd
import click
import os, sys
import numpy as np


def generate_agg_stats(fps):
    """ Generate aggregate DataFrame from input file list

    Arguments:
    fps - list of stat file paths
    """
    stats = None
    for i, stats_file in enumerate(fps):
        if i == 0:
            stats = pd.read_csv(stats_file)
        else:
            stats = stats.append(pd.read_csv(stats_file))
    return stats


def add_offensive_efficiency(stats):
    """
    100 x Pts / 0.5 * ((Tm FGA + 0.4 * Tm FTA - 1.07 * (Tm ORB / (Tm ORB + Opp DRB))
    * (Tm FGA - Tm FG) + Tm TOV) + (Opp FGA + 0.4 * Opp FTA - 1.07
    * (Opp ORB / (Opp ORB + Tm DRB)) * (Opp FGA - Opp FG) + Opp TOV))
    """
    stats['oe'] = 100 * stats['pts'] / 0.5 * ((stats['fga'] + 0.4 * stats['fta'] - 1.07 \
      * (stats['orb'] / (stats['orb'] + stats['opp_drb'])) * (stats['fga'] - stats['fg']) + stats['tov'])
      + (stats['opp_fga'] + 0.4 * stats['opp_fta'] - 1.07 * (stats['opp_orb'] / (stats['opp_orb'] + stats['drb']))
      * (stats['opp_fga'] - stats['opp_fg']) + stats['opp_tov']))
    return stats


def add_efficient_offensive_production(stats):
    """
    raw_eop = (.76 * ast + pts) * OE
    """
    #stats['oe2'] = (stats['fg'] + stats['ast']) / (stats['fga'] - stats['orb'] + stats['ast'] + stats['tov'])
    stats['raw_eop'] = (.76 * stats['ast'] + stats['pts']) * stats['oe2']
    stats['eop'] = stats['raw_eop'] * (np.sum(stats['pts']) / (stats['oe'] * (stats['pts'] + .76 * stats['ast'])))
    return stats


def add_defensive_efficiency(stats):
    """
    possessions = FGA + 0.475 x FTA - ORB + TO
    de = (Opponent's Points Allowed/ Opponent's Possessions) x 100
    """
    opp_pos = stats['opp_fga'] + 0.475 * stats['opp_fta'] - stats['opp_orb'] + stats['opp_tov']
    stats['de'] = (stats['opp_pts'] / opp_pos) * 100
    return stats


def add_pythag_win_expectation(stats):
    """ Add pythag based on points per minute (modified pythag)

    pythag = pts^x / (pts^x + opp_pts^x)
    where x is 16.5 as determined in A Starting Point for Analyzing Basketball
    """
    stats['pythag'] = stats['ppm']**16.5 / (stats['ppm']**16.5 + stats['opp_ppm']**16.5)
    return stats


def add_rebounding_percentages(stats):
    """ Add four factors from 'starting point' article

    oreb% = oreb / (oreb+dreb)
    dreb% = dreb / (oreb+dreb)
    """
    stats['or%'] = stats['orpm'] / (stats['orpm'] + stats['drpm'])
    stats['dr%'] = stats['drpm'] / (stats['orpm'] + stats['drpm'])
    return stats


def add_ast_to_tov(stats):
    """
    """
    stats['ast/tov'] = stats['ast_pm'] / stats['to_pm']
    return stats


def add_turnovers_per_possession(stats):
    """
    topp = to/poss
    where poss = FGA + 0.5 * FTA - OREB + TO
    """
    stats['fgapm'] = stats['fga'] / stats['mp']
    stats['possessionspm'] = ((stats['fgapm'] + 0.5) * stats['ftapm']) - stats['orpm'] + stats['tovpm']
    stats['topp'] = stats['tovpm'] / stats['possessionspm']
    return stats


def convert_stats_to_ranks(data):
    """ Convert stats and metrics to rank for the season

    Arguments:
    data - pandas dataframe of team season stats
    """
    for key in data.keys():
        if key not in ('kaggle_id', 'team_name', 'season'):
            data[key] = data[key].rank(pct=True)
    return data

def create_games(stats, games, season, game_type):
    """
    """
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

    for game in games[games['Season'] == season].iterrows():
        game = game[1]
        wteam = stats[(stats['kaggle_id'] == game['Wteam']) & (stats['season'] == season)]
        lteam = stats[(stats['kaggle_id'] == game['Lteam']) & (stats['season'] == season)]

        wkaggle = wteam['kaggle_id'].unique()[0]
        lkaggle = lteam['kaggle_id'].unique()[0]

        wstats = wteam[features].values[0]
        lstats = lteam[features].values[0]

        output.write('1,')
        output.write(','.join(str(x) for x in wstats))
        output.write(',')
        output.write(','.join(str(x) for x in lstats))
        output.write('\n')

        output.write('0,')
        output.write(','.join(str(x) for x in lstats))
        output.write(',')
        output.write(','.join(str(x) for x in wstats))
        output.write('\n')

        diff_stats = wstats-lstats
        diff_output.write('1,')
        diff_output.write(','.join(str(x) for x in diff_stats))
        diff_output.write('\n')

        diff_stats = lstats-wstats
        diff_output.write('0,')
        diff_output.write(','.join(str(x) for x in diff_stats))
        diff_output.write('\n')

    output.close()


def filter_biased_stats(stats):
    """ Return dataframe without biased stats

    Most total stats indicate team advancement
    """
    del stats['opp_trb']
    return stats


def filter_out_nontourney_teams(stats, games):
    """

    Remove bias inherent in nontourney teams
    - including bad team stats inflates all tournament team stats
    """
    tourney_teams = []
    [tourney_teams.append(t) for t in games['Wteam']]
    [tourney_teams.append(t) for t in games['Lteam']]
    tourney_teams = set(tourney_teams)

    return stats[stats['kaggle_id'].isin(tourney_teams)]

@click.command()
@click.argument('game_type')
def run(game_type):
    """ Generate final games file for model use

    Arguments:
    box - pandas DataFrame of yearly box-score stats for each team
    adv - pandas DataFrame of yearly advanced metric stats for each team
    """
    if game_type == 'regular_season':
        games = pd.read_csv('data/original/RegularSeasonDetailedResults.csv')
    elif game_type == 'tourney':
        games = pd.read_csv('data/original/TourneyDetailedResults.csv')
    else:
        sys.exit('{} is an unknown game type'.format(game_type))

    for season in games['Season'].unique():
        print season
        stats = pd.read_csv('data/intermediate/team_regular_season_stats.csv')
        stats = stats[stats['season'] == season]
        #stats = filter_out_nontourney_teams(stats, games[games['Season'] == season])

        stats['fg_pct'] = stats['fgm']/stats['fga']
        stats['fg3_pct'] = stats['fgm3']/stats['fga3']
        stats['opp_fg_pct'] = stats['opp_fgm']/stats['opp_fga']
        stats['opp_fg3_pct'] = stats['opp_fgm3']/stats['opp_fga3']
        stats['ast/to'] = stats['ast']/stats['to']

        stats_to_convert = ['fgm','fga','fgm3','fga3','ast','blk','ftm','fta',
                            'stl','score','or','dr','pf','to']
        for s in stats_to_convert:
            stats['{}_pm'.format(s)] = stats['{}'.format(s)]/stats['minutes_played']
            stats['opp_{}_pm'.format(s)] = stats['opp_{}'.format(s)]/stats['minutes_played']
            del stats['{}'.format(s)]
            del stats['opp_{}'.format(s)]
        del stats['minutes_played']

        #stats = add_turnovers_per_possession(stats)
        #stats = add_rebounding_percentages(stats)
        #stats = add_pythag_win_expectation(stats)
        #stats = add_offensive_efficiency(stats)
        #stats = add_defensive_efficiency(stats)
        #stats = add_efficient_offensive_production(stats)

        stats.to_csv('data/final/{}.csv'.format(season), index=None)

        create_games(stats, games, season, game_type)


if __name__ == '__main__':
    run()
