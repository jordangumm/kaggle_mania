#!/usr/bin/env python
""" Parse HTML data to generate box score statistics """

# conda execute
# env:
#  - python >=2
#  - beautifulsoup4
#  - pandas
#  - tqdm
# run_with: python2


import pandas as pd
import math
import csv
import os, sys
import re

from bs4 import BeautifulSoup
from tqdm import tqdm


def convert_html_tables_to_csv():
    """ Pull out all html tables and save them as csv with id name

    """
    html_path = './data/html'

    def cell_text(cell):
        return " ".join(cell.stripped_strings)

    for season in tqdm(xrange(2010,2018)):
        for f in tqdm(os.listdir(os.path.join(html_path, str(season)))):
            html_fp = os.path.join(html_path, str(season), f)
            html = open(html_fp).read()
            if 'Page Not Found (404 error)' in html:
                continue
            soup = BeautifulSoup(html, 'lxml')

            for table in soup.findAll('table'):
                output_fp = './data/intermediate/{}/{}/{}'.format(season,table['id'],f.replace('html','csv'))
                if not os.path.exists(os.path.dirname(output_fp)):
                    os.makedirs(os.path.dirname(output_fp))
                output = csv.writer(open(output_fp, 'w+'))
                for row in table.findAll('tr'):
                    col = map(cell_text, row.find_all(re.compile('t[dh]')))
                    output.writerow(col)
                output.writerow([])


def get_stat_balance(player_df, team_df, team_id, stat):
    """ Return float representing team stat balance

    PB = h(PT)/h(MP)
    """
    team_stat = float(team_df[stat].unique()[0])
    team_mp = float(team_df['MP'].unique()[0])

    h_st = 0.0
    h_mp = 0.0
    for i, player in player_df.iterrows():
        player_stat = float(player[stat])
        player_mp = float(player['MP'])

        if 0.0 in [player_stat, player_mp]: continue
        h_st += (player_stat/team_stat)*math.log(player_stat/team_stat)
        h_mp += (player_mp/team_mp)*math.log(player_mp/team_mp)
    return h_st/h_mp


def generate_player_metrics():
    """ output player_regular_season_stats.csv

    """
    data_path = './data/intermediate'
    output_df = None
    for season in tqdm(xrange(2010,2018)):
	
        for f in tqdm(os.listdir(os.path.join(data_path, str(season), 'per_game'))):
            metrics = {}
            metrics['kaggle_id'] = int(f.split('_')[-1].split('.')[0])
            metrics['season'] = season

            player_df = pd.read_csv(os.path.join(data_path, str(season), 'per_game', f))
            team_df = pd.read_csv(os.path.join(data_path, str(season), 'team_stats', f))
            team_df = team_df[team_df['Unnamed: 0'] == 'Team']

            metrics['pts_balance'] = get_stat_balance(player_df, team_df, metrics['kaggle_id'], 'PTS')
            metrics['pf_balance'] = get_stat_balance(player_df, team_df, metrics['kaggle_id'], 'PF')
            metrics['to_balance'] = get_stat_balance(player_df, team_df, metrics['kaggle_id'], 'TOV')
            metrics['blk_balance'] = get_stat_balance(player_df, team_df, metrics['kaggle_id'], 'BLK')
            metrics['stl_balance'] = get_stat_balance(player_df, team_df, metrics['kaggle_id'], 'STL')
            metrics['ast_balance'] = get_stat_balance(player_df, team_df, metrics['kaggle_id'], 'AST')
            metrics['trb_balance'] = get_stat_balance(player_df, team_df, metrics['kaggle_id'], 'TRB')
            #metrics['fta_balance'] = get_stat_balance(player_df, team_df, metrics['kaggle_id'], 'FTA')
            metrics['ft_balance'] = get_stat_balance(player_df, team_df, metrics['kaggle_id'], 'FT')
            #metrics['3pa_balance'] = get_stat_balance(player_df, team_df, metrics['kaggle_id'], '3PA')
            metrics['3p_balance'] = get_stat_balance(player_df, team_df, metrics['kaggle_id'], '3P')
            #metrics['2pa_balance'] = get_stat_balance(player_df, team_df, metrics['kaggle_id'], '2PA')
            metrics['2p_balance'] = get_stat_balance(player_df, team_df, metrics['kaggle_id'], '2P')
            #metrics['fga_balance'] = get_stat_balance(player_df, team_df, metrics['kaggle_id'], 'FGA')
            #metrics['fg_balance'] = get_stat_balance(player_df, team_df, metrics['kaggle_id'], 'FG')

            if type(output_df) == type(None):
                output_df = pd.DataFrame(metrics, index=xrange(0,len(metrics.keys())))
            else:
                output_df = output_df.append(pd.DataFrame(metrics, index=xrange(0,len(metrics.keys()))))
    output_df.drop_duplicates().to_csv(os.path.join(data_path, 'player_metrics.csv'), index=None)


def generate_player_stats():
    """ use per-game stats and output player_stats.csv
    """
    def convert_player_height(height, pos):
        """ Return height in inches
        """
        if type(height) != type(''):
            if pos == 'G':
                return 75
            elif pos == 'F':
                return 79
            else:
                return 82
        ft, inches = height.split('-')
        return int(ft) * 12 + int(inches)

    def convert_player_class(player_class):
        """ Return class as integer
        """
        if player_class == 'FR':
            return 0
        elif player_class == 'SO':
            return 1
        elif player_class == 'JR':
            return 2
        elif player_class == 'SR':
            return 4
        else:
            return 1 # assume sophomore for simplicity

    def convert_player_pos(pos, height):
        """ Return possition as integer
        """
        if pos == 'G':
            return 0
        elif pos == 'F':
            return 1
        elif pos == 'C':
            return 2
        else:
            if height < 76: # 6ft 4in
                return 1
            elif height < 81:
                return 2
            return 3

    data_path = './data/intermediate'
    output_df = None
    for season in xrange(2010,2018):
        for team_num, f in enumerate(os.listdir(os.path.join(data_path, str(season), 'per_game'))):
            metrics = {}
            metrics['kaggle_id'] = int(f.split('_')[-1].split('.')[0])
            metrics['season'] = season

            player_df = pd.read_csv(os.path.join(data_path, str(season), 'per_game', f))
            #player_df = player_df[player_df['G'] > 8]
            roster_df = pd.read_csv(os.path.join(data_path, str(season), 'roster', f))


            player_df = player_df.sort_values(['G'], ascending=False) # sort players by games played
            player_df.index = range(1,len(player_df) + 1)

            for i, player in player_df.iterrows():
                if i > 6: break
                for metric in ('PTS','BLK','STL','AST','TRB','FTA','FT','3PA','3P','2PA','2P','FGA','FG'):
                    metrics['{}_pg_{}'.format(metric.lower(), i)] = player[metric]

                player_class = roster_df[roster_df['Player'] == player['Player']]['Class'].unique()[0]
                player_pos = roster_df[roster_df['Player'] == player['Player']]['Pos'].unique()[0]
                player_height = roster_df[roster_df['Player'] == player['Player']]['Height'].unique()[0]

                metrics['class_{}'.format(i)] = convert_player_class(player_class)
                metrics['pos_{}'.format(i)] = convert_player_pos(player_pos, player_height) # one hot this
                metrics['height_{}'.format(i)] = convert_player_height(player_height, player_pos)

            if type(output_df) == type(None):
                output_df = pd.DataFrame(metrics, index=xrange(0,len(metrics.keys())))
            else:
                output_df = output_df.append(pd.DataFrame(metrics, index=xrange(0,len(metrics.keys()))))
    output_df.drop_duplicates().to_csv(os.path.join(data_path, 'player_stats.csv'), index=None)


convert_html_tables_to_csv()
generate_player_metrics()
#generate_player_stats()
