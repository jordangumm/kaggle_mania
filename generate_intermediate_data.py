"""

TODO:
Home vs Neutral vs Away winning percentage over the regular season?

"""


import pandas as pd
import numpy as np
import os


games = pd.read_csv('data/original/RegularSeasonDetailedResults.csv')
team_names = pd.read_csv('data/original/Teams.csv')


def get_per_minute_stat(team_wins, team_losses, stat):
    """
    """
    return team_wins['W{}'.format(stat)].sum() + team_losses['L{}'.format(stat)].sum()


def get_per_minute_opp_stat(team_wins, team_losses, stat):
    """
    """
    return team_wins['L{}'.format(stat)].sum() + team_losses['W{}'.format(stat)].sum()


def get_minutes_played(team_wins, team_losses):
    mp = (len(team_wins) * 40) + (team_wins['Numot'].sum() * 5)
    mp += (len(team_losses) * 40) + (team_losses['Numot'].sum() * 5)
    return mp


def get_per_minute_stats(team_wins, team_losses):
    """ Return combined per minute stats for team
    Arguments:
    team_wins - pandas DataFrame of won games for team in season
    team_losses - pandas DataFrame of lost games for team in season
    """
    output = {}

    for stat in team_wins.keys():
        if stat[0] != 'W' or 'team' in stat or 'loc' in stat: continue
        stat = stat[1:]
        output['{}'.format(stat)] = get_per_minute_stat(team_wins,team_losses,stat)
        output['opp_{}'.format(stat)] = get_per_minute_opp_stat(team_wins,team_losses,stat)
    return output


def get_rpi(season_games, team):
    """ A rank based on team's wins and losses plus strength of schedule

    RPI = (WP * 0.25) + (OWP * 0.50) + (OOWP * 0.25)
    """
    def get_wp(team):
        """ Return team win percentage """
        team_wins = season_games[season_games['Wteam'] == team]
        team_losses = season_games[season_games['Lteam'] == team]
        return float(len(team_wins)) / float(len(team_losses) + len(team_wins))

    def get_opponents(team):
        """ Return list of all opponents """
        team_wins = season_games[season_games['Wteam'] == team]
        team_losses = season_games[season_games['Lteam'] == team]
        opponents = [] # to be a list of all game opponents
        [opponents.append(t) for t in team_wins['Lteam']]
        [opponents.append(t) for t in team_losses['Wteam']]
        return opponents

    def get_owp(team):
        """ Return float average opponent win percentage """
        opponents = get_opponents(team)
        opponent_wps = [get_wp(t) for t in opponents]
        return float(np.sum(opponent_wps)) / float(len(opponents))

    def get_oowp(team):
        """ Return float opponent's average opponent win percentage """
        opponents = get_opponents(team)
        opponent_owps = [get_owp(t) for t in opponents]
        return float(np.sum(opponent_owps)) / float(len(opponents))

    wp = get_wp(team)
    owp = get_owp(team)
    oowp = get_oowp(team)
    return (wp * 0.25) + (owp * 0.50) + (oowp * 0.25)

print games.keys()

output = None
for i, season in enumerate(games['Season'].unique()):
    print season
    season_games = games[games['Season'] == season]
    teams = set(season_games['Wteam'].unique().tolist() + season_games['Lteam'].unique().tolist())

    for k, team in enumerate(teams):
        team_wins = season_games[season_games['Wteam'] == team]
        team_losses = season_games[season_games['Lteam'] == team]

        team_stats = get_per_minute_stats(team_wins, team_losses)
        team_stats['minutes_played'] = get_minutes_played(team_wins, team_losses)
        team_stats['rpi'] = get_rpi(season_games, team)
        team_stats['avg_spread'] = float((team_wins['Wscore'].sum() - team_wins['Lscore'].sum()) + (team_losses['Lscore'].sum() - team_losses['Wscore'].sum())) / float(len(team_wins) + len(team_losses))
        team_stats['kaggle_id'] = team
        team_stats['team_name'] = team_names[team_names['Team_Id'] == team]['Team_Name'].unique()[0]
        team_stats['season'] = season

        print '{}: {}'.format(team_stats['team_name'], team_stats['rpi'])

        if type(output) == type(None):
            output = pd.DataFrame(team_stats, index=[i+k])
        else:
            output = output.append(pd.DataFrame(team_stats, index=[i+k]))
output_dp = 'data/intermediate/'
if not os.path.isdir(output_dp):
    os.mkdir(output_dp)

output.to_csv(os.path.join(output_dp, 'team_regular_season_stats.csv'), index=None)
