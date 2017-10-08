

def min_max_normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        if feature_name in ('kaggle_id', 'team_name', 'season'): continue
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def get_per_minute_stats(team_wins, team_losses):
    """ Return combined per minute stat differentials for team
    Arguments:
    team_wins - pandas DataFrame of won games for team in season
    team_losses - pandas DataFrame of lost games for team in season
    """
    output = {}

    for stat in team_wins.keys():
        if stat[0] != 'W' or 'team' in stat or 'loc' in stat: continue
        stat = stat[1:]
        output['{}'.format(stat)] = float(get_team_stat(team_wins,team_losses,stat) - get_opp_stat(team_wins,team_losses,stat)) / float(get_minutes_played(team_wins, team_losses))
    return output


### Four Factors
# Metrics that reduce stats down to factors of winning.
# Theorized to not weigh the same with regard to winning.
#
def get_effective_field_goal_pct(team_wins, team_losses):
    """ Return eFG%
    (FGM + 0.5 * 3PM) / FGA

    Should this be opponent differentiated?
    """
    fgm = team_wins['Wfgm'].sum() + team_losses['Lfgm'].sum()
    fgm3 = team_wins['Wfgm3'].sum() + team_losses['Lfgm3'].sum()
    fga = team_wins['Wfga'].sum() + team_losses['Lfga'].sum()
    return (fgm + 0.5 + fgm3) / fga

def get_turnovers_per_possession(team_wins, team_losses):
    """ Return TOVpp
    tov/poss
    """
    pass

def get_offensive_rebounding_pct(team_wins, team_losses):
    """ Return offensive rebound percentage estimate """
    pass

def get_free_throw_rate(team_wins, team_losses):
    """ Return free throw rate """
    pass
### End Four Factors
