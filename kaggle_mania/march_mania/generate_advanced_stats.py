import pandas as pd
from tqdm import tqdm
import os, sys
import click



### Four Factors
# Metrics that reduce stats down to factors of winning.
# Theorized to not weigh the same with regard to winning.
#
def calculate_possessions(stats):
    """
    Best estimates are averaged over both team and opponents
    POSS = 0.976 * (FGA + 0.44 * FTAt - OREBt + TOt)
    """
    possessions = (0.976 + (stats['fga'] + 0.44 * stats['fta'] - stats['or'] + stats['to']))
    possessions += (0.976 + (stats['opp_fga'] + 0.44 * stats['opp_fta'] - stats['opp_or'] + stats['opp_to']))
    possessions /= 2.0
    return possessions

def get_effective_field_goal_pct(fgm, fgm3, fga):
    """ Return eFG%
    (FGM + 0.5 * 3PM) / FGA

    Should this be opponent differentiated?
    """
    return float(fgm + 0.5 + fgm3) / float(fga)


def get_turnovers_per_possession(tov, possessions):
    """ Return TOVpp
    tov/poss
    """
    return float(tov) / float(possessions)


def get_offensive_rebounding_pct(orb, opp_drb):
    """ Offensive Rebounding Percentage
    OREBt / (OREBt + DREBo)
    """
    return float(orb) / float(orb + opp_drb)


def get_defensive_rebounding_pct(drb, opp_orb):
    """ Offensive Rebounding Percentage
    DREBt / (DREBt + OREBo)
    """
    return float(drb) / float(drb + opp_orb)


def get_free_throw_rate(free_throws, possessions):
    """
    FTM / POSS
    """
    return free_throws / possessions
### End Four Factors


@click.command()
@click.argument('input_fp', default='data/intermediate/base_team_stats.csv')
@click.argument('output_fp', default='data/intermediate/advanced_team_stats.csv')
def run(input_fp, output_fp):
    """ Compute advanced statistics for use in game examples

    Depends on correctly formatted CSV base stats file """
    if os.path.exists(output_fp): sys.exit('Advanced statistics already calculated!')
    if not os.path.exists(input_fp):
        sys.exit('{} does not exist.  Base statistics need to be generated.'.format(input_fp))

    stats = pd.read_csv(input_fp)

    output = None
    for i, season in enumerate(tqdm(stats['season'].unique())):
        teams = stats[stats['season'] == season]['team_name'].unique()
        for k, team_name in enumerate(tqdm(teams)):
            team_season_stats = stats[(stats['season'] == season) & (stats['team_name'] == team_name)]

            possessions = calculate_possessions(team_season_stats)

            team_output = {}
            team_output['season'] = season
            team_output['team_name'] = team_name
            team_output['kaggle_id'] = team_season_stats['kaggle_id']
            team_output['efg_pct'] = get_effective_field_goal_pct(fgm = team_season_stats['fgm'],
                                                                  fgm3 = team_season_stats['fgm3'],
                                                                  fga = team_season_stats['fga'])
            team_output['opp_efg_pct'] = get_effective_field_goal_pct(fgm = team_season_stats['fgm'],
                                                                  fgm3 = team_season_stats['fgm3'],
                                                                  fga = team_season_stats['fga'])
            team_output['topp'] = get_turnovers_per_possession(team_season_stats['to'], possessions)
            team_output['opp_topp'] = get_turnovers_per_possession(team_season_stats['opp_to'], possessions)

            team_output['ftr'] = get_free_throw_rate(free_throws = team_season_stats['ftm'], possessions = possessions)
            team_output['opp_ftr'] = get_free_throw_rate(free_throws = team_season_stats['opp_ftm'], possessions = possessions)

            team_output['or_pct'] = get_offensive_rebounding_pct(orb=team_season_stats['or'], opp_drb=team_season_stats['opp_dr'])
            team_output['opp_or_pct'] = get_offensive_rebounding_pct(orb=team_season_stats['opp_or'], opp_drb=team_season_stats['dr'])

            team_output['dr_pct'] = get_defensive_rebounding_pct(drb=team_season_stats['dr'], opp_orb=team_season_stats['opp_or'])
            team_output['opp_dr_pct'] = get_defensive_rebounding_pct(drb=team_season_stats['opp_dr'], opp_orb=team_season_stats['or'])

            team_output['rpi'] = team_season_stats['rpi']

            if type(output) == type(None):
                output = pd.DataFrame(team_output, index=[i+k])
            else:
                output = output.append(pd.DataFrame(team_output, index=[i+k]))

    output.to_csv(output_fp, index=None)



if __name__ == "__main__":
    run()
