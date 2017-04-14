
from bs4 import BeautifulSoup
import click
import sys, os
import pandas as pd
import urllib

from team_list import teams


def get_html(season):
    """ Query and save html for each team into data/<season>

    """
    if not os.path.exists('data/html/{}'.format(season)):
        os.makedirs('data/html/{}'.format(season))
    else:
        return

    for team in teams:
        try:
            page = urllib.urlopen(
                'http://www.sports-reference.com/cbb/schools/{}/{}.html'.format(
                                                        team['sports-reference'],
                                                        season))
            soup = BeautifulSoup(page.read().decode('ascii', 'ignore'), 'lxml')
            soup.prettify()
            output =  open('data/html/{}/{}.html'.format(season, team['kaggle_id']), 'w+')
            output.write(soup.prettify().encode('utf-8').strip())
            output.close()
        except Exception as e:
            print "{} doesn't exist for {} season".format(team['team_name'], season)
            print e


if __name__ == "__main__":
    seasons = xrange(2003,2018)
    for season in seasons:
        print 'fetching {} team html'.format(season)
        get_html(season)
