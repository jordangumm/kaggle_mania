#!/usr/bin/env python

# conda execute
# env:
#  - python >=2
#  - beautifulsoup4
#  - click
#  - pandas
#  - tqdm
# run_with: python2

from bs4 import BeautifulSoup
import click
import sys, os
import pandas as pd
import urllib

from team_list import teams
from tqdm import tqdm


def get_html(season):
    """ Query and save html for each team into data/<season>

    """
    if not os.path.exists('data/html/{}'.format(season)):
        os.makedirs('data/html/{}'.format(season))

    for team in tqdm(teams):
	if os.path.exists('data/html/{}/{}.html'.format(season, team['kaggle_id'])):
	    continue
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
            print("{} doesn't exist for {} season".format(team['team_name'], season))
            print(e)


if __name__ == "__main__":
    seasons = xrange(2010,2018)
    for season in tqdm(seasons):
        get_html(season)
