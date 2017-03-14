import pandas as pd
import csv
import os
import re

from bs4 import BeautifulSoup

html_path = './data/html'

def cell_text(cell):
    return " ".join(cell.stripped_strings)

for season in xrange(2003,2018):
    print season
    for f in os.listdir(os.path.join(html_path, str(season))):
        html_fp = os.path.join(html_path, str(season), f)
        html = open(html_fp).read()
        if 'Page Not Found (404 error)' in html:
            continue
        #print html_fp
        soup = BeautifulSoup(html, 'lxml')

        for table in soup.findAll('table'):
            #print table['id']
            output_fp = './data/intermediate/{}/{}_{}'.format(season,table['id'],f.replace('html','csv'))
            if not os.path.exists(os.path.dirname(output_fp)):
                os.makedirs(os.path.dirname(output_fp))
            output = csv.writer(open(output_fp, 'w+'))
            for row in table.findAll('tr'):
                col = map(cell_text, row.find_all(re.compile('t[dh]')))
                output.writerow(col)
            output.writerow([])
