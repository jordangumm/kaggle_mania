source ./miniconda/bin/activate

conda install conda-execute --channel=conda-forge

echo 'Web Scraping Seasons...'
conda execute -v web_scraper.py

echo 'Generating data from HTML...'
conda execute -v generate_html_data.py

echo 'Generating intermediate data...'
conda execute -v generate_intermediate_data.py

echo 'Generating features...'
conda execute -v generate_final_data.py tourney
