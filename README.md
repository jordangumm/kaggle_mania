# March Madness Prediction
AutoML experiments using Maxout Networks and Advanced Analytics for predicting March Madness games.

The main purpose of this repository is to provide a platform for experimenting with automatated machine learning methods.  So far the focus has been on automated training (feature and hyperparameter selection) of different Bayesian (Monte Carlo Dropout) Maxout Network implementations (general, residual and dense connection architectures) using evolutionary procedures and cross-validated bagging.

## Setup
1. Sign up for Kaggle account and take note of username and password for use in build script (downloading initial box score stats).
2. Install [anaconda](https://www.continuum.io/downloads) for building reproducible environment.
4. Build the project: `$./build.bash <kaggle_username> <kaggle_password>`
5. Train bagged model: `$conda execute -v bagging_procedure.py`
6. Full model selection pipeline: `$conda execute -v model_selection.py`
