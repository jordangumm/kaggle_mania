#!/bin/sh

USERNAME=$1
PASSWORD=$2

unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    *)          machine="UNKNOWN:${unameOut}"
esac

# Install independent Miniconda env if not done already
if [ ! -d "./miniconda" ]; then
    if [ $machine = "Linux" ]; then
        echo ${machine}
        wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O ./miniconda.sh
    elif [ $machine = "Mac" ]; then
        echo ${machine}
        wget https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh -O ./miniconda.sh
    else
        echo ${machine}
	exit 1
    fi
    bash miniconda.sh -b -p ./miniconda
    rm ./miniconda.sh
fi

source ./miniconda/bin/activate

conda config --add channels conda-forge
conda install -y beautifulsoup4 click pandas tqdm numpy scikit-learn lime h5py theano lasagne deap

pip install kaggle-cli

if [ ! -d "kaggle_mania/march_mania/data/raw" ]; then
    mkdir -p kaggle_mania/march_mania/data/raw && cd kaggle_mania/march_mania/data/raw
    kg download -u "$USERNAME" -p "$PASSWORD" -c 'march-machine-learning-mania-2017'
fi

python kaggle_mania/march_mania/generate_intermediate_data.py

exit 0
