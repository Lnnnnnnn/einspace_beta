#!/bin/bash

# create our conda environment
conda create -n einsearch python=3.10 -y
source activate base
conda activate einsearch

# install required packages
conda install --channel conda-forge pygraphviz -y
pip install -r requirements.txt
