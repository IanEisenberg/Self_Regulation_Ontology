# Self_Regulation_Ontology
![CircleCI](https://circleci.com/gh/poldrack/Self_Regulation_Ontology.svg?style=svg&circle-token=c2c503d9ef106e45769fa00ca689b3b10d882c9d)

This is the main reposistory for analyses and data for the UH2 Self Regulation Ontology project.

## Setting up the repository

In order to use the code, you first need to create a version of the settings file, using the following steps:

1. Copy the file "Self_Regulation_Settings_example.txt" to a new file called "Self_Regulation_Settings.txt"

2. Using your favorite text editor, edit the file to specify the location of the project directory on the line 
starting with "base directory".  For example, on my computer it looks like:
base_directory:/Users/poldrack/code/Self_Regulation_Ontology/

Note: If you do not create a settings file, one will be created by the setup.py file (see below) with default values

## Organization of the repository

Data: contains all of the original and derived data

data_preparation: code for preparing derived data

utils: utilities for loading/saving data and metadata

other directories are specific to particular analyses - for any analysis you wish to add, please give it a descriptive name along with your initials - e.g. "irt_analyses_RP"


## Setting up python environment

### for all analyses besides data_preparation
pip install -r requirements1.txt
pip install -r requirements2.txt
python setup.py install
rpy2 needs to be installed

rpy2 can be install using conda install rpy2
if errors occur when install R packages in the conda environment these commands may fix the issues:
conda install gxx_linux-64
conda install gfortran_linux-64

### data preparation requires install expfactory-analysis, as below:
pip install git+https://github.com/IanEisenberg/expfactory-analysis

### R setup
install:

GPArotation
missForest
psych
lme4
qgraph
mpath
dynamicTreeCut

### Example sequence of installation steps using anaconda
* conda create -n SRO python=3.5.3
* source activate SRO
* pip install -r requirements1.txt
* pip install -r requirements2.txt
* conda install -c r rpy2
* conda install -c r r
* pip install git+https://github.com/IanEisenberg/expfactory-analysis

## Docker usage

to build run:
`docker build --rm -t sro .`

Mount the Data and Results directory from the host into the container at /SRO/Data and /SRO/Results respectively

To start bash in the docker container with the appropriate mounts run:
`docker run --entrypoint /bin/bash -v /home/ian/Experiments/expfactory/Self_Regulation_Ontology/Data:/SRO/Data -it sro`
