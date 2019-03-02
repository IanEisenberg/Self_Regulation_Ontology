#!/usr/bin/env python3
import pandas as pd
from os import path

data_dir = '/oak/stanford/groups/russpold/users/ieisenbe/uh2/behavioral_data/mturk_retest_output/trial_num/ordered'
labels = ['retest_0.25', 'retest_0.5', 'retest_0.75', 'complete_0.25', 'complete_0.5', 'complete_0.75']

for label in labels:
    path.join(data_dir,label + '_DV.json')
    DVs = pd.read_json(path.join(data_dir,label + '_DV.json'))
    DVs.dropna(axis = 1, how = 'all', inplace = True)
    DVs.to_csv(path.join(data_dir, label + '_variables_exhaustive.csv'))
