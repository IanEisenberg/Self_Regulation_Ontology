#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 20:13:32 2017

@author: ian
"""
import glob
import pandas as pd
from selfregulation.utils.utils import get_behav_data




df = get_behav_data(file='demographic_health.csv', full_dataset=True)

failed_dataset = sorted(glob.glob('../Data/Failed*'))[0]
failed_subjects = get_behav_data(dataset=failed_dataset, file='demographic_health.csv')


fmri_dataset = sorted(glob.glob('../Data/Fmri*'))[0]
fmri_subjects = get_behav_data(dataset=fmri_dataset, file='demographic_health.csv')
fmri_subjects.index = ['fmri_'+i for i in fmri_subjects.index]

all_subjects = pd.concat([df,failed_subjects,fmri_subjects])

# total of 662 workers in mturk sample
worker_counts = pd.read_json('../Data/Local/worker_counts.json', typ='series')
total_workers = len(worker_counts)+len(fmri_subjects)

# separate into groups
groups = all_subjects.groupby(['HispanicLatino','Sex','Race']).Age.count()
