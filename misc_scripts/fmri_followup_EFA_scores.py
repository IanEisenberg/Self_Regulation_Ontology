#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 13:28:14 2018

@author: ian
"""

from os import path
from dimensional_structure.utils import transfer_scores
from selfregulation.utils.utils import get_behav_data, get_info
from selfregulation.utils.result_utils import load_results

# get contextualizing results
results_dataset = 'Complete_03-29-2018'
results = load_results(datafile=results_dataset)

# get fmri data
fmri_dataset= 'Fmri_Followup_10-22-2018'
data = get_behav_data(dataset=fmri_dataset)
# remove data where participants are missing more than 20% of the tasks
tasks = data.copy()
tasks.columns = [i.split('.')[0] for i in tasks.columns]
successful_subjects = (tasks.isnull().reset_index().melt(id_vars=['index']) \
                       .groupby(['index', 'variable']).mean() \
                       .groupby('index').sum()<12)
successful_subjects = successful_subjects[successful_subjects['value']]
data = data.loc[successful_subjects.index]

task_scores = transfer_scores(data, results['task'])
survey_scores = transfer_scores(data, results['survey'])

# save the scores
basename = 'factorscores_results-%s.csv' % results_dataset
task_scores.to_csv(path.join(get_info('base_directory'),
                             'Data',
                             fmri_dataset,
                             'task_'+basename))
survey_scores.to_csv(path.join(get_info('base_directory'),
                             'Data',
                             fmri_dataset,
                             'survey_'+basename))