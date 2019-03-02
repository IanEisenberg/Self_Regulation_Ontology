# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:56:35 2016

@author: ian
"""
from math import floor, ceil
import numpy
import pandas as pd
from selfregulation.utils.utils import get_behav_data



df = get_behav_data(dataset = 'Discovery_11-20-2016', file = 'Individual_Measures/stop_signal.csv.gz')

worker = 0
df = df.query('worker_id == "%s"' % df.worker_id.unique()[worker])

#remove practice
df = df.query('exp_stage not in ["practice","NoSS_practice"]').reset_index(drop = True)

dvs = {}
# Calculate SSRT for both conditions
for c in df.condition.unique():
    c_df = df[df.condition == c]
    
    #SSRT
    go_trials = c_df.query('SS_trial_type == "go"')
    stop_trials = c_df.query('SS_trial_type == "stop"')
    sorted_go = go_trials.query('rt != -1').rt.sort_values(ascending = True)
    prob_stop_failure = (1-stop_trials.stopped.mean())
    corrected = prob_stop_failure/numpy.mean(go_trials.rt!=-1)
    index = corrected*len(sorted_go)
    index = [floor(index), ceil(index)]
    dvs['SSRT_' + c] = {'value': sorted_go.iloc[index].mean() - stop_trials.SS_delay.mean(), 'valence': 'Neg'}