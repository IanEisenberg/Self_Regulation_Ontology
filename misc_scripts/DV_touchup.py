#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:56:41 2018

@author: ian
"""
import numpy as np
import pandas as pd

# fix threebytwo
behav_path = '/mnt/temp/mturk_retest_output/threebytwo_mturk_retest_DV.json'
dvs = pd.read_json(behav_path)
param = 'drift'
dvs['task_switch_cost_hddm_' + param] = dvs['hddm_' + param + '_task_switch'] - dvs['hddm_' + param + '_task_stay']
dvs.to_json(behav_path)

# update motor selective
task = 'motor_selective_stop_signal'
behav_path = '/mnt/temp/mturk_retest_output/%s_mturk_retest_DV.json' % task

dvs = pd.read_json(behav_path)
dvs.loc[:, 'proactive_control_rt'] = dvs.selective_proactive_control
dvs.loc[:,'proactive_control_hddm_drift'] = dvs.condition_sensitivity_hddm_drift
dvs.loc[:, 'reactive_control_rt'] = dvs.reactive_control
dvs.drop(['selective_proactive_control','condition_sensitivity_hddm_drift','reactive_control'], axis=1, inplace=True)
dvs.to_json(behav_path)

# update stim selective
task = 'stim_selective_stop_signal'
behav_path = '/mnt/temp/mturk_complete_output/%s_mturk_complete_DV.json' % task

dvs = pd.read_json(behav_path)
dvs.loc[:,'reactive_control_hddm_drift'] = dvs.condition_sensitivity_hddm_drift
dvs.drop(['condition_sensitivity_hddm_drift'], axis=1, inplace=True)
dvs.to_json(behav_path)

# update stop signal
task = 'stop_signal'
behav_path = '/mnt/temp/mturk_retest_output/%s_mturk_retest_DV.json' % task

dvs = pd.read_json(behav_path)
dvs.loc[:,'proactive_slowing_hddm_drift'] = dvs.condition_sensitivity_hddm_drift
dvs.loc[:,'proactive_slowing_hddm_thresh'] = dvs.condition_sensitivity_hddm_thresh

dvs.loc[:,'proactive_slowing_rt'] = dvs.proactive_slowing

dvs.drop(['condition_sensitivity_hddm_drift', 'condition_sensitivity_hddm_thresh', 'proactive_slowing'], axis=1, inplace=True)
dvs.to_json(behav_path)

# larger update
task = 'local_global_letter'
subset = 'retest'
behav_path = '/mnt/OAK/mturk_%s_output/%s_mturk_%s_DV.json' % (subset, task, subset)
fix_path = '/mnt/OAK/mturk_%s_fix/%s_mturk_%s_DV.json' % (subset, task, subset)
dvs = pd.read_json(behav_path)
fix_dvs = pd.read_json(fix_path)
missing = set(fix_dvs)-set(dvs)
dvs = pd.concat([dvs, fix_dvs.loc[:, missing]], axis=1)
dvs.sort_index(1, inplace=True)
dvs.to_json(behav_path)

# update
for subset in ['mturk_complete', 'mturk_retest']:
    fullDV_loc = '/mnt/OAK/%s_DV.json' % subset
    fullvalence_loc = '/mnt/OAK/%s_DV_valence.json' % subset
    dv = pd.read_json(fullDV_loc)
    valence = pd.read_json(fullvalence_loc)
    
    updateDV = '/mnt/OAK/%s_fix/%s_DV.json' % (subset, subset)
    updatevalence = '/mnt/OAK/%s_fix/%s_DV_valence.json' % (subset, subset)
    dv_fix = pd.read_json(updateDV)
    valence_fix = pd.read_json(updatevalence)
    
    # add non-overlapping columns
    overlapping_columns = list(set(dv_fix.columns) & set(dv.columns))
    assert np.mean(np.mean(dv_fix[overlapping_columns] - dv[overlapping_columns])) == 0
    
    nonoverlapping_columns = list(set(dv_fix.columns) - set(dv.columns))
    dv[nonoverlapping_columns] = dv_fix[nonoverlapping_columns]
    valence[nonoverlapping_columns] = valence_fix[nonoverlapping_columns]
    
    dv.to_json(fullDV_loc)
    valence.to_json(fullvalence_loc)
