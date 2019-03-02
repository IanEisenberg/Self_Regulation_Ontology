#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 18:47:50 2017

@author: ian
"""
import pickle
from expanalysis.experiments.processing import calc_exp_DVs
import pandas as pd
from selfregulation.utils.utils import get_behav_data

df = get_behav_data(file = 'Individual_Measures/two_stage_decision.csv.gz')

workers = list(df.worker_id.unique())

df = df.query('worker_id in %s' % workers)

DV_tests = []
for repeats in range(100):
    print(repeats)
    DVs, valence, description = calc_exp_DVs(df)
    DVs.columns = [i + '_run%s' % str(repeats) for i in DVs.columns]
    DV_tests.append(DVs)
DV_tests = pd.concat(DV_tests, axis=1)
DV_tests.to_pickle('two_stage_tests.pkl')

N = len(DV_tests.columns)
corr = DV_tests.corr()
DV_reliabilities = {}
for c in range(5):
    r = DV_tests.corr().iloc[5+c:N:5,c].mean()
    DV_reliabilities[DV_tests.columns[c]] = r
    
pickle.dump(DV_reliabilities, open('two_stage_reliabilities.pkl','wb'))
