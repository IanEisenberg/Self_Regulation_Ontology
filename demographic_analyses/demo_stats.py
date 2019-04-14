#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 18:41:59 2019

@author: ian
"""
import numpy as np
from selfregulation.utils.utils import get_demographics, get_recent_dataset

demo=get_demographics(get_recent_dataset(), drop_categorical=False)
race_info = np.unique(demo.Race, return_counts=True)
race_info = {k.lstrip():v for k,v in zip(race_info[0], race_info[1])}
race_percentiles = {k:np.round(v/demo.shape[0]*100,2) for k,v in race_info.items()}
age_stats = demo.Age.describe()

print('** Race Statistics **')
for x,y in race_percentiles.items():
    print (x, ':',  y)
print('Hispanic %', demo.HispanicLatino.mean().round(3))
print('** Age and Sex **')
print(age_stats)
print('Female %', demo.Sex.mean().round(3))
