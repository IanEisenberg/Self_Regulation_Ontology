#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 18:41:59 2019

@author: ian
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


# plots
sns.set_context('paper')
size=5
def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=size*3)
    ax.grid(False)

f, axes = plt.subplots(1,3,figsize=(size*4,size))
educ_ax, income_ax, age_ax = axes
plt.subplots_adjust(wspace=.3)

# plot education
labels, counts = np.unique(demo.HighestEducation, return_counts=True)
educ_labels = ["","Did Not Complete High School", "High School/GED", "Some College", "Bachelor's Degree", "Master's Degree", "Advanced Graduate work or Ph.D"]

educ_ax.barh(labels, counts, color='', edgecolor='k', linewidth=size/3)
educ_ax.set_yticklabels(educ_labels, size=size*3)
educ_ax.set_xlabel("# Participants", size=size*4)
educ_ax.set_title('Education', size=size*4)
style_ax(educ_ax)

# plot income 
(demo.HouseholdIncome/1000).hist(bins=20, ax=income_ax, 
                          histtype='step', orientation='horizontal',
                          color='k', linewidth=size/3)
income_ax.hlines(demo.HouseholdIncome.median()/1000, *income_ax.get_xlim(),
           color='r', linestyle='--', lw=size/3)
income_ax.set_xlabel("# Participants", size=size*4)
income_ax.set_ylabel("Dollars (1000's)", size=size*4)
income_ax.set_title('Income', size=size*4)
style_ax(income_ax)

# plot age
demo.Age.hist(bins=20, ax=age_ax, 
                          histtype='step', orientation='horizontal',
                          color='k', linewidth=size/3)
age_ax.hlines(demo.Age.median(), *age_ax.get_xlim(),
           color='r', linestyle='--', lw=size/3)
age_ax.set_xlabel("# Participants", size=size*4)
age_ax.set_ylabel("Years", size=size*4)
age_ax.set_title('Age', size=size*4)
style_ax(age_ax)

