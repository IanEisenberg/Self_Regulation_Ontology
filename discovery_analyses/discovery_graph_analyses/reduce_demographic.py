# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 15:40:41 2016

@author: ian
"""
import numpy as np
from os import path
import pandas as pd
import seaborn as sns
from sklearn import decomposition
import sys

sys.path.append('../utils')
from utils import get_info,get_survey_data
from plot_utils import dendroheatmap

dataset='Discovery_9-26-16'
basedir=get_info('base_directory')
derived_dir=path.join(basedir,'Data/Derived_Data/%s'%dataset)

def get_demog_data():
    demogdata=pd.read_csv(path.join(derived_dir,'surveydata/demographics.tsv'),index_col=0,delimiter='\t')
    healthdata=pd.read_csv(path.join(derived_dir,'surveydata/health_ordinal.tsv'),index_col=0,delimiter='\t')
    alcdrugs=pd.read_csv(path.join(derived_dir,'surveydata/alcohol_drugs_ordinal.tsv'),index_col=0,delimiter='\t')
    assert all(demogdata.index==healthdata.index)
    assert all(demogdata.index==alcdrugs.index)
    demogdata=demogdata.merge(healthdata,left_index=True,right_index=True)
    demogdata=demogdata.merge(alcdrugs,left_index=True,right_index=True)
    # remove a couple of outliers - this is only for cases when we include BMI/obesity
    demogdata=demogdata.query('WeightPounds>50')
    demogdata=demogdata.query('HeightInches>36')
    demogdata=demogdata.query('CaffienatedSodaCansPerDay>-1')
    demogdata=demogdata.assign(BMI=demogdata['WeightPounds']*0.45 / (demogdata['HeightInches']*0.025)**2)
    demogdata=demogdata.assign(Obese=(demogdata['BMI']>30).astype('int'))
    return demogdata
    
demog_data = get_demog_data()
# remove subject who had issues with the demographic survey (missing required answers for some reasons)
failed_index = demog_data.index[demog_data.loc[:,'TrafficAccidentsLifeCount'].isnull()]
demog_data.drop(failed_index, inplace = True)
# remove incomplete demographic variables
demog_data.dropna(axis = 1, inplace = True)
#remove non-numeric
demog_data = demog_data._get_numeric_data()
# change variables to factors if they have fewer than 10 unique variables
for c in demog_data.columns[demog_data.apply(lambda x: len(np.unique(x)))<10]:
    if len(np.unique(demog_data.loc[:,c])) > 2:
        demog_data.loc[:,c] = demog_data.loc[:,c].astype('category', ordered=True)
    else:
        demog_data.loc[:,c] = demog_data.loc[:,c].astype('category', ordered=False)


# use polycor to calculate correlation amongst demographic data
import readline
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

utils = importr('utils')
ts=robjects.r('ts')
polycor = importr('polycor')
psych = importr('psych')

pandas2ri.activate()
polycor_out = polycor.hetcor(pandas2ri.py2ri(demog_data))


# dimensionality reduction
pca_data = np.matrix(polycor_out[0])
pca = decomposition.PCA()
pca.fit(pca_data)

# plot explained variance vs. components
sns.plt.plot(pca.explained_variance_ratio_.cumsum())

pca.n_components = 2
reduced_df = pca.fit_transform(pca_data)
sns.plt.scatter(reduced_df[:,0], reduced_df[:,1])