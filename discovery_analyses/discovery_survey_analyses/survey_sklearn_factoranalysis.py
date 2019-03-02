"""
based loosely on PyIBP/example/example.py
using full LA2K dataset
"""

import os,glob,sys
import numpy,pandas
import json

from sklearn.preprocessing import scale
from sklearn.decomposition import FactorAnalysis
from sklearn import cross_validation

# this is kludgey but it works
sys.path.append('../utils')
from utils import get_info,get_survey_data

dataset='Discovery_9-26-16'
basedir=get_info('base_directory')
derived_dir=os.path.join(basedir,'Data/Derived_Data/%s'%dataset)


data,surveykey=get_survey_data(dataset)

cdata=data.values


kf = cross_validation.KFold(cdata.shape[0], n_folds=4)

max_components=30

sc=numpy.zeros((max_components,4))

for n_components in range(1,max_components):
    fa=FactorAnalysis(n_components=n_components)
    fold=0
    for train,test in kf:
        train_data=cdata[train,:]
        test_data=cdata[test,:]

        fa.fit(train_data)
        sc[n_components,fold]=fa.score(test_data)
        fold+=1

meanscore=numpy.mean(sc,1)
meanscore[0]=-numpy.inf
maxscore=numpy.argmax(meanscore)
print ('crossvalidation suggests %d components'%maxscore)

# now run it on full dataset to get components
fa=FactorAnalysis(n_components=maxscore)
fa.fit(cdata)

for c in range(maxscore):
    s=numpy.argsort(fa.components_[c,:])
    print('')
    print('component %d'%c)
    for i in range(3):
        print('%f: %s %s'%(fa.components_[c,s[i]],data.columns[s[i]],surveykey[data.columns[s[i]]]))
    for i in range(len(s)-4,len(s)-1):
        print('%f: %s %s'%(fa.components_[c,s[i]],data.columns[s[i]],surveykey[data.columns[s[i]]]))
