"""
assess the effect of dropping a particular task on the similarity across subjects
"""


import os,glob,sys,itertools
import numpy,pandas
import json
from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()
print('using %d cores'%num_cores)

from sklearn.preprocessing import scale
from sklearn.decomposition import FactorAnalysis
from sklearn import cross_validation

# this is kludgey but it works
sys.path.append('../utils')
from utils import get_info,get_behav_data

dataset=get_info('dataset')
print('using dataset:',dataset)
basedir=get_info('base_directory')
derived_dir=os.path.join(basedir,'Data/Derived_Data/%s'%dataset)


data=pandas.read_csv(os.path.join(derived_dir,'taskdata_clean_cutoff3.00IQR_imputed.csv'))

cdata=scale(data.values)
nsubs=data.shape[0]
subcorr=numpy.corrcoef(cdata)[numpy.triu_indices(nsubs,1)]

# get task names and indicator
tasknames=[i.split('.')[0] for i in data.columns]
tasks=list(set(tasknames))
tasks.sort()

ntasks=8 # number of tasks to select - later include time

tasknums=[i for i in range(len(tasks))]

allcombs=[i for i in itertools.combinations(range(32),8)]

cc=numpy.zeros(len(allcombs))
chosen_tasks={}

def get_subset_corr(x,ct,data):
    tasknames=[i.split('.')[0] for i in data.columns]
    tasks=list(set(tasknames))
    tasks.sort()
    chosen_vars=[]
    for i in ct:
        vars=[j for j in range(len(tasknames)) if tasknames[j].split('.')[0]==tasks[i]]
        chosen_vars+=vars

    chosen_data=data.ix[:,chosen_vars].values
    chosen_data=scale(chosen_data)
    subcorr_subset=numpy.corrcoef(chosen_data)[numpy.triu_indices(nsubs,1)]
    return(numpy.corrcoef(subcorr,subcorr_subset)[0,1])

use_parallel=True
if use_parallel:
    cc = Parallel(n_jobs=num_cores)(delayed(get_subset_corr)(x,ct,data) for x,ct in enumerate(allcombs))

else:
    for x,ct in enumerate(allcombs):
        cc[x]=get_subset_corr(x,ct,cdata)
        if x>4:
            break

numpy.save('cc.npy',cc)
