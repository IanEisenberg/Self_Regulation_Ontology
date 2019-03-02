#!/usr/bin/env python3

"""
prepare metadata for tasks
"""



import os,pickle,sys
import json

from selfregulation.utils.utils import get_info
import selfregulation.prediction.behavpredict as behavpredict
basedir=get_info('base_directory')
dataset=get_info('dataset')
outdir=os.path.join(basedir,'Data/Derived_Data/%s'%dataset)

bp=behavpredict.BehavPredict(verbose=True,
         drop_na_thresh=100,n_jobs=1,
         skip_vars=['RetirementPercentStocks',
         'HowOftenFailedActivitiesDrinking',
         'HowOftenGuiltRemorseDrinking',
         'AlcoholHowOften6Drinks'])
bp.load_behav_data('task')
taskvars=list(bp.behavdata.columns)
taskvars.sort()
with open(os.path.join(outdir,'task_metadata.tsv'),'w') as f:
    for t in taskvars:
        t=t.replace('.0','')
        t_s=t.split('.')
        if t_s[1]=='hddm_drift':
            t_s[1]='Drift rate (HDDM)'
        elif t_s[1]=='hddm_thresh':
            t_s[1]='Threshold (HDDM)'
        elif t_s[1]=='hddm_non_decision':
            t_s[1]='Nondecision time (HDDM)'
        else:
            t_s[1]=t_s[1].replace('_',' ')
        if t_s[1].find('hddm drift')>0:
            t_s[1]=t_s[1].replace('hddm drift','')
            t_s[1]=t_s[1].replace('  ',' ')
            t_s[1]=t_s[1]+' (HDDM drift)'
        if len(t_s)>2:
            t_s[1]=t_s[1]+' (%s)'%t_s[2]
        print(t,t_s[1])
        f.write('%s\t%s\t%s\n'%(t,t_s[0],t_s[1]))
