#!/usr/bin/env python3
"""
export metdata to csv for Mackinnon group
"""

import os,pickle,sys
import json

from selfregulation.utils.utils import get_info,get_behav_data
basedir=get_info('base_directory')
dataset=get_info('dataset')
print('using dataset:',dataset)
datadir=os.path.join(basedir,'Data/%s'%dataset)
outdir=os.path.join(basedir,'Data/Derived_Data/%s'%dataset)
metadata=pickle.load(open(os.path.join(outdir,'survey_metadata.pkl'),'rb'))
surveys=list(metadata.keys())
surveys.sort()
with open(os.path.join(outdir,'survey_metadata.tsv'),'w') as f:
    for s in surveys:
        items=list(metadata[s].keys())
        items.sort()
        items.remove('MeasurementToolMetadata')
        for i in items:
            print(metadata[s][i])
            levels=list(metadata[s][i]['Levels'].keys())
            levels.sort()
            options='\t'.join(['%s:%s'%(k,metadata[s][i]['Levels'][k]) for k in levels])
            f.write('%s\t%s\t%s\n'%(i,
                metadata[s][i]['Description'],
                options))
