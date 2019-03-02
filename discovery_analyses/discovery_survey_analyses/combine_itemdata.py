#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 12:53:24 2016

@author: poldrack
"""

import os,glob
import pandas
import json

basedir = os.path.join(get_info('base_directory'),'discovery_survey_analyses')

files=glob.glob(os.path.join(basedir,'surveydata/*'))
files.sort()

try:
    del data
except:
    pass

all_metadata={}
for f in files:
    
    d=pandas.read_csv(f,sep='\t')
    code=f.split('/')[-1].replace('.tsv','')
    try:
        data=data.merge(d,on='worker')
    except:
        data=d
    with open(f.replace('tsv','json').replace('surveydata','metadata'), encoding='utf-8') as data_file:
        md = json.loads(data_file.read())
    all_metadata[code]=md

data.to_csv('surveydata.csv',index=False)
with open('all_survey_metadata.json', 'w') as outfile:
            json.dump(all_metadata, outfile, sort_keys = True, indent = 4,
                  ensure_ascii=False)