#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create variable key for use in R

Created on Fri Sep  2 15:25:16 2016

@author: poldrack
"""

import json

with open('all_survey_metadata.json', encoding='utf-8') as data_file:
        md = json.loads(data_file.read())

all_metadata={}
for measure in md.keys():
    for k in md[measure].keys():
        all_metadata[k]=md[measure][k]

with open('variable_key.txt','w') as f:
    for k in all_metadata.keys():
        f.write('%s\t%s\n'%(k,all_metadata[k]['Description']))
    