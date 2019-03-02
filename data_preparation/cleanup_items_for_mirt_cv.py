#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
need to clean up individual items that have a small number of partiuclar
responses in order to allow crossvalidation with equated response options

Created on Fri Sep  2 15:04:04 2016

@author: poldrack
"""

import os
import pandas,numpy
import scipy.stats


import os,glob,sys
import pandas
import json
try:
    dataset=sys.argv[1]
    min_freq=int(sys.argv[2])
    print('using minimum frequency=',min_freq)
    usefull=bool(int(sys.argv[3]))
except:
    print('usage: python cleanup_items_for_mirt_cv.py <dataset> <min items> <use all data 1/0>')
    sys.exit()

from cleanup_item_dist import cleanup_item_dist,get_respdist


from selfregulation.utils.utils import get_info,get_behav_data
basedir=get_info('base_directory')
#dataset=get_info('dataset')
if usefull:
    print('using full dataset')
    derived_dir=os.path.join(basedir,'Data/Derived_Data/%s'%dataset.replace('Discovery','Combined').replace('Validation','Combined'))
else:
    print('using dataset:',dataset)
    derived_dir=os.path.join(basedir,'Data/Derived_Data/%s'%dataset)
datadir=os.path.join(basedir,'data/%s'%dataset)

if not os.path.exists(derived_dir):
    os.makedirs(derived_dir)
print('saving to',derived_dir)

data=get_behav_data(file='subject_x_items.csv',full_dataset=usefull)

maxnans=5

fixdata=data.copy()
dropped={}
fixed={}
for c in data.columns:

    f,dropflag=cleanup_item_dist(c,fixdata,verbose=False,minresp=min_freq)
    fixdata[c]=f
    u,h=get_respdist(f)
    if numpy.sum(numpy.isnan(data[c]),0)>maxnans:
        print('dropping %s due to too many NaNs'%c)
        dropflag=True
    if dropflag:
        del fixdata[c]
        dropped[c]=(u,h)
        print('dropping',c)
    else:
        fixed[c]=(u,h)
        diff_resps=numpy.sum(fixdata[c]!=data[c])
        if diff_resps>0:
            print(diff_resps,c)


fixdata.to_csv(os.path.join(derived_dir,'surveydata_fixed_minfreq%d.csv'%min_freq))
