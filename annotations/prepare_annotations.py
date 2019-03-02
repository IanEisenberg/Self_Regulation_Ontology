"""
clean up patrick's annotations
"""

import sys,os,json
from collections import defaultdict
import numpy,pandas

# this is kludgey but it works
sys.path.append('../utils')
from utils import get_info,get_behav_data

dataset=get_info('dataset')
print('using dataset:',dataset)
basedir=get_info('base_directory')
derived_dir=os.path.join(basedir,'Data/Derived_Data/%s'%dataset)
annot_dir=os.path.join(basedir,'annotations')

# load tsv
f=open(os.path.join(annot_dir,'UH2TaskMeasureLabeling.txt'))

header=f.readline().strip().split('\t')
print(header)
annotations=defaultdict(lambda:{})
for l in f.readlines():
    l_s=l.strip().split('\t')
    varname=l_s[0].split('""')[1]
    annot=[i.lstrip().replace(' ','_') for i in l_s[1].replace('"','').lower().split(',')]
    annotations[varname]['process']=annot
    if len(l_s)>2:
        drift=l_s[2].replace('"','').lower()
    else:
        drift=''
    annotations[varname]['drift']=drift
    diff=False
    if len(l_s)>3:
        d=l_s[3].replace('"','').lower()
        if d=='yes':
            diff=True
    annotations[varname]['diff']=diff
    oneoff=False
    if len(l_s)>4:
        s=l_s[4].replace('"','').lower()
        if s=='x':
            oneoff=True
    annotations[varname]['oneoff']=oneoff

with open(os.path.join(annot_dir,'task_annotations.json'), 'w') as outfile:
        json.dump(annotations, outfile, sort_keys = True, indent = 4,
              ensure_ascii=False)
