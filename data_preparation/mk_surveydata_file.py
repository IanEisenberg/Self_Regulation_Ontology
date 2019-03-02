#!/usr/bin/env python3
"""
save all survey data to single file and generate variable key
for MIRT anlayses
"""

import os,glob,json,sys
import numpy,pandas

# this is kludgey but it works
sys.path.append('../utils')

from utils import get_info
dataset='Discovery_9-26-16'
basedir=get_info('base_directory')
derived_dir=os.path.join(basedir,'data/Derived_Data/%s'%dataset)


datafiles=glob.glob(os.path.join(derived_dir,'surveydata/*survey.tsv'))

exclude_files=['alcohol_drugs.tsv','alcohol_drugs_ordinal.tsv','demographics.tsv',
              'demographics_ordinal.tsv','health.tsv','health_ordinal.tsv']
alldata=pandas.DataFrame()
allmetadata={}

for f in datafiles:
    d=pandas.read_csv(f,delimiter='\t',index_col=0)
    if not f in exclude_files:
        alldata=pandas.concat([alldata,d],axis=1)
    mdfile=f.replace('.tsv','.json').replace('surveydata','metadata')
    with open(mdfile) as md:
        metadata = json.loads(md.read())
    for m in metadata.keys():
        if m=="MeasurementToolMetadata":
            continue
        allmetadata[m]=metadata[m]['Description']
    for c in d.columns:
        assert c in metadata

alldata.to_csv(os.path.join(derived_dir,'surveydata.csv'))
mdkeys=list(allmetadata.keys())
mdkeys.sort()
shortnames=[]

with open(os.path.join(derived_dir,'surveyitem_key.txt'),'w') as f:
    for k in mdkeys:
        sp=k.split('_')
        if len(sp)==3:
            svname=''.join(i for i in sp[0][:6] if not i.isdigit())
            shortname=svname+'%02d'%int(sp[-1])
            print(k,shortname)
        else:
            svname1=''.join(i for i in sp[0][:3] if not i.isdigit())
            svname2=''.join(i for i in sp[1][:3] if not i.isdigit())
            shortname=svname1+svname2+'%02d'%int(sp[-1])
            print(k,shortname)
            shortnames.append(shortname)
        itemtext=allmetadata[k].replace("'",'')
        f.write('%s\t%s\t%s\n'%(k,shortname,itemtext))
# make sure all are unique
assert len(shortnames)==len(set(shortnames))
