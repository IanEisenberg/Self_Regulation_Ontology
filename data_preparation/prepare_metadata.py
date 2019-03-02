#!/usr/bin/env python3
"""
create metadata in json format for all measures

notes from Gina:
*=fixed
1. For meaningful_variables_metadata, the "SubscaleVarNums" column contains "SumAll"
(for the sum of all items), "MeanAll" (for the mean of all items), or a list of
items used to compute the scale score.  For this last option, is it always true
that these items were simply summed or averaged?  In a previous email,
Ian told me that this was not true for the Three-Factor Eating Questionnaire.
For this last option, will you also please indicate whether a sum or mean was
computed?

*2. For meaningful_variables_metadata, the "SubscaleVarNums" column is formatted as a
time for some of the subscales (e.g., "4:07:11 AM" for eating_survey.emotional_eating).

3. For item_level_metadata, "ResponseOptions" for eating_survey.19 should be changed
from "1:2;2:4;3:6;4:8" to reflect that original scores of 1 or 2 were recoded to 1,
original scores of 3 or 4 were recoded to 2, etc.

*4. For item_level_metadata, odd characters sometimes appear in the item text
(e.g., â€œ for the Sensation-Seeking Survey).

IAN 5. For item_level_metadata, will you please clarify what "Misc" means in the
"Scoring" column?

6. For outcome_metadata, will you please clarify what "ForwardShifted"
means in the "Scoring" column?

7. Are we planning to create a separate document noting any errors that occurred
while administering the items (e.g., missing sensation_seeking_survey.10
due to response option error, only administered 30 of the 31 items on
the Short Self-Regulation Questionnaire)?  This could just be a Word
file in the same folder as these metadata files.


"""

from metadata_validator import validate_exp
from collections import OrderedDict
import pandas,unicodedata

import os,pickle,sys,math
import json
from selfregulation.utils.utils import get_info,get_behav_data,get_item_metadata,get_var_category

from measure_dictionaries import measure_longnames,measure_termurls,measure_sobcurls

basedir=get_info('base_directory')
dataset=get_info('dataset')
outdir=os.path.join(basedir,'Data/Derived_Data/%s'%dataset)

def get_subscale_vars():
    subscale_data=pandas.read_csv('../references/survey_subscale_reference.csv',
                                index_col=0)
    subscale_var_dict={}
    for v in subscale_data.index:
        if subscale_data.loc[v].iloc[2]=='sum items':
            subscale_var_dict[v]='SumAll'
        elif subscale_data.loc[v].iloc[2]=='mean items':
            subscale_var_dict[v]='MeanAll'
        else:
            d=[]
            for i in subscale_data.loc[v]:
                try:
                    d.append(str(int(i)))
                except:
                    pass
            subscale_var_dict[v]=':'.join(d)
    return subscale_var_dict

subscale_var_dict=get_subscale_vars()

# first get variable-level metadata
# based on variable set in meaningful_variables

behavdata=get_behav_data(dataset)
measures={}
for c in list(behavdata.columns):
    c_s=c.split('.')
    m=c_s[0]
    v='.'.join(c_s[1:])
    if not m in measures:
        measures[m]={'dataElements':[]}
    measures[m]['dataElements'].append(v)

metadata={}
# three entries are: class, type, and whether we are looking at beginning
# of string - this lets us find differences
task_vars=[('hddm_drift','DDMDriftRate','rate',True),
            ("hddm_non_decision",'DDMNondecisionTime','seconds',True),
            ('hddm_thresh','DDMThreshold','other',True),
            ("load",'load','count',False),
            ('hyp_discount_rate','hyperbolicDiscountRate','rate',True),
            ('SSRT','SSRT','milliseconds',True),
            ('span','span','count',False),
            ('percent','other','percentage',False),
            ('SSRT','differenceSSRT','seconds',False),
            ('hddm_drift','differenceDDMDriftRate','rate',False),
            ('slowing','differenceSlowing','seconds',False)]

def get_clean_title(t):
        t=t.replace('.0','')
        if t=='hddm_drift':
            t='Drift rate (HDDM)'
        elif t=='hddm_thresh':
            t='Threshold (HDDM)'
        elif t=='hddm_non_decision':
            t='Nondecision time (HDDM)'
        else:
            t=t.replace('_',' ')
        if t.find('hddm drift')>0:
            t=t.replace('hddm drift','')
            t=t.replace('  ',' ')
            t.strip()
            t=t+'(HDDM drift)'
        #if len(t_s)>2:
        #    t=t_s[1]+' (%s)'%t_s[2]
        return(t)

for m in measures.keys():
    metadata[m]={'measureType':get_var_category(m),
        'title':measure_longnames[m],
        'URL':{'CognitiveAtlasURL':measure_termurls[m],
                'SOBCURL':measure_sobcurls[m]},
        "expFactoryName":m,
        'dataElements':{}}
    for e in measures[m]['dataElements']:
        metadata[m]['dataElements'][e]={}
        if get_var_category(m)=='survey':
            metadata[m]['dataElements'][e]['variableClass']='surveySummary'
            metadata[m]['dataElements'][e]['variableUnits']='ordinal'
            metadata[m]['dataElements'][e]['title']=e.replace('_',' ')
            try:
                metadata[m]['dataElements'][e]['subscaleVarNums']=subscale_var_dict['%s.%s'%(m,e)]
            except KeyError:
                print('no key: %s.%s'%(m,e))
                metadata[m]['dataElements'][e]['subscaleVarNums']=''

        else:
            # get variable type for tasks
            metadata[m]['dataElements'][e]['title']=get_clean_title(e)
            for k in task_vars:
                if k[3] is True:
                    #print(m,e,k)
                    if e.find(k[0])==0:
                        #print("found!")
                        metadata[m]['dataElements'][e]['variableClass']=k[1]
                        metadata[m]['dataElements'][e]['variableUnits']=k[2]
                else:
                    #print(m,e,k)
                    if e.find(k[0])>0:
                        #print("found!")
                        metadata[m]['dataElements'][e]['variableClass']=k[1]
                        metadata[m]['dataElements'][e]['variableUnits']=k[2]
            # override switch cost hddm definition
            if e.find('switch_cost')>-1:
                metadata[m]['dataElements'][e]['variableClass']='differenceSwitchCost'
                metadata[m]['dataElements'][e]['variableUnits']='milliseconds'

            if not 'variableClass' in metadata[m]['dataElements'][e]:
                #print('not found, setting to other')
                metadata[m]['dataElements'][e]['variableClass']='other'
                metadata[m]['dataElements'][e]['variableUnits']='other'
    if get_var_category(m)=='survey':
        item_md=get_item_metadata(m)
        metadata[m]['dataElements']['surveyItems']=item_md


# get demog/health data
# we can use the metadata files that were generated by mturk_save_data.py

health_metadata=json.load(open(os.path.join(basedir,'Data',dataset,'metadata/health.json'),'r'))
health_metadata.pop('MeasurementToolMetadata')
alcdrug_metadata=json.load(open(os.path.join(basedir,'Data',dataset,'metadata/alcohol_drugs.json'),'r'))
alcdrug_metadata.pop('MeasurementToolMetadata')
demog_metadata=json.load(open(os.path.join(basedir,'Data',dataset,'metadata/demographics.json'),'r'))
demog_metadata.pop('MeasurementToolMetadata')

metadata['demographics_survey']={'measureType':'outcomes',
                            'title':'Demographics survey',
                            'URL':{},
                            'expFactoryName':'demographics_survey',
                            'dataElements':{'surveyItems':[]}}
metadata['k6_survey']={'measureType':'outcomes',
                            'title':'K6 mental health survey',
                            'URL':{},
                            'expFactoryName':'k6_survey',
                            'dataElements':{'surveyItems':[]}}
metadata['alcohol_drugs_survey']={'measureType':'outcomes',
                            'title':'Alcohol/drug use survey',
                            'URL':{},
                            'expFactoryName':'alcohol_drug_survey',
                            'dataElements':{'surveyItems':[]}}

def get_reformatted_metadata(orig_metadata):
    surveyItems=[]
    for i in orig_metadata:
        tmp={'expFactoryName':i,
            'text':orig_metadata[i]['Description'].lstrip(),
            'questionNum':None,
            'responseOptions':{}} #OrderedDict()}
        if 'LevelsOrig' in orig_metadata[i]:
            # make dictionary to translate from orig levels to new Levels
            levels_inverse_dict = {v: k for k, v in orig_metadata[i]['Levels'].items()}
            for o in orig_metadata[i]['LevelsOrig']:
                textOrig=orig_metadata[i]['LevelsOrig'][o]
                valueNew=levels_inverse_dict[textOrig].lstrip()
                tmp['responseOptions'][valueNew]={'text':orig_metadata[i]['LevelsOrig'][o].lstrip(),
                                                'valueOrig':o.lstrip()}
            tmp['variableUnits']='ordinal'
        elif 'Levels' in orig_metadata[i]:
            if len(orig_metadata[i]['Levels'])>0:
                for o in orig_metadata[i]['Levels']:
                    tmp['responseOptions'][o.lstrip()]={'text':orig_metadata[i]['Levels'][o].lstrip(),
                                                    'valueOrig':o.lstrip()}
                tmp['variableUnits']='ordinal'
        else:
                tmp['responseOptions']='n/a'
                tmp['variableUnits']='other'
        if 'Nominal' in orig_metadata[i]:
            tmp['variableUnits']='nominal'
        # get scoring type: Forward, Reverse, or ForwardShifted
        data=[]
        for o in tmp['responseOptions']:
            try:
                data.append([int(o),int(tmp['responseOptions'][o]['valueOrig'])])
            except:
                pass
        df=pandas.DataFrame(data)
        if len(df)>0:
            cc=df.corr().iloc[0,1]
            if all(df[0]==df[1]):
                tmp['scoring']='Forward'
            elif cc>0.5:
                tmp['scoring']='ForwardShifted'
            elif cc<0.5:
                tmp['scoring']='Reverse'
            else:
                tmp['scoring']='Other'
        surveyItems.append(tmp)
    return(surveyItems)

metadata['demographics_survey']['dataElements']['surveyItems']=get_reformatted_metadata(demog_metadata)
metadata['k6_survey']['dataElements']['surveyItems']=get_reformatted_metadata(health_metadata)
metadata['alcohol_drugs_survey']['dataElements']['surveyItems']=get_reformatted_metadata(alcdrug_metadata)

# need to reformat these to fit with the current structure



# doublecheck that everythign is there

json.dump(metadata, open('./metadata.json','w'), sort_keys = True, indent = 4,
                  ensure_ascii=False)
