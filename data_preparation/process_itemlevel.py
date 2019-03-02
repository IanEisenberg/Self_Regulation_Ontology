#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process the item-level data to generate separate files
assumes that the data are in the main directory
"""

import numpy,pandas
import os,pickle,sys
import json

sys.path.append('../utils')

from utils import get_info
basedir=get_info('base_directory')
dataset=get_info('dataset')
print('using dataset:',dataset)
datadir=os.path.join(basedir,'data/%s'%dataset)
outdir=os.path.join(basedir,'data/Derived_Data/%s'%dataset)
if not os.path.exists(outdir):
    os.mkdir(outdir)

survey_longnames={'eating_survey':'Eating survey',
                  'selection_optimization_compensation_survey':'Selection-Optimization-Compensation survey',
                  'sensation_seeking_survey':'Sensation seeking survey',
                  'erq_survey':'Emotion regulation questionnaire',
                  'impulsive_venture_survey':"I7 impulsiveness and venturesomeness survey",
                  'upps_impulsivity_survey':'UPPS+P',
                  'five_facet_mindfulness_survey':"Five facet mindfulness survey",
                  'bis11_survey':"Barratt Impulsiveness Survey",
                  'bis_bas_survey':'BIS/BAS',
                  'dospert_eb_survey':"DOSPERT - expected benefits",
                  'future_time_perspective_survey':"Future time perspective survey",
                  'mpq_control_survey':"Multidimensional personality questionnaire",
                  'brief_self_control_survey':"Brief self-control survey",
                  'time_perspective_survey':"Time perspective survey",
                  'theories_of_willpower_survey':"Theories of willpower survey",
                  'mindful_attention_awareness_survey':"Mindful attention/awareness survey",
                  'self_regulation_survey':"Self-regulation survey",
                  'leisure_time_activity_survey':'Leisure time activity survey',
                  'dospert_rp_survey':"DOSPERT - risk perception",
                  'grit_scale_survey':"Grit scale",
                  'ten_item_personality_survey':"Ten item personality survey",
                  'dospert_rt_survey':"DOSPERT - risk taking",
                  'dickman_survey':"Dickman impulsivity survey"}

survey_termurls={'erq_survey':"http://www.cognitiveatlas.org/term/id/trm_56bbead1a7ed4",
                'impulsive_venture_survey':"http://www.cognitiveatlas.org/term/id/trm_56a91e3e982f9",
                'upps_impulsivity_survey':"http://www.cognitiveatlas.org/term/id/trm_56a91a92043bc",
                'five_facet_mindfulness_survey':"http://www.cognitiveatlas.org/term/id/trm_56ab12e0f1a61",
                'bis11_survey':"http://www.cognitiveatlas.org/term/id/trm_55a6a8e81b7f4",
                'bis_bas_survey':"http://www.cognitiveatlas.org/term/id/trm_56a9137d9dce1",
                'leisure_time_activity_survey':"http://www.cognitiveatlas.org/term/id/trm_56bbe12994926",
                'future_time_perspective_survey':"http://www.cognitiveatlas.org/term/id/trm_56a915fe77945",
                'brief_self_control_survey':"http://www.cognitiveatlas.org/term/id/trm_56a915461cd91",
                'eating_survey':"http://www.cognitiveatlas.org/term/id/trm_56aac5f6e4702",
                'theories_of_willpower_survey':"http://www.cognitiveatlas.org/term/id/trm_56a91a3082c31",
                'mindful_attention_awareness_survey':"http://www.cognitiveatlas.org/term/id/trm_56abcba3df89b",
                'dospert_eb_survey':"http://www.cognitiveatlas.org/term/id/trm_5696abecf2569",
                'self_regulation_survey':"http://www.cognitiveatlas.org/term/id/trm_56a91ed5f1ccc",
                'time_perspective_survey':"http://www.cognitiveatlas.org/term/id/trm_56a91e92eab46",
                'dospert_rp_survey':"http://www.cognitiveatlas.org/term/id/trm_5696abecf2569",
                'grit_scale_survey':"http://www.cognitiveatlas.org/term/id/trm_56a9166421494",
                'ten_item_personality_survey':"http://www.cognitiveatlas.org/term/id/trm_56a919a478935",
                'dospert_rt_survey':"http://www.cognitiveatlas.org/term/id/trm_5696abecf2569",
                'dickman_survey':"http://www.cognitiveatlas.org/term/id/trm_55a6a95f66508",
                'mpq_control_survey':"http://www.cognitiveatlas.org/term/id/trm_55a6aa62c54f8",
                'sensation_seeking_survey':'',
                'selection_optimization_compensation_survey':''}


def get_data(datadir=datadir):

    datafile=os.path.join(datadir,'items.csv')

    data=pandas.read_csv(datafile,index_col=0)
    return data,basedir


def get_survey_items(data):
    survey_items={}
    for i,r in data.iterrows():
        if not r.survey in survey_items.keys():
            survey_items[r.survey]={}
        if not r.item_text in survey_items[r.survey].keys():
            survey_items[r.survey][r.item_text]=r
    return survey_items



def save_metadata(survey_items,
                  outdir=os.path.join(outdir, 'metadata')):

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    survey_metadata={}
    for k in survey_items.keys():

        survey_dict={"MeasurementToolMetadata": {"Description": survey_longnames[k],
            "TermURL": survey_termurls[k]}}
        for i in survey_items[k]:
            r=survey_items[k][i]
            itemoptions=eval(r.options)
            itemid='_'.join(itemoptions[0]['id'].split('_')[:-1])
            assert itemid not in survey_dict  # check for duplicates
            survey_dict[itemid]={}
            survey_dict[itemid]['Description']=r.item_text
            survey_dict[itemid]['Levels']={}
            for ii in itemoptions:
                survey_dict[itemid]['Levels'][ii['value']]=ii['text']
        with open(os.path.join(outdir,'%s.json'%k), 'w') as outfile:
            json.dump(survey_dict, outfile, sort_keys = True, indent = 4,
                  ensure_ascii=False)
        survey_metadata[k]=survey_dict
    return survey_metadata,outdir

def add_survey_item_labels(data):
    item_ids=[]
    for i,r in data.iterrows():
        itemoptions=eval(r.options)
        item_ids.append('_'.join(itemoptions[0]['id'].split('_')[:-1]))
    data['item_id']=item_ids
    return data

def save_data(data,survey_metadata,
              outdir=os.path.join(outdir,'surveydata')):

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for k in survey_metadata.keys():
        matchdata=data.query("survey=='%s'"%k)
        unique_items=list(matchdata.item_id.unique())
        surveydata=pandas.DataFrame({'worker':list(matchdata.worker.unique())})
        for i in unique_items:
            matchitem=matchdata.query('item_id=="%s"'%i)
            matchitem=pandas.DataFrame({'worker':matchitem.worker,i:matchitem.coded_response})
            surveydata=surveydata.merge(matchitem,on='worker')
        surveydata.to_csv(os.path.join(outdir,'%s.tsv'%k),sep='\t',index=False)
    return outdir

if __name__=='__main__':
    data,basedir=get_data()
    survey_items=get_survey_items(data)
    survey_metadata,metadatdir=save_metadata(survey_items)
    data=add_survey_item_labels(data)
    datadir=save_data(data,survey_metadata)
    pickle.dump((data,survey_metadata),open(os.path.join(outdir,'surveydata.pkl'),'wb'))
