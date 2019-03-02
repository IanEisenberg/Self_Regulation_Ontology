#!/usr/bin/env python3
import numpy,pandas
import os

from selfregulation.utils.metadata_utils import metadata_subtract_one,metadata_replace_zero_with_nan
from selfregulation.utils.metadata_utils import metadata_change_two_to_zero_for_no
from selfregulation.utils.metadata_utils import write_metadata

def get_demog_items(data):
    demog_items={}
    for i,r in data.iterrows():
        if not r.text in demog_items.keys():
            demog_items[r.text]=r
    return demog_items

def setup_itemid_dict():
    nominalvars=[]
    id_to_name={}
    id_to_name['demographics_survey_2']='Sex'
    nominalvars.append('demographics_survey_2')
    id_to_name['demographics_survey_3']='Age'
    id_to_name['demographics_survey_4']='Race'
    nominalvars.append('demographics_survey_4')
    id_to_name['demographics_survey_5']='OtherRace'
    nominalvars.append('demographics_survey_5')
    id_to_name['demographics_survey_6']='HispanicLatino'
    nominalvars.append('demographics_survey_6')
    id_to_name['demographics_survey_7']='HighestEducation'
    id_to_name['demographics_survey_8']='HeightInches'
    id_to_name['demographics_survey_9']='WeightPounds'
    id_to_name['demographics_survey_10']='RelationshipStatus'
    nominalvars.append('demographics_survey_10')
    id_to_name['demographics_survey_11']='DivorceCount'
    id_to_name['demographics_survey_12']='LongestRelationship'
    id_to_name['demographics_survey_13']='RelationshipNumber'
    id_to_name['demographics_survey_14']='ChildrenNumber'
    id_to_name['demographics_survey_15']='HouseholdIncome'
    id_to_name['demographics_survey_16']='RetirementAccount'
    id_to_name['demographics_survey_17']='RetirementPercentStocks'
    id_to_name['demographics_survey_18']='RentOwn'
    nominalvars.append('demographics_survey_18')
    id_to_name['demographics_survey_19']='MortgageDebt'
    id_to_name['demographics_survey_20']='CarDebt'
    id_to_name['demographics_survey_21']='EducationDebt'
    id_to_name['demographics_survey_22']='CreditCardDebt'
    id_to_name['demographics_survey_23']='OtherDebtSources'
    nominalvars.append('demographics_survey_23')
    id_to_name['demographics_survey_24']='OtherDebtAmount'
    id_to_name['demographics_survey_25']='CoffeeCupsPerDay'
    id_to_name['demographics_survey_26']='TeaCupsPerDay'
    id_to_name['demographics_survey_27']='CaffienatedSodaCansPerDay'
    id_to_name['demographics_survey_28']='CaffieneOtherSourcesDayMG'
    id_to_name['demographics_survey_29']='GamblingProblem'
    nominalvars.append('demographics_survey_29')
    id_to_name['demographics_survey_30']='TrafficTicketsLastYearCount'
    id_to_name['demographics_survey_31']='TrafficAccidentsLifeCount'
    id_to_name['demographics_survey_32']='ArrestedChargedLifeCount'
    id_to_name['demographics_survey_33']='MotivationForParticipation'
    nominalvars.append('demographics_survey_33')
    id_to_name['demographics_survey_34']='MotivationOther'
    nominalvars.append('demographics_survey_34')
    return id_to_name,nominalvars

def get_metadata(demog_items):
    id_to_name,nominalvars=setup_itemid_dict()
    demog_dict={"MeasurementToolMetadata": {"Description": 'Demographics',
            "TermURL": ''}}
    for i in demog_items.keys():
        r=demog_items[i]
        itemoptions=r.options
        itemid='_'.join(r['id'].split('_')[:3])
        assert itemid not in demog_dict  # check for duplicates
        demog_dict[itemid]={}
        demog_dict[itemid]['Description']=r.text
        demog_dict[itemid]['Levels']={}
        if itemid in nominalvars:
            demog_dict[itemid]['Nominal']=True
        levelctr=0
#        if itemoptions is not None:
#        if isinstance(itemoptions, list):
#                for i in itemoptions:
#                    if not 'value' in i:
        if type(itemoptions) in [list,dict]:
                for ii in itemoptions:
                    if not 'value' in ii:
                        value=levelctr
                        levelctr+=1
                    else:
                        value=ii['value']
                    demog_dict[itemid]['Levels'][value]=ii['text']
    #rename according to more useful names
    demog_dict_renamed={}
    for k in demog_dict.keys():
        if not k in id_to_name.keys():
            demog_dict_renamed[k]=demog_dict[k]
        else:
            demog_dict_renamed[id_to_name[k]]=demog_dict[k]
    return demog_dict_renamed


def add_demog_item_labels(data):
    item_ids=[]
    for i,r in data.iterrows():
        item_ids.append('_'.join(r['id'].split('_')[:3]))
    data['item_id']=item_ids
    return data


def fix_item(d,v,metadata):
    """
    clean up responses and fix associated metadata
    """

    id_to_name,nominalvars=setup_itemid_dict()
    vname=id_to_name[v]
    # variables that need to have one subtracted
    subtract_one=['ArrestedChargedLifeCount','ChildrenNumber',
                    'RelationshipNumber','TrafficAccidentsLifeCount',
                    'TrafficTicketsLastYearCount','RentOwn']
    if vname in subtract_one:
        tmp=[int(i) for i in d]
        d.iloc[:]=numpy.array(tmp)-1
        print('subtrated one:',v,vname)

        metadata=metadata_subtract_one(metadata)

    # replace zero for "prefer not to say" with nan
    replace_zero_with_nan=['CarDebt','CreditCardDebt','EducationDebt',
                            'MortgageDebt','OtherDebtAmount']
    if vname in replace_zero_with_nan:
        tmp=numpy.array([float(i) for i in d.iloc[:]])
        tmp[tmp==0]=numpy.nan
        d.iloc[:]=tmp
        print('replaced %d zeros with nan:'%numpy.sum(numpy.isnan(tmp)),v,vname)
        metadata=metadata_replace_zero_with_nan(metadata)

    # replace 2 for "no" with zero
    change_two_to_zero_for_no=['RetirementAccount']
    if vname in change_two_to_zero_for_no:
        tmp=numpy.array([float(i) for i in d.iloc[:]])
        tmp[tmp==2]=0
        d.iloc[:]=tmp
        print('changed two to zero for no:',v,vname)
        metadata=metadata_change_two_to_zero_for_no(metadata)

    return d,metadata

def save_demog_data(data,survey_metadata, outdir):
    id_to_name,nominalvars=setup_itemid_dict()
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    unique_items=list(data.item_id.unique())
    surveydata=pandas.DataFrame(index=list(data.worker_id.unique()))
    for i in unique_items:
        qresult=data.query('item_id=="%s"'%i)
        matchitem=qresult.response
        matchitem.index=qresult['worker_id']
        matchitem,survey_metadata[id_to_name[i]]=fix_item(matchitem,i,survey_metadata[id_to_name[i]])
        surveydata.ix[matchitem.index,i]=matchitem

    surveydata_renamed=surveydata.rename(columns=id_to_name)
    surveydata_renamed.to_csv(os.path.join(outdir,'demographics.csv'))
    for v in nominalvars:
        del surveydata[v]
    surveydata_renamed_ord=surveydata.rename(columns=id_to_name)
    surveydata_renamed_ord.to_csv(os.path.join(outdir,'demographics_ordinal.csv'))

    return outdir,surveydata_renamed

def process_demographics(data, outdir, meta_outdir):
    id_to_name,nominalvars=setup_itemid_dict()
    demog_items=get_demog_items(data)
    demog_metadata=get_metadata(demog_items)
    data=add_demog_item_labels(data)
    datadir,surveydata_renamed=save_demog_data(data,demog_metadata, outdir)
    metadatadir=write_metadata(demog_metadata,'demographics.json',
        outdir=meta_outdir)
    return surveydata_renamed
        

    
    
    
