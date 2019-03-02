#!/usr/bin/env python3
import numpy,pandas
import os

from selfregulation.utils.metadata_utils import metadata_subtract_one,metadata_replace_zero_with_nan
from selfregulation.utils.metadata_utils import metadata_change_two_to_zero_for_no
from selfregulation.utils.metadata_utils import write_metadata

def get_alcohol_drug_items(data):
    alcohol_drug_items={}
    for i,r in data.iterrows():
        if not r.text in alcohol_drug_items.keys():
            alcohol_drug_items[r.text]=r
    return alcohol_drug_items


def setup_itemid_dict():
    nominalvars=[]
    id_to_name={}
    id_to_name['alcohol_drugs_survey_2']='LifetimeSmoke100Cigs'
    id_to_name['alcohol_drugs_survey_3']='HowLongSmoked'
    id_to_name['alcohol_drugs_survey_4']='SmokeEveryDay'
    id_to_name['alcohol_drugs_survey_5']='CigsPerDay'
    # note: this one has "don't know" as response #5, need to fix
    id_to_name['alcohol_drugs_survey_6']='HowSoonSmokeAfterWaking'
    id_to_name['alcohol_drugs_survey_7']='OtherTobaccoProducts'
    nominalvars.append('alcohol_drugs_survey_7')
    id_to_name['alcohol_drugs_survey_8']='AlcoholHowOften'
    id_to_name['alcohol_drugs_survey_9']='AlcoholHowManyDrinksDay'
    id_to_name['alcohol_drugs_survey_10']='AlcoholHowOften6Drinks'
    id_to_name['alcohol_drugs_survey_11']='HowOftenCantStopDrinking'
    id_to_name['alcohol_drugs_survey_12']='HowOftenFailedActivitiesDrinking'
    id_to_name['alcohol_drugs_survey_13']='HowOftenDrinkMorning'
    id_to_name['alcohol_drugs_survey_14']='HowOftenGuiltRemorseDrinking'
    id_to_name['alcohol_drugs_survey_15']='HowOftenUnableRememberDrinking'
    id_to_name['alcohol_drugs_survey_16']='InjuredDrinking'
    nominalvars.append('alcohol_drugs_survey_16')
    id_to_name['alcohol_drugs_survey_17']='RelativeFriendConcernedDrinking'
    id_to_name['alcohol_drugs_survey_18']='CannabisPast6Months'
    id_to_name['alcohol_drugs_survey_20']='CannabisHowOften'
    id_to_name['alcohol_drugs_survey_21']='CannabisHoursStoned'
    id_to_name['alcohol_drugs_survey_22']='HowOftenCantStopCannabis'
    id_to_name['alcohol_drugs_survey_23']='HowOftenFailedActivitiesCannabis'
    id_to_name['alcohol_drugs_survey_24']='HowOftenDevotedTimeCannabis'
    id_to_name['alcohol_drugs_survey_25']='HowOftenMemoryConcentrationProblemCannabis'
    id_to_name['alcohol_drugs_survey_26']='HowOftenHazardousCannabis'
    id_to_name['alcohol_drugs_survey_27']='CannabisConsideredReduction'
    id_to_name['alcohol_drugs_survey_29']='OtherDrugs'
    id_to_name['alcohol_drugs_survey_30']='AbuseMoreThanOneDrugAtATime'
    id_to_name['alcohol_drugs_survey_31']='AbleToStopDrugs'
    id_to_name['alcohol_drugs_survey_32']='BlackoutFlashbackDrugUse'
    id_to_name['alcohol_drugs_survey_33']='FeelBadGuiltyDrugUse'
    id_to_name['alcohol_drugs_survey_34']='SpouseParentsComplainDrugUse'
    id_to_name['alcohol_drugs_survey_35']='NeglectedFamilyDrugUse'
    id_to_name['alcohol_drugs_survey_36']='EngagedInIllegalActsToObtainDrugs'
    id_to_name['alcohol_drugs_survey_37']='WidthdrawalSymptoms'
    id_to_name['alcohol_drugs_survey_38']='MedicalProblemsDueToDrugUse'
    return id_to_name,nominalvars

def get_metadata(alcohol_drug_items):
    id_to_name,nominalvars=setup_itemid_dict()
    alcohol_drug_dict={"MeasurementToolMetadata": {"Description": 'Health',
            "TermURL": ''}}
    for i in alcohol_drug_items:
            r=alcohol_drug_items[i]
            itemoptions=r.options
            itemid='_'.join(r['id'].split('_')[:4])
            assert itemid not in alcohol_drug_dict  # check for duplicates
            alcohol_drug_dict[itemid]={}
            alcohol_drug_dict[itemid]['Description']=r.text
            alcohol_drug_dict[itemid]['Levels']={}
            if itemid in nominalvars:
                alcohol_drug_dict[itemid]['Nominal']=True
            levelctr=0
            if type(itemoptions) in [list,dict]:
                for ii in itemoptions:
                    if not 'value' in ii:
                        value=levelctr
                        levelctr+=1
                    else:
                        value=ii['value']
                    alcohol_drug_dict[itemid]['Levels'][value]=ii['text']
    #rename according to more useful names
    alcohol_drug_dict_renamed={}
    for k in alcohol_drug_dict.keys():
        if not k in id_to_name.keys():
            alcohol_drug_dict_renamed[k]=alcohol_drug_dict[k]
        else:
            alcohol_drug_dict_renamed[id_to_name[k]]=alcohol_drug_dict[k]
    return alcohol_drug_dict_renamed

def add_alcohol_drug_item_labels(data):
    item_ids=[]
    for i,r in data.iterrows():
        item_ids.append('_'.join(r['id'].split('_')[:4]))
    data['item_id']=item_ids
    return data

def fix_item(d,v,metadata):
    """
    clean up responses
    """
    id_to_name,nominalvars=setup_itemid_dict()
    vname=id_to_name[v]
    # variables that need to have scale reversed - from 5-1 to 1-5
    reverse_scale=['DaysPhysicalHealthFeelings','Depressed','EverythingIsEffort',
                'Hopeless','Nervous','RestlessFidgety','Worthless']
    if vname in reverse_scale:
        tmp=numpy.array([float(i) for i in d])
        d.iloc[:]=tmp*-1 + 5
        print('reversed scale:',v,vname)
        metadata=metadata_reverse_scale(metadata)

    # variables that need to have one subtracted
    subtract_one=["AlcoholHowOften","AlcoholHowOften6Drinks",
                "HowOftenCantStopDrinking", "HowOftenDrinkMorning",
                "HowOftenFailedActivitiesDrinking","HowOftenGuiltRemorseDrinking"]
    if vname in subtract_one:
        tmp=[int(i) for i in d]
        d.iloc[:]=numpy.array(tmp)-1
        print('subtrated one:',v,vname)
        metadata=metadata_subtract_one(metadata)

    # replace 2 for "no" with zero
    change_two_to_zero_for_no=['LifetimeSmoke100Cigs']
    if vname in change_two_to_zero_for_no:
        tmp=numpy.array([float(i) for i in d.iloc[:]])
        tmp[tmp==2]=0
        d.iloc[:]=tmp
        print('changed two to zero for no:',v,vname)
        metadata=metadata_change_two_to_zero_for_no(metadata)

    return d,metadata

def save_alcohol_drug_data(data,survey_metadata,outdir):
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
    surveydata_renamed.to_csv(os.path.join(outdir,'alcohol_drugs.csv'))
    for v in nominalvars:
        del surveydata[v]
    surveydata_renamed_ord=surveydata.rename(columns=id_to_name)
    surveydata_renamed_ord.to_csv(os.path.join(outdir,'alcohol_drugs_ordinal.csv'))

    return outdir,surveydata_renamed
    
def process_alcohol_drug(data, outdir, meta_outdir):
    id_to_name,nominalvars=setup_itemid_dict()
    alcohol_drug_items=get_alcohol_drug_items(data)
    alcohol_drug_metadata=get_metadata(alcohol_drug_items)
    data=add_alcohol_drug_item_labels(data)
    datadir,surveydata_renamed=save_alcohol_drug_data(data,alcohol_drug_metadata, outdir)
    metadatadir=write_metadata(alcohol_drug_metadata,'alcohol_drugs.json',
        outdir=meta_outdir)
    return surveydata_renamed
        

