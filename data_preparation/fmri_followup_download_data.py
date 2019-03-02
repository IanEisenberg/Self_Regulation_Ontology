#!/usr/bin/env python3
import argparse
from expanalysis.experiments.jspsych import calc_time_taken, get_post_task_responses
from expanalysis.experiments.processing import post_process_data
from expanalysis.results import get_filters
from os import path
import pandas as pd
from selfregulation.utils.data_preparation_utils import calc_trial_order, \
    convert_date, convert_fmri_ids, download_data, get_bonuses, get_fmri_pay, \
    quality_check_correction
from selfregulation.utils.utils import get_info

parser = argparse.ArgumentParser(description='fMRI Analysis Entrypoint Script.')
parser.add_argument('--job', help='Specifies what part of the script to run. Options: download, extras, post, all").', default='post')

# get options
args = parser.parse_args()
job = args.job

#load Data
token = get_info('expfactory_token')
try:
    data_dir=get_info('data_directory')
except Exception:
    data_dir=path.join(get_info('base_directory'),'Data')

if job == 'download' or job == "all":
    #***************************************************
    # ********* Load Data **********************
    #**************************************************        
    pd.set_option('display.width', 200)
    figsize = [16,12]
    #set up filters
    filters = get_filters()
    drop_columns = ['battery_description', 'experiment_reference', 'experiment_version', \
             'experiment_name','experiment_cognitive_atlas_task']
    for col in drop_columns:
        filters[col] = {'drop': True}
    
    #***************************************************
    # ********* Download Data**********************
    #**************************************************  
    #load Data
    f = open(token)
    access_token = f.read().strip()  
    data = download_data(data_dir, access_token, filters = filters,  
                         battery = 'Self Regulation fMRI Battery',
                         url = 'http://www.expfactory.org/new_api/results/63/',
                         file_name = 'fmri_followup_data.pkl')
    
    data.reset_index(drop = True, inplace = True)
    
if job in ['extras', 'all']:
    print('Beginning "Extras"')
    #Process Data
    if job == "extras":
        #load Data
        data = pd.read_pickle(path.join(data_dir, 'fmri_followup_data.pkl'))
        data.reset_index(drop = True, inplace = True)
        print('Finished loading raw data')
        
    #***************************************************
    # ********* Add extras to data **********************
    #**************************************************  
    #anonymize data
    id_file = path.join(get_info('base_directory'), 'data_preparation', 
                        'samples', 'fmri_followup_expfactory_id_conversion.json')
    convert_fmri_ids(data, id_file=id_file)
    
    # record subject completion statistics
    (data.groupby('worker_id').count().finishtime).to_json(path.join(data_dir, 'admin', 'fmri_followup_worker_counts.json'))
    
    # add a few extras
    convert_date(data)
    bonuses = get_bonuses(data)
    calc_time_taken(data)
    get_post_task_responses(data)   
    calc_trial_order(data)
    
    # save data
    data.to_pickle(path.join(data_dir, 'fmri_followup_data_extras.pkl'))
    
    # calculate pay
    pay = get_fmri_pay(data)
    pay.to_json(path.join(data_dir, 'admin', 'fmri_followup_worker_pay.json'))
    print('Finished saving worker pay')
    
if job in ['post', 'all']:
    print('Beginning "Post"')
    #Process Data
    if job == "post":
        data = pd.read_pickle(path.join(data_dir, 'fmri_followup_data_extras.pkl'))
        data.reset_index(drop = True, inplace = True)
        print('Finished loading raw data')
        
    #***************************************************
    # ********* Post process data **********************
    #************************************************** 
    
    post_process_data(data)
    # correct for bugged stop signal quality correction
    quality_check_correction(data)
    data.to_pickle(path.join(data_dir,'fmri_followup_data_post.pkl'))
