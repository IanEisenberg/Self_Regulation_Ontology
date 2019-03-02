from expanalysis.experiments.jspsych import calc_time_taken, get_post_task_responses
from expanalysis.experiments.processing import post_process_data
import json
from os import path
import pandas as pd
import pickle

from selfregulation.utils.data_preparation_utils import calc_trial_order, convert_date, get_bonuses, get_pay, remove_failed_subjects
from selfregulation.utils.retest_data_utils import anonymize_retest_data

#set token and data directory

data_dir=path.join('/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/','Retest_01-23-2018', 'Local')

#load data 
#data = pd.read_json(path.join(data_dir, 'mturk_retest_data.json'))
data = pd.read_pickle(path.join(data_dir, 'mturk_retest_data.pkl'))

# In case index got messed up
data.reset_index(drop = True, inplace = True)

worker_lookup = anonymize_retest_data(data, data_dir)
json.dump(worker_lookup, open(path.join(data_dir, 'retest_worker_lookup.json'),'w'))

# record subject completion statistics
(data.groupby('worker_id').count().finishtime).to_json(path.join(data_dir, 'retest_worker_counts.json'))

# add a few extras
convert_date(data)
bonuses = get_bonuses(data)
calc_time_taken(data)
get_post_task_responses(data)   
calc_trial_order(data)

# save data (gives an error but works? - NO EMPTY FILE. FIGURE OUT!
# seems to be a memory issue; temp solution with pickling)
file_name = 'mturk_retest_data_extras'

try:
    data.to_json(path.join(data_dir, file_name+'.json'))
except:
    pickle.dump(data, open(path.join(data_dir, file_name+'.pkl'), 'wb'), -1)
    

# calculate pay
pay = get_pay(data)
pay.to_json(path.join(data_dir, 'retest_worker_pay.json'))

# create dataframe to hold failed data
failed_data = pd.DataFrame()

post_process_data(data)

failures = remove_failed_subjects(data)
failed_data = pd.concat([failed_data,failures])
#data.to_json(path.join(data_dir,'mturk_retest_data_post.json'))
pickle.dump(data, open(path.join(data_dir, 'mturk_retest_data_post.pkl'), 'wb'), -1)

# save failed data
failed_data = failed_data.reset_index(drop = True)
failed_data.to_json(path.join(data_dir, 'mturk_retest_failed_data_post.json'))
