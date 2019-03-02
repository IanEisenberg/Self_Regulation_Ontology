from glob import glob
import pandas as pd
import sys

model_dir = sys.argv[1]
tasks = [sys.argv[2]]
subsets = [sys.argv[3]]
output_dir = sys.argv[4]

if tasks == ['all']:
    tasks = ['adaptive_n_back', 'attention_network_task', 'choice_reaction_time', 'directed_forgetting', 'dot_pattern_expectancy' , 'local_global_letter', 'motor_selective_stop_signal' , 'recent_probes', 'shape_matching', 'simon', 'stim_selective_stop_signal', 'stop_signal', 'stroop', 'threebytwo']

if subsets == ['both']:
    subsets = ['retest', 't1']

for subset in subsets:
    subset_concat = pd.DataFrame()
    
    for task in tasks:
      task_path = model_dir + task + '_' + subset +'_*_hddm_flat.csv'
      file_list = glob(task_path)
      task_concat = pd.DataFrame()
      
      for file in file_list:
          sub_file = pd.read_csv(file)
          task_concat = task_concat.append(sub_file)
      
      task_concat = task_concat.add_prefix(task+'.')
      task_concat.rename(columns={task+'.Unnamed: 0':'subj_id'}, inplace=True)
      
      if sys.argv[2] != 'all':
          task_concat.to_csv(output_dir+task+'_'+subset+'_hddm_flat.csv')
      
      elif subset_concat.empty:
          subset_concat = task_concat
          
      else:    
          subset_concat = subset_concat.merge(task_concat, on=['subj_id'], how = 'outer')
          
    if sys.argv[2] == 'all':
        subset_concat.to_csv(output_dir+subset+'_hddm_flat.csv')