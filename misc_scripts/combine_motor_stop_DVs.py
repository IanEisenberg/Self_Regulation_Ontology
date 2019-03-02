import pandas as pd
from os import path
base_dir = '/mnt/OAK'
input_dir = 'mturk_complete_fix'
output_dir = 'mturk_complete_output'

reactive = pd.read_json(path.join(base_dir, input_dir, 
'motor_selective_stop_signal_mturk_complete_DV.json'))

proactive = pd.read_json(path.join(base_dir, input_dir, 
'motor_selective_stop_signal_mturk_complete_DV_2.json'))

# needed before jspsych_processing was fixed
reactive.reactive_control_hddm_drift = reactive.reactive_control_hddm_drift*-1
proactive.proactive_control_hddm_drift = proactive.proactive_control_hddm_drift*-1


missing = set(reactive.columns) - set(proactive.columns)
proactive = pd.concat([proactive, reactive.loc[:, missing]], axis=1)
proactive.sort_index(1, inplace=True)
proactive.to_json(path.join(base_dir, output_dir, 
'motor_selective_stop_signal_mturk_complete_DV.json'))

# also valence
reactive = pd.read_json(path.join(base_dir, input_dir, 
'motor_selective_stop_signal_mturk_complete_DV_valence.json'))

proactive = pd.read_json(path.join(base_dir, input_dir, 
'motor_selective_stop_signal_mturk_complete_DV_valence_2.json'))

missing = set(reactive.columns) - set(proactive.columns)
proactive = pd.concat([proactive, reactive.loc[:, missing]], axis=1)
proactive.sort_index(1, inplace=True)
proactive.to_json(path.join(base_dir, output_dir, 
'motor_selective_stop_signal_mturk_complete_DV_valence.json'))
