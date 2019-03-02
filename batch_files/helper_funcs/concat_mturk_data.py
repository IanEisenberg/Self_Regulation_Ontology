#!/usr/bin/env python3
from os import path
import pandas as pd
from selfregulation.utils.utils import get_info

try:
    data_dir=get_info('data_directory')
except Exception:
    data_dir=path.join(get_info('base_directory'),'Data')

complete = None
# concatenate discovery and validation data into one complete
discovery_path = path.join(data_dir, 'mturk_discovery_data_post.pkl')
validation_path = path.join(data_dir, 'mturk_validation_data_post.pkl')
complete_path = path.join(data_dir, 'mturk_complete_data_post.pkl')
if path.exists(discovery_path) and path.exists(validation_path):
    discovery = pd.read_pickle(discovery_path)
    validation = pd.read_pickle(validation_path)
    complete = pd.concat([discovery, validation])
    complete.to_pickle(complete_path)

# separate complete into two data subsets for particularly memory intensive analyses (DDM)
if path.exists(complete_path):
    if not complete:
        complete = pd.read_pickle(complete_path)
    workers = complete.worker_id.unique()
    mid = len(workers)//2
    subset1 = complete.query('worker_id in %s' % list(workers)[:mid])
    subset2 = complete.query('worker_id in %s' % list(workers)[mid:])
    subset1.to_pickle(path.join(data_dir, 'mturk_complete_subset1_data_post.pkl'))
    subset2.to_pickle(path.join(data_dir, 'mturk_complete_subset2_data_post.pkl'))
