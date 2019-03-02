#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
from os import path
from selfregulation.utils.utils import get_info

# ************************************
# set discovery sample
# ************************************
def set_discovery_sample(n, discovery_n, seed = None):
    """
    :n: total size of sample
    :discovery_n: number of discovery subjects
    :param seed: if set, use as the seed for randomization
    :return array: array specifying which subjects, in order, are discovery/validation
    """
    if seed:
        np.random.seed(seed)
    sample = ['discovery']*discovery_n + ['validation']*(n-discovery_n)
    np.random.shuffle(sample)
    return sample

    
seed = 1960

#set discovery sample
n = 500
discovery_n = 200
subjects = ['s' + str(i).zfill(3) for i in range(1,n+1)]
subject_order = set_discovery_sample(n, discovery_n, seed)
subject_assignment_df = pd.DataFrame({'dataset': subject_order}, index = subjects)
subject_assignment_df.to_csv('samples/subject_assignment.csv')

# ************************************
# set test-retest sample
# ************************************

#define test-retest sample function
data_dir=get_info('base_directory')
local_dir = path.join(data_dir,'Data','Local')
# read preprocessed data
discovery_data = pd.read_json(path.join(local_dir,'mturk_discovery_data_post.json')).reset_index(drop = True)
worker_lookup = json.load(open(path.join(local_dir,"worker_lookup.json"),'r'))

def get_testretest_order(n = 200, seed = seed):
    """
    :n: total size of sample
    :discovery_n: number of discovery subjects
    :param seed: if set, use as the seed for randomization
    :return array: array specifying which subjects, in order, are discovery/validation
    """
    np.random.seed(seed)
    order = np.random.permutation(range(0,n))
    return order

def get_retest_workers(data, n = 200):
    retest_order = get_testretest_order()
    retest_workers_anonym = np.sort(data.worker_id.unique())[retest_order[0:n]]
    retest_workers = [worker_lookup[w] for w in retest_workers_anonym]
    return pd.Series(retest_workers)
    
retest_workers = get_retest_workers(discovery_data)
retest_workers.to_csv('samples/retest_workers.csv')