#!/usr/bin/env python3
import argparse
import sys
sys.path.append('/oak/stanford/groups/russpold/users/zenkavi/expfactory-analysis/expanalysis/experiments')
#from expanalysis.experiments.processing import get_exp_DVs_proptrials
from processing import get_exp_DVs_proptrials
from glob import glob
from os import path
import pandas as pd

#from selfregulation.utils.utils import get_info

#try:
#    data_dir=get_info('data_directory')
#except Exception:
    #data_dir=path.join(get_info('base_directory'),'Data')

data_dir = '/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_03-29-2018'

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('exp_id')
parser.add_argument('data')
parser.add_argument('--no_group', action='store_false')
# HDDM params
parser.add_argument('--out_dir', default=data_dir)
parser.add_argument('--hddm_samples', default=None, type=int)
parser.add_argument('--hddm_burn', default=None, type=int)
parser.add_argument('--hddm_thin', default=None, type=int)
parser.add_argument('--no_parallel', action='store_false')
parser.add_argument('--num_cores', default=None, type=int)
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--proptrials', default=1, type=float)
parser.add_argument('--rand', default='no', type=str)

args = parser.parse_args()

exp_id = args.exp_id
data = args.data
out_dir = args.out_dir
use_group = args.no_group
# HDDM variables
hddm_samples = args.hddm_samples
hddm_burn= args.hddm_burn
hddm_thin= args.hddm_thin
parallel = args.no_parallel
num_cores = args.num_cores
# mode for motor selective stop signal
mode = args.mode
proptrials = args.proptrials
rand = args.rand

#load Data
if(data=="retest"):
    dataset = pd.read_csv(path.join(data_dir, 'Individual_Measures', exp_id + '.csv.gz'),compression='gzip')

if(data=="complete"):
    dataset = pd.read_csv(path.join(data_dir, 't1_data/Individual_Measures', exp_id + '.csv.gz'),compression='gzip')

print('loaded dataset for %s' % exp_id)
#calculate DVs
group_kwargs = {'outfile': path.join(out_dir, exp_id),
                'parallel': parallel,
                'num_cores': num_cores}

if hddm_samples is not None:
    group_kwargs.update({'samples': hddm_samples,
                         'burn': hddm_burn,
                         'thin': hddm_thin})
if mode is not None:
    group_kwargs['mode'] = mode

print('Getting DVs for task %s %s %s %s' %(exp_id, data, str(proptrials), rand))

DV_df, valence_df, description = get_exp_DVs_proptrials(dataset, proptrials, rand, use_group_fun=use_group, group_kwargs=group_kwargs)

num_previous = len(glob(path.join(out_dir, exp_id + '_' + data + '_DV.json')))
postfix = '' if num_previous==0 else '_'+str(num_previous+1)
if not DV_df is None:
    if(rand=='yes'):
        DV_df.to_json(path.join(out_dir, 'random',exp_id + '_' + data + '_' + str(proptrials) + '_rand_DV%s.json' % postfix))
        print('save complete in %s' % path.join(out_dir, 'random',exp_id + '_' + data + '_' + str(proptrials) + '_rand_DV%s.json' % postfix))
    else:
        DV_df.to_json(path.join(out_dir, 'ordered',exp_id + '_' + data + '_' + str(proptrials) + '_DV%s.json' % postfix))
        print('save complete in %s' % path.join(out_dir, 'ordered',exp_id + '_' + data + '_' + str(proptrials) + '_DV%s.json' % postfix))
