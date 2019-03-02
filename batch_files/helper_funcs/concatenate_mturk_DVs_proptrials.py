#!/usr/bin/env python3
import glob
import os
import pandas

#complete
output_loc = '/oak/stanford/groups/russpold/users/ieisenbe/uh2/behavioral_data/mturk_retest_output/trial_num/ordered/'

for subset in ['complete', 'retest']:
    for proptrials in ['0.25', '0.5', '0.75']:
        print('*'*79)
        print('Extracting Subset %s for %s of trials' % (subset, proptrials))
        print('*'*79)
        DVs = pandas.DataFrame()
        for exp_file in glob.glob(os.path.join(output_loc, '*%s_%s' % (subset, proptrials)+ '*DV.json')):
            base_name = os.path.basename(exp_file)
            exp = base_name.replace('_%s_%s_DV.json' % (subset, proptrials),'')
            print('Complete: Extracting %s DVs' % exp)
            exp_DVs = pandas.read_json(exp_file)
            exp_DVs.columns = [exp + '.' + c for c in exp_DVs.columns]
            DVs = pandas.concat([DVs,exp_DVs], axis = 1)
        DVs.to_json(os.path.join(output_loc, '%s_%s_DV.json' % (subset, proptrials)))
