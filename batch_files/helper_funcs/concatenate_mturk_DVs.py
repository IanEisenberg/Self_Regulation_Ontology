#!/usr/bin/env python3
import glob
import os
import pandas

#complete
output_loc = '/oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/'

for subset in ['mturk_complete', 'mturk_retest']:
    print('*'*79)
    print('Extracting Subset %s' % subset)
    print('*'*79)
    DVs = pandas.DataFrame()
    valence = pandas.DataFrame()
    for exp_file in glob.glob(os.path.join(output_loc, '%s_output' % subset, '*DV.json')):
        base_name = os.path.basename(exp_file)
        exp = base_name.replace('_%s_DV.json' % subset,'')
        print('Complete: Extracting %s DVs' % exp)
        exp_DVs = pandas.read_json(exp_file)
        exp_valence = pandas.read_json(exp_file.replace('.json','_valence.json'))
        exp_DVs.columns = [exp + '.' + c for c in exp_DVs.columns]
        exp_valence.columns = [exp + '.' + c for c in exp_valence.columns]
        DVs = pandas.concat([DVs,exp_DVs], axis = 1)
        valence = pandas.concat([valence,exp_valence], axis = 1)

    DVs.to_json(os.path.join(output_loc, '%s_DV.json' % subset))
    valence.to_json(os.path.join(output_loc, '%s_DV_valence.json' % subset))
