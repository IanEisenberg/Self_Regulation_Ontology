from os import path
import pandas as pd

data_dir = '/oak/stanford/groups/russpold/users/ieisenbe/uh2/behavioral_data/mturk_retest_output/hddm_refits/'
out_dir = '/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_12-19-2018/Local/'

#Read in DVs and valence
label = 'hddm_refit'
DV_df = pd.read_json(path.join(data_dir,'mturk_' + label + '_DV.json'))
valence_df = pd.read_json(path.join(data_dir,'mturk_' + label + '_DV_valence.json'))

readme_lines = []

#drop na columns
DV_df.dropna(axis = 1, how = 'all', inplace = True)
DV_df.to_csv(path.join(out_dir, 'hddm_refits_exhaustive.csv'))
readme_lines += ["hddm_refits_exhaustive.csv: all variables for hddm's for retest subjects' t1 data\n\n"]

readme = open(path.join(out_dir, "README.txt"), "a")
readme.writelines(readme_lines)
readme.close()
