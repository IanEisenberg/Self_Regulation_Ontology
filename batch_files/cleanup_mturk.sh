source activate $HOME/conda_envs/SRO
python helper_funcs/concatenate_mturk_DVs.py

# cleanup output of post processing and concatenation
behavioral_loc=/oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/

mv ${behavioral_loc}/*discovery* ${behavioral_loc}/derivatives
mv ${behavioral_loc}/*validation* ${behavioral_loc}/derivatives
mv ${behavioral_loc}/*extras* ${behavioral_loc}/derivatives
mv ${behavioral_loc}/*incomplete* ${behavioral_loc}/derivatives
mv ${behavioral_loc}/*failed* ${behavioral_loc}/derivatives


