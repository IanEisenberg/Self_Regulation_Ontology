#!/usr/bin/env python3
import argparse
import datetime
from expanalysis.experiments.processing import  extract_experiment
from glob import glob
from os import makedirs, path
import numpy as np
import pandas as pd
from data_preparation.process_alcohol_drug import process_alcohol_drug
from data_preparation.process_demographics import process_demographics
from data_preparation.process_health import process_health
from selfregulation.utils.data_preparation_utils import convert_var_names, drop_failed_QC_vars, drop_vars, get_items
from selfregulation.utils.data_preparation_utils import remove_correlated_task_variables, remove_outliers, save_task_data
from selfregulation.utils.data_preparation_utils import transform_remove_skew
from selfregulation.utils.utils import get_info
from selfregulation.utils.r_to_py_utils import missForest
from selfregulation.utils.reference_utils import gen_reference_item_text


parser = argparse.ArgumentParser(description='fMRI Analysis Entrypoint Script.')
parser.add_argument('--labels', help="labels of the datasets",
                    nargs='+', default=['mturk_complete', 'mturk_retest'])
args = parser.parse_args()
data_labels = args.labels

#******************************
#*** Save Data *********
#******************************
date = datetime.date.today().strftime("%m-%d-%Y")

output_dir=path.join(get_info('base_directory'),'Data')
#load Data
data_dir=get_info('data_directory')


# read preprocessed data
datasets = []
for label in data_labels:
    try:
        data = pd.read_pickle(path.join(data_dir,label + '_data_post.pkl')).reset_index(drop = True)
    except FileNotFoundError:
        print("Couldn't find %s" % label + '_data_post.pkl')
        continue
    if 'mturk' in label:
        directory = path.join(output_dir,label.split('mturk_')[1].title() + '_' + date)
    else:
        directory = path.join(output_dir,label.title() + '_' + date)
    if not path.exists(directory):
        makedirs(directory)
    try:
        DVs = pd.read_json(path.join(data_dir,label + '_DV.json'))
        DVs_valence = pd.read_json(path.join(data_dir,label + '_DV_valence.json'))
    except ValueError:
        print("Couldn't find %s DV datasets" % label)
        DVs = []
        DVs_valence = []
    datasets.append((data,directory, DVs, DVs_valence))
    
# calculate DVs
for data,directory, DV_df, valence_df in datasets:
    readme_lines = []
    meta_dir = path.join(directory,'metadata')
    reference_dir = path.join(directory,'references')
    if not path.exists(meta_dir):
        makedirs(meta_dir)
    if not path.exists(reference_dir):
        makedirs(reference_dir)
    # save target datasets
    print('Saving to %s...' % directory)
    print('Saving target measures...')
    demog_data = extract_experiment(data,'demographics_survey')
    demog_data = process_demographics(demog_data, directory, meta_dir)
    alcohol_drug_data = extract_experiment(data,'alcohol_drugs_survey')
    alcohol_drug_data = process_alcohol_drug(alcohol_drug_data, directory, meta_dir)
    health_data = extract_experiment(data,'k6_survey')
    health_data = process_health(health_data, directory, meta_dir)
    activity_level = DV_df.pop('leisure_time_activity_survey.activity_level')
    # concatenate targets
    target_data = pd.concat([demog_data, alcohol_drug_data, 
                             health_data, activity_level], axis = 1)
    target_data.to_csv(path.join(directory,'demographic_health.csv'))
    # save items
    items_df = get_items(data)
    print('Saving items...')
    subjectsxitems = items_df.pivot('worker','item_ID','coded_response')
    # ensure there are the correct number of items
    if subjectsxitems.shape[1] != 593:
        print('Wrong number of items found for label: %s' % label)
        continue
    # save items
    items_df.to_csv(path.join(directory, 'items.csv.gz'), compression = 'gzip')
    subjectsxitems.to_csv(path.join(directory, 'subject_x_items.csv'))
    convert_var_names(subjectsxitems)
    assert np.max([len(name) for name in subjectsxitems.columns])<=8, \
        "Found column names longer than 8 characters in short version"
    # save Individual Measures
    save_task_data(directory, data)
    if 'Complete' in directory:
        # save demographic targets reference
        np.savetxt(path.join(reference_dir,'demographic_health_reference.csv'), target_data.columns, fmt = '%s', delimiter=",")
        gen_reference_item_text(items_df)

    readme_lines += ["demographics_survey.csv: demographic information from expfactory-surveys\n\n"]
    readme_lines += ["alcohol_drug_survey.csv: alcohol, smoking, marijuana and other drugs from expfactory-surveys\n\n"]
    readme_lines += ["ky_survey.csv: mental health and neurological/health conditions from expfactory-surveys\n\n"]
    readme_lines += ["items.csv.gz: gzipped csv of all item information across surveys\n\n"]
    readme_lines += ["subject_x_items.csv: reshaped items.csv such that rows are subjects and columns are individual items\n\n"]
    readme_lines += ["Individual Measures: directory containing gzip compressed files for each individual measures\n\n"]
    # ************************************
    # ********* Save DV dataframes **
    # ************************************
    def get_flip_list(valence_df):
        #flip negative signed valence DVs
        valence_df = valence_df.replace(to_replace={np.nan: 'NA'})
        flip_df = np.floor(valence_df.replace(to_replace ={'Pos': 1, 'NA': 1, 'Neg': -1}).mean())
        valence_df = pd.Series(data = [col.unique()[0] for i,col in valence_df.iteritems()], index = valence_df.columns)
        return flip_df, valence_df

    if len(DV_df) > 0:
        # drop failed QC vars
        drop_failed_QC_vars(DV_df,data)
        
        #save valence
        flip_df, valence_df = get_flip_list(valence_df)
        flip_df.to_csv(path.join(directory, 'DV_valence.csv'))
        readme_lines += ["DV_valence.csv: Subjective assessment of whether each variable's 'natural' direction implies 'better' self regulation\n\n"]
        
        #drop na columns
        DV_df.dropna(axis = 1, how = 'all', inplace = True)
        DV_df.to_csv(path.join(directory, 'variables_exhaustive.csv'))
        readme_lines += ["variables_exhaustive.csv: all variables calculated for each measure\n\n"]
          
        # drop other columns of no interest
        subset = drop_vars(DV_df, saved_vars = ['adaptive_n_back.hddm_drift_load', 'simple_reaction_time.avg_rt', 'shift_task.acc'])
        # make subset without EZ variables
        noDDM_subset = drop_vars(DV_df, saved_vars = ["\.acc$", "\.avg_rt$"])
        noDDM_subset = drop_vars(noDDM_subset, drop_vars = ['EZ', 'hddm'])
        noDDM_subset.to_csv(path.join(directory, 'meaningful_variables_noDDM.csv'))
        readme_lines += ["meaningful_variables_noDDM.csv: subset of exhaustive data to only meaningful variables with DDM parameters removed\n\n"]
        # make subset without acc/rt vars and just EZ DDM
        EZ_subset = drop_vars(subset, drop_vars = ['_acc', '_rt', 'hddm'], saved_vars = ['simple_reaction_time.avg_rt', 'dospert_rt_survey'])
        EZ_subset.to_csv(path.join(directory, 'meaningful_variables_EZ.csv'))
        readme_lines += ["meaningful_variables_EZ.csv: subset of exhaustive data to only meaningful variables with rt/acc parameters removed (replaced by EZ DDM params)\n\n"]
        # make subset without acc/rt vars and just hddm DDM
        hddm_subset = drop_vars(subset, drop_vars = ['_acc', '_rt', 'EZ'], saved_vars = ['simple_reaction_time.avg_rt', 'dospert_rt_survey'])
        hddm_subset.to_csv(path.join(directory, 'meaningful_variables_hddm.csv'))
        readme_lines += ["meaningful_variables_hddm.csv: subset of exhaustive data to only meaningful variables with rt/acc parameters removed (replaced by hddm DDM params)\n\n"]
        
        # save files that are selected for use
        selected_variables = hddm_subset
        selected_variables.to_csv(path.join(directory, 'meaningful_variables.csv'))
        readme_lines += ["meaningful_variables.csv: Same as meaningful_variables_hddm.csv\n\n"]
        
        # clean data
        selected_variables_clean = transform_remove_skew(selected_variables)
        selected_variables_clean = remove_outliers(selected_variables_clean)
        selected_variables_clean = remove_correlated_task_variables(selected_variables_clean)
        selected_variables_clean.to_csv(path.join(directory, 'meaningful_variables_clean.csv'))
        readme_lines += ["meaningful_variables_clean.csv: same as meaningful_variables.csv with skewed variables transformed and then outliers removed \n\n"]
        
        # imputed data
        selected_variables_imputed, error = missForest(selected_variables_clean)
        selected_variables_imputed.to_csv(path.join(directory, 'meaningful_variables_imputed.csv'))
        readme_lines += ["meaningful_variables_imputed.csv: meaningful_variables_clean.csv after imputation with missForest\n\n"]

        #save selected variables
        selected_variables_reference = valence_df
        selected_variables_reference.loc[selected_variables.columns].to_csv(path.join(reference_dir, 'selected_variables_reference.csv'))
                
        # save task data subset
        task_data = drop_vars(selected_variables, ['survey'], saved_vars = ['holt','cognitive_reflection'])
        task_data.to_csv(path.join(directory, 'taskdata.csv'))
        task_data_clean = drop_vars(selected_variables_clean, ['survey'], saved_vars = ['holt','cognitive_reflection'])
        task_data_clean.to_csv(path.join(directory, 'taskdata_clean.csv'))
        task_data_imputed = drop_vars(selected_variables_imputed, ['survey'], saved_vars = ['holt','cognitive_reflection'])
        task_data_imputed.to_csv(path.join(directory, 'taskdata_imputed.csv'))
        readme_lines += ["taskdata*.csv: taskdata are the same as meaningful_variables excluded surveys. Note that imputation is performed on the entire dataset including surveys\n\n"]
        
        # create task selection dataset
        task_selection_data = drop_vars(selected_variables_imputed, ['stop_signal.SSRT_low', '^stop_signal.proactive'])
        task_selection_data.to_csv(path.join(directory,'meaningful_variables_imputed_for_task_selection.csv'))
        task_selection_taskdata = drop_vars(task_data_imputed, ['stop_signal.SSRT_low', '^stop_signal.proactive'])
        task_selection_taskdata.to_csv(path.join(directory,'taskdata_imputed_for_task_selection.csv'))
        #save selected variables
        selected_variables_reference.loc[task_selection_data.columns].to_csv(path.join(reference_dir, 'selected_variables_for_task_selection_reference.csv'))
        
        files = glob(path.join(directory,'*csv'))
        files = [f for f in files if not any(i in f for i in ['demographic','health','alcohol_drug'])]
        for f in files:
            name = f.split('/')[-1]
            df = pd.DataFrame.from_csv(f)
            convert_var_names(df)
            df.to_csv(path.join(directory, 'short_' + name))
            print('short_' + name)
        readme_lines += ["short*.csv: short versions are the same as long versions with variable names shortened using variable_name_lookup.csv\n\n"]
        
        # write README
        readme = open(path.join(directory, "README.txt"), "w")
        readme.writelines(readme_lines)
        readme.close()
