'''
Utility functions for the ontology project
'''
from datetime import datetime
from expanalysis.experiments.processing import extract_row, extract_experiment
from expanalysis.results import Result
from expanalysis.experiments.utils import remove_duplicates, result_filter
from selfregulation.utils.utils import get_info
import json
import numpy as np
import os
import pandas as pd
from time import time


#***************************************************
# ********* Helper Functions **********************
#**************************************************
def anonymize_data(data):
    complete_workers = (data.groupby('worker_id').count().finishtime>=63)
    complete_workers = list(complete_workers[complete_workers].index)
    workers = data.groupby('worker_id').finishtime.max().sort_values().index
    # make new ids
    new_ids = []
    id_index = 1
    for worker in workers:
        if worker in complete_workers:
            new_ids.append('s' + str(id_index).zfill(3))
            id_index += 1
        else:
            new_ids.append(worker)
    data.replace(workers, new_ids,inplace = True)
    return {x:y for x,y in zip(new_ids, workers)}

def calc_bonuses(data):
    bonus_experiments = ['angling_risk_task_always_sunny', 'two_stage_decision',
                         'columbia_card_task_hot', 'columbia_card_task_cold', 'hierarchical_rule',
                         'kirby','discount_titrate','bickel_titrator']
    bonuses = []
    for row in data.iterrows():
        if row[1]['experiment_exp_id'] in bonus_experiments:
            try:
                df = extract_row(row[1], clean = False)
            except TypeError:
                bonuses.append(np.nan)
                continue
            bonus = df.iloc[-1].get('performance_var','error')
            if pd.isnull(bonus):
                bonus = df.iloc[-5].get('performance_var','error')
            if not isinstance(bonus,(int,float)):
                bonus = json.loads(bonus)['amount']
            bonuses.append(bonus)
        else:
            bonuses.append(np.nan)
    data.loc[:,'bonus'] = bonuses
    data.loc[:,'bonus_zscore'] = data['bonus']
    means = data.groupby('experiment_exp_id').bonus.mean()
    std = data.groupby('experiment_exp_id').bonus.std()
    for exp in bonus_experiments:
        data.loc[data.experiment_exp_id == exp,'bonus_zscore'] = (data[data.experiment_exp_id == exp].bonus-means[exp])/std[exp]
        
def calc_trial_order(data):
    sorted_data = data.sort_values(by = ['worker_id','finishtime'])
    num_exps = data.groupby('worker_id')['finishtime'].count() 
    order = []    
    for x in num_exps:
        order += range(x)
    data.loc[sorted_data.index, 'trial_order'] = order
    
def check_timing(df):
    df.loc[:, 'time_diff'] = df['time_elapsed'].diff()
    timing_cols = pd.concat([df['block_duration'], df.get('feedback_duration'), df['timing_post_trial'].shift(1)], axis = 1)
    df.loc[:, 'expected_time'] = timing_cols.sum(axis = 1)
    df.loc[:, 'timing_error'] = df['time_diff'] - df['expected_time']
    errors = [df[abs(df['timing_error']) < 500]['timing_error'].mean(), df[df['timing_error'] < 500]['timing_error'].max()]
    return errors

def convert_date(data):
    new_date = data.loc[:,'finishtime'].map(lambda date: datetime.strptime(date[:-8],'%Y-%m-%dT%H:%M:%S'))
    data.loc[:,'finishtime'] = new_date
        
def convert_fmri_ids(data, id_file):
    conversion_lookup = json.load(open(id_file,'r'))
    data.worker_id.replace(conversion_lookup, inplace = True)

def convert_item_names(to_convert):
    '''Convert array of variable names or columns/index of a dataframe. Assumes that all values either
    come from short of long variable names. If a dataframe is passed, variable conversion
    is done in place.
    '''
    assert(isinstance(to_convert, (list, np.ndarray, pd.DataFrame))), \
        'Object to convert must be a list, numpy array or pandas DataFrame'
    var_lookup = pd.Series.from_csv('../data_preparation/item_name_lookup.csv')
    inverse_lookup = pd.Series(index = var_lookup.values, data = var_lookup.index)
    
    if type(to_convert) == pd.DataFrame:
        # convert columns if there are dependent variable names
        if to_convert.columns[0] in var_lookup:
            new_columns = [var_lookup.loc[c] if c in var_lookup.index else c for c in to_convert.columns]
        elif to_convert.columns[0] in inverse_lookup:
            new_columns = [inverse_lookup.loc[c] if c in inverse_lookup.index else c for c in to_convert.columns]
        else:
            new_columns = to_convert.columns
        to_convert.columns = new_columns
        # convert index if there are dependent variable names
        if to_convert.index[0] in var_lookup:
            new_index = [var_lookup.loc[i] if i in var_lookup.index else i for i in to_convert.index]
        elif to_convert.index[0] in inverse_lookup:
            new_index = [inverse_lookup.loc[i] if i in inverse_lookup.index else i for i in to_convert.index]
        else: 
            new_index = to_convert.index
        to_convert.index = new_index
    elif isinstance(to_convert, (list, np.ndarray)):
        if to_convert[0] in var_lookup:
            return  [var_lookup.loc[c] if c in var_lookup.index else c for c in to_convert]
        elif to_convert[0] in inverse_lookup:
            return  [inverse_lookup.loc[c] if c in inverse_lookup.index else c for c in to_convert]
    
def convert_var_names(to_convert):
    '''Convert array of variable names or columns/index of a dataframe. Assumes that all values either
    come from short of long variable names. If a dataframe is passed, variable conversion
    is done in place.
    '''
    assert(isinstance(to_convert, (list, np.ndarray, pd.DataFrame))), \
        'Object to convert must be a list, numpy array or pandas DataFrame'
    reference_location = os.path.join(get_info('base_directory'), 'references', 'variable_name_lookup.csv')
    var_lookup = pd.Series.from_csv(reference_location)
    inverse_lookup = pd.Series(index = var_lookup.values, data = var_lookup.index)
    
    if type(to_convert) == pd.DataFrame:
        # convert columns if there are dependent variable names
        if to_convert.columns[0] in var_lookup:
            new_columns = [var_lookup.loc[c] if c in var_lookup.index else c for c in to_convert.columns]
        elif to_convert.columns[0] in inverse_lookup:
            new_columns = [inverse_lookup.loc[c] if c in inverse_lookup.index else c for c in to_convert.columns]
        else:
            new_columns = to_convert.columns
        to_convert.columns = new_columns
        # convert index if there are dependent variable names
        if to_convert.index[0] in var_lookup:
            new_index = [var_lookup.loc[i] if i in var_lookup.index else i for i in to_convert.index]
        elif to_convert.index[0] in inverse_lookup:
            new_index = [inverse_lookup.loc[i] if i in inverse_lookup.index else i for i in to_convert.index]
        else: 
            new_index = to_convert.index
        to_convert.index = new_index
    elif isinstance(to_convert, (list, np.ndarray)):
        if to_convert[0] in var_lookup:
            return  [var_lookup.loc[c] if c in var_lookup.index else c for c in to_convert]
        elif to_convert[0] in inverse_lookup:
            return  [inverse_lookup.loc[c] if c in inverse_lookup.index else c for c in to_convert]
            
    
def download_data(data_loc, access_token = None, filters = None, 
                  battery = None, save = True, url = None, file_name=None):
    start_time = time()
    #Load Results from Database
    results = Result(access_token, filters = filters, url = url)
    data = results.data
    if 'experiment_exp_id' not in data.columns:
        data.loc[:,'experiment_exp_id'] = [x['exp_id'] for x in data['experiment']]
    if 'experiment_template' not in data.columns:
        data.loc[:,'experiment_template'] = [x['template'] for x in data['experiment']]
    if battery:
        data = result_filter(data, battery = battery)

    # remove duplicates
    remove_duplicates(data)
    
    # remove a few mistakes from data
    data = data.query('worker_id not in ["A254JKSDNE44AM", "A1O51P5O9MC5LX"]') # Sandbox workers
    data.reset_index(drop = True, inplace = True)    
    
    # if saving, save the data and the lookup file for anonymized workers
    if save == True:
        if file_name == None:
            file_name = 'mturk_data.json'
        if file_name[-4:] == 'json':
            data.to_json(os.path.join(data_loc,file_name))
        elif file_name[-3:] == 'pkl':
            data.to_pickle(os.path.join(data_loc,file_name))
        print('Finished saving')
    
    finish_time = (time() - start_time)/60
    print('Finished downloading data. Time taken: ' + str(finish_time))
    return data                 

def drop_failed_QC_vars(df, data):
    failed_exps = data.query('passed_QC==False')
    for i, values in failed_exps[['experiment_exp_id','worker_id']].iterrows():
        df.loc[values[1],df.filter(regex = values[0]).columns] = np.nan

def drop_vars(data, drop_vars = [], saved_vars = []):
    if len(drop_vars) == 0:
        # variables that are calculated without regard to their actual interest
        basic_vars = ["\.missed_percent$","\.acc$","\.avg_rt_error$","\.std_rt_error$","\.avg_rt$","\.std_rt$"]
        #unnecessary ddm params
        ddm_vars = ['.*\.(EZ|hddm)_(drift|thresh|non_decision).+$']
        # variables that are of theoretical interest, but we aren't certain enough to include in 2nd stage analysis
        exploratory_vars = ["\.congruency_seq", "\.post_error_slowing$"]
        # task variables that are irrelevent to second stage analysis, either because they are correlated
        # with other DV's or are just of no interest. Each row is a task
        task_vars = ["demographics", # demographics
                    "(keep|release)_loss_percent", # angling risk task
                    ".first_order", "bis11_survey.total", # bis11
                    "bis_bas_survey.BAS_total", 
                    "dietary_decision.prop_healthy_choice", # dietary decision
                    "dot_pattern_expectancy.*errors", # DPX errors
                    "eating_survey.total", # eating total score
                    "five_facet_mindfulness_survey.total", 
                    "\.risky_choices$", "\.number_of_switches", # holt and laury
                    "boxes_opened$", # information sampling task
                    "_total_points$", # IST
                    "\.go_acc$", "\.nogo_acc$", "\.go_rt$", "go_nogo.*error.*", #go_nogo
                    "discount_titrate.hyp_discount_rate", "discount_titrate.hyp_discount_rate_(glm|nm)"  #delay discounting
                    "kirby.percent_patient","kirby.hyp_discount_rate$",  "kirby.exp_discount.*", 
                    "\.warnings$", "_notnow$", "_now$", #kirby and delay discounting
                    "auc", # bickel
                    "local_global_letter.*error.*", # local global errors
                    "PRP_slowing", # PRP_two_choices
                    "shape_matching.*prim.*", # shape matching prime measures
                    "sensation_seeking_survey.total", # SSS
                    "DDS", "DNN", "DSD", "SDD", "SSS", "DDD", "stimulus_interference_rt", # shape matching
                    "shift_task.*errors", "shift_task.model_fit", "shift_task.conceptual_responses", #shift task
                    "shift_task.fail_to_maintain_set", 'shift_task.perseverative_responses', # shift task continued
                     "go_acc","stop_acc","go_rt_error","go_rt_std_error", "go_rt","go_rt_std", # stop signal
                     "stop_rt_error","stop_rt_error_std","SS_delay", "^stop_signal.SSRT$", # stop signal continue
                     "stop_signal.*errors", "inhibition_slope", # stop signal continued
                     "stroop.*errors", # stroop
                     "threebytwo.*inhibition", # threebytwo
                     "num_correct", "weighted_performance_score", # tower of london
                     "sentiment_label" ,# writing task
                     "log_ll", "match_pct", "min_rss", #fit indices
                     "num_trials", "num_stop_trials"#num trials
                    ]
        drop_vars = basic_vars + exploratory_vars + task_vars + ddm_vars
    drop_vars = '|'.join(drop_vars)
    if len(saved_vars) > 0 :
        saved_vars = '|'.join(saved_vars)
        saved_columns = data.filter(regex=saved_vars)
        dropped_data =  data.drop(data.filter(regex=drop_vars).columns, axis = 1)
        final_data = dropped_data.join(saved_columns).sort_index(axis = 1)
    else:
        final_data = data.drop(data.filter(regex=drop_vars).columns, axis = 1)
    return final_data
    
def get_bonuses(data, mean=10, limit=10):
    if 'bonus_zscore' not in data.columns:
        calc_bonuses(data)
    workers_finished = data.groupby('worker_id').count().finishtime==63
    index = list(workers_finished[workers_finished].index)
    tmp_data = data.query('worker_id in %s' % index)
    tmp_bonuses = tmp_data.groupby('worker_id').bonus_zscore.mean()
    min_score = tmp_bonuses.min()
    max_score = tmp_bonuses.max()
    num_tasks_bonused = data.groupby('worker_id').bonus_zscore.count()
    bonuses = data.groupby('worker_id').bonus_zscore.mean()
    bonuses = (bonuses-min_score)/(max_score-min_score)*limit+(mean-limit/2)
    bonuses = bonuses.map(lambda x: round(x,1))*num_tasks_bonused/8
    print('Finished getting bonuses')
    return bonuses

def get_credit(data):
    credit_array = []
    for i,row in data.iterrows():
        if row['experiment_template'] in 'jspsych':
            df = extract_row(row, clean = False)
            credit_var = df.iloc[-1].get('credit_var',np.nan)
            if credit_var != None:
                credit_array.append(float(credit_var))
            else:
                credit_array.append(np.nan)
        else:
            credit_array.append(np.nan)
    data.loc[:,'credit'] = credit_array   
    
def get_items(data):
    excluded_surveys = ['holt_laury_survey']
    items = []
    responses = []
    responses_text = []
    options = []
    workers = []
    item_nums = []
    exps = []
    for exp in data.experiment_exp_id.unique():
        if 'survey' in exp and exp not in excluded_surveys:
            survey = extract_experiment(data,exp)
            try:
                responses += list(survey.response.map(lambda x: float(x)))
            except ValueError:
                continue
            items += list(survey.text)
            responses_text += [str(i) for i in list(survey.response_text)]
            options += list(survey.options)
            workers += list(survey.worker_id)
            item_nums += list(survey.question_num)
            exps += [exp] * len(survey.text)
    
    items_df = pd.DataFrame({'survey': exps, 'worker': workers, 'item_text': items, 'coded_response': responses,
                             'response_text': responses_text, 'options': options}, dtype = float)
    items_df.loc[:,'item_num'] = [str(i).zfill(2) for i in item_nums]
    items_df.loc[:,'item_ID'] = items_df['survey'] + '.' + items_df['item_num'].astype(str)
    items_df=items_df[['worker','item_ID','coded_response','item_text','response_text','options','survey','item_num']]
    return items_df
    
    
def get_pay(data):
    assert 'ontask_time' in data.columns, \
        'Task time not found. Must run "calc_time_taken" first.' 
    all_exps = data.experiment_exp_id.unique()
    exps_completed = data.groupby('worker_id').experiment_exp_id.unique()
    exps_not_completed = exps_completed.map(lambda x: list(set(all_exps) - set(x) - set(['selection_optimization_compensation'])))
    completed = exps_completed[exps_completed.map(lambda x: len(x)>=63)]
    almost_completed = exps_not_completed[exps_not_completed.map(lambda x: x == ['angling_risk_task_always_sunny'])]
    not_completed = exps_not_completed[exps_not_completed.map(lambda x: len(x)>0 and x != ['angling_risk_task_always_sunny'])]
    # remove stray completions
    not_completed.loc[[i for i in not_completed.index if 's0' not in i]]
    # calculate time taken
    task_time = data.groupby('experiment_exp_id').ontask_time.mean()/60+2 # +2 for generic instruction time
    time_spent = exps_completed.map(lambda x: np.sum([task_time[i] if task_time[i]==task_time[i] else 3 for i in x])/60)
    time_missed = exps_not_completed.map(lambda x: np.sum([task_time[i] if task_time[i]==task_time[i] else 3 for i in x])/60)
    # calculate pay
    completed_pay = pd.Series(data = 60, index = completed.index)
    prorate_pay = 60-time_missed[almost_completed.index]*6
    reduced_pay = time_spent[not_completed.index]*2 + np.floor(time_spent[not_completed.index])*2
    #remove anyone who was double counted
    reduced_pay.drop(list(completed_pay.index) + list(prorate_pay.index), inplace = True, errors = 'ignore')
    pay= pd.concat([completed_pay, reduced_pay,prorate_pay]).map(lambda x: round(x,1)).to_frame(name = 'base')
    pay['bonuses'] = get_bonuses(data)
    pay['total'] = pay.sum(axis = 1)
    return pay

def get_fmri_pay(data):
    assert 'ontask_time' in data.columns, \
        'Task time not found. Must run "calc_time_taken" first.' 
    all_exps = data.experiment_exp_id.unique()
    exps_completed = data.groupby('worker_id').experiment_exp_id.unique()
    exps_not_completed = exps_completed.map(lambda x: list(set(all_exps) - set(x) - set(['selection_optimization_compensation'])))
    completed = exps_completed[exps_completed.map(lambda x: len(x)>=63)]
    not_completed = exps_not_completed[exps_not_completed.map(lambda x: len(x)>0)]
    # calculate time taken
    # get time taken for each task from previous mturk sample
    time_path = os.path.join(get_info('base_directory'),'references','experiment_lengths.json')
    task_time = json.load(open(time_path))
    time_missed = exps_not_completed.map(lambda x: np.sum([task_time[i] if task_time[i] is not None else 3 for i in x])/60)
    # calculate pay
    completed_pay = pd.Series(data = 100, index = completed.index)
    prorate_pay = 100-time_missed[not_completed.index]*10
    #remove anyone who was double counted
    pay= pd.concat([completed_pay, prorate_pay]).map(lambda x: round(x,1)).to_frame(name = 'base')
    pay['bonuses'] = get_bonuses(data, 15, 10)
    pay['total'] = pay.sum(axis = 1)
    return pay

def get_worker_demographics(worker_id, data):
    df = data[(data['worker_id'] == worker_id) & (data['experiment_exp_id'] == 'demographics_survey')]
    if len(df) == 1:
        race = df.query('text == "What is your race?"')['response'].unique()
        hispanic = df.query('text == "Are you of Hispanic, Latino or Spanish origin?"')['response_text']
        sex = df.query('text == "What is your sex?"')['response_text']
        age = float(df.query('text == "How old are you?"')['response'])
        return {'age': age, 'sex': sex, 'race': race, 'hispanic': hispanic}
    else:
        return np.nan
    
def print_time(data, time_col = 'ontask_time'):
    '''Prints time taken for each experiment in minutes
    :param time_col: Dataframe column of time in seconds
    '''
    df = data.copy()    
    assert time_col in df, \
        '"%s" has not been calculated yet. Use calc_time_taken method' % (time_col)
    #drop rows where time can't be calculated
    df = df.dropna(subset = [time_col])
    time = (df.groupby('experiment_exp_id')[time_col].mean()/60.0).round(2)
    print(time)
    return time
                   
def quality_check(data):
    """
    Checks data to make sure each experiment passed some "gut check" measures
    Used to exclude data on individual tasks or whole subjects if they fail
    too many tasks.
    NOTE: This function has an issue such that it inappropriately evaluates
    stop signal tasks based on the number of missed responses. Rather than 
    changing the function (which would affect our samples which are already
    determined) I am leaving it, and introducing a quality check correction
    that will be performed after subjects are already rejected
    """
    start_time = time()
    rt_thresh_lookup = {
        'angling_risk_task_always_sunny': 0,
        'simple_reaction_time': 150    
    }
    acc_thresh_lookup = {
        'digit_span': 0,
        'hierarchical_rule': 0,
        'information_sampling_task': 0,
        'probabilistic_selection': 0,
        'ravens': 0,
        'shift_task': 0,
        'spatial_span': 0,
        'tower_of_london': 0
        
    }
    missed_thresh_lookup = {
        'information_sampling_task': 1,
        'go_nogo': 1,
        'tower_of_london': 2
    }
    
    response_thresh_lookup = {
        'angling_risk_task_always_sunny': np.nan,
        'columbia_card_task_cold': np.nan,
        'discount_titrate': np.nan,
        'digit_span': np.nan,
        'go_nogo': .98,
        'kirby': np.nan,
        'simple_reaction_time': np.nan,
        'spatial_span': np.nan,
    }
    
    templates = data.groupby('experiment_exp_id').experiment_template.unique()
    data.loc[:,'passed_QC'] = True
    for exp in data.experiment_exp_id.unique():
        try:
            if templates.loc[exp] == 'jspsych':
                print('Running QC on ' + exp)
                df = extract_experiment(data, exp)
                rt_thresh = rt_thresh_lookup.get(exp,200)
                acc_thresh = acc_thresh_lookup.get(exp,.6)
                missed_thresh = missed_thresh_lookup.get(exp,.25)
                response_thresh = response_thresh_lookup.get(exp,.95)
                
                # special cases...
                if exp == 'information_sampling_task':
                    df.groupby('worker_id').which_click_in_round.value_counts()
                    passed_response = df.groupby('worker_id').which_click_in_round.mean() > 2
                    passed_rt = pd.Series([True] * len(passed_response), index = passed_response.index)
                    passed_miss = pd.Series([True] * len(passed_response), index = passed_response.index)
                    passed_acc = pd.Series([True] * len(passed_response), index = passed_response.index)
                elif exp == 'go_nogo':
                    passed_rt = df.query('rt != -1').groupby('worker_id').rt.median() >= rt_thresh
                    passed_miss = df.groupby('worker_id').rt.agg(lambda x: np.mean(x == -1)) < missed_thresh
                    df.correct = pd.to_numeric(df.correct)
                    passed_acc = df.groupby('worker_id').correct.mean() >= acc_thresh
                    passed_response = np.logical_not(df.groupby('worker_id').key_press.agg(
                                                            lambda x: np.any(pd.value_counts(x) > pd.value_counts(x).sum()*response_thresh)))
                elif exp == 'psychological_refractory_period_two_choices':
                    passed_rt = (df.groupby('worker_id').median()[['choice1_rt','choice2_rt']] >= rt_thresh).all(axis = 1)
                    passed_acc = df.query('choice1_rt != -1').groupby('worker_id').choice1_correct.mean() >= acc_thresh
                    passed_miss = ((df.groupby('worker_id').choice1_rt.agg(lambda x: np.mean(x!=-1) >= missed_thresh)) \
                                        + (df.groupby('worker_id').choice2_rt.agg(lambda x: np.mean(x>-1) >= missed_thresh))) == 2
                    passed_response1 = np.logical_not(df.query('choice1_rt != -1').groupby('worker_id').choice1_key_press.agg(
                                                            lambda x: np.any(pd.value_counts(x) > pd.value_counts(x).sum()*response_thresh)))
                    passed_response2 = np.logical_not(df.query('choice2_rt != -1').groupby('worker_id').choice2_key_press.agg(
                                                            lambda x: np.any(pd.value_counts(x) > pd.value_counts(x).sum()*response_thresh)))
                    passed_response = np.logical_and(passed_response1,passed_response2)
                elif exp == 'ravens':
                    passed_rt = df.query('rt != -1').groupby('worker_id').rt.median() >= rt_thresh
                    passed_acc = df.query('rt != -1').groupby('worker_id').correct.mean() >= acc_thresh
                    passed_response = np.logical_not(df.groupby('worker_id').stim_response.agg(
                                                            lambda x: np.any(pd.value_counts(x) > pd.value_counts(x).sum()*response_thresh)))
                    passed_miss = pd.Series([True] * len(passed_rt), index = passed_rt.index)
                elif exp == 'tower_of_london':
                    passed_rt = df.groupby('worker_id').rt.median() >= rt_thresh
                    passed_acc = df.query('trial_id == "feedback"').groupby('worker_id').correct.mean() >= acc_thresh
                    # Labeling someone as "missing" too many problems if they don't make enough moves
                    passed_miss = (df.groupby(['worker_id','problem_id']).num_moves_made.max().reset_index().groupby('worker_id').mean() >= missed_thresh).num_moves_made
                    passed_response = pd.Series([True] * len(passed_rt), index = passed_rt.index)
                elif exp == 'two_stage_decision':
                    passed_rt = (df.groupby('worker_id').median()[['rt_first','rt_second']] >= rt_thresh).all(axis = 1)
                    passed_miss = df.groupby('worker_id').trial_id.agg(lambda x: np.mean(x == 'incomplete_trial')) < missed_thresh
                    passed_acc = pd.Series([True] * len(passed_rt), index = passed_rt.index)
                    passed_response = pd.Series([True] * len(passed_rt), index = passed_rt.index)
                    passed_response1 = np.logical_not(df.query('rt_first != -1').groupby('worker_id').key_press_first.agg(
                                                            lambda x: np.any(pd.value_counts(x) > pd.value_counts(x).sum()*response_thresh)))
                    passed_response2 = np.logical_not(df.query('rt_second != -1').groupby('worker_id').key_press_second.agg(
                                                            lambda x: np.any(pd.value_counts(x) > pd.value_counts(x).sum()*response_thresh)))
                    passed_response = np.logical_and(passed_response1,passed_response2)
                elif exp == 'writing_task':
                    passed_response = df.query('trial_id == "write"').groupby('worker_id').final_text.agg(lambda x: len(x[0]) > 100)
                    passed_acc = pd.Series([True] * len(passed_response), index = passed_response.index)
                    passed_rt = pd.Series([True] * len(passed_response), index = passed_response.index)
                    passed_miss = pd.Series([True] * len(passed_response), index = passed_response.index)
                # everything else
                else:
                    passed_rt = df.query('rt != -1').groupby('worker_id').rt.median() >= rt_thresh
                    passed_miss = df.groupby('worker_id').rt.agg(lambda x: np.mean(x == -1)) < missed_thresh
                    if 'correct' in df.columns:
                        df.correct = pd.to_numeric(df.correct)
                        passed_acc = df.query('rt != -1').groupby('worker_id').correct.mean() >= acc_thresh
                    else:
                        passed_acc = pd.Series([True] * len(passed_rt), index = passed_rt.index)
                    if 'mouse_click' in df.columns:
                        passed_response = np.logical_not(df.query('rt != -1').groupby('worker_id').mouse_click.agg(
                                                            lambda x: np.any(pd.value_counts(x) > pd.value_counts(x).sum()*response_thresh)))
                    elif 'key_press' in df.columns:
                        passed_response = np.logical_not(df.query('rt != -1').groupby('worker_id').key_press.agg(
                                                            lambda x: np.any(pd.value_counts(x) > pd.value_counts(x).sum()*response_thresh)))   
                                                            
                passed_df = pd.concat([passed_rt,passed_acc,passed_miss,passed_response], axis = 1).fillna(False, inplace = False)
                passed = passed_df.all(axis = 1)
                failed = passed[passed == False]
                for subj in failed.index:
                    data.loc[(data.experiment_exp_id == exp) & (data.worker_id == subj),'passed_QC'] = False
        except AttributeError as e:
            print('QC could not be run on experiment %s' % exp)
            print(e)
    finish_time = (time() - start_time)/60
    print('Finished QC. Time taken: ' + str(finish_time))

def quality_check_correction(data):
    """
    This function corrects the issues with the stop signal tasks mentioned above
    """
    for exp in ['stop_signal','motor_selective_stop_signal',
                'stim_selective_stop_signal']:
        df = extract_experiment(data, exp)
        rt_thresh = 200
        acc_thresh = .6
        missed_thresh = .25
        response_thresh = .95
        passed_rt = df.query('rt != -1 and SS_trial_type=="go"').groupby('worker_id').rt.median() >= rt_thresh
        passed_miss = df.query('SS_trial_type=="go"').groupby('worker_id').rt.agg(lambda x: np.mean(x == -1)) < missed_thresh
        passed_acc = df.query('rt != -1').groupby('worker_id').correct.mean() >= acc_thresh
        passed_response = np.logical_not(df.query('rt != -1').groupby('worker_id').key_press.agg(
                                                lambda x: np.any(pd.value_counts(x) > pd.value_counts(x).sum()*response_thresh))) 
        passed_df = pd.concat([passed_rt,passed_acc,passed_miss,passed_response], axis = 1).fillna(False, inplace = False)
        passed = passed_df.all(axis = 1)
        failed = passed[passed == False]
        for subj in failed.index:
            data.loc[(data.experiment_exp_id == exp) & (data.worker_id == subj),'passed_QC'] = False
        for subj in passed.index:
            data.loc[(data.experiment_exp_id == exp) & (data.worker_id == subj),'passed_QC'] = True
    
def remove_failed_subjects(data):
    if 'passed_QC' not in data.columns:
        quality_check(data)
    failed_workers = data.groupby('worker_id').passed_QC.sum() < 60
    failed_workers = list(failed_workers[failed_workers].index)
    # drop workers
    failed_data = data[data['worker_id'].isin(failed_workers)]
    data.drop(failed_data.index, inplace = True)
    return failed_data

def remove_correlated_task_variables(data, threshold=.85):
    tasks = np.unique([i.split('.')[0] for i in data.columns])
    columns_to_remove = []
    for task in tasks:
        task_data = data.filter(regex = '^%s' % task)
        corr_mat = task_data.corr().replace({1:0})
        i=0
        while True:
            kept_indices = np.where(abs(corr_mat.iloc[:,i])<threshold)[0]
            corr_mat = corr_mat.iloc[kept_indices,kept_indices]
            i+=1
            if i>=corr_mat.shape[0]:
                break
        columns_to_remove += list(set(task_data.columns)-set(corr_mat.columns))
    print( '*' * 50)
    print('Dropping %s variables with correlations above %s' % (len(columns_to_remove), threshold))
    print( '*' * 50)
    print('\n'.join(columns_to_remove))
    data = drop_vars(data,columns_to_remove)
    return data
    
def remove_outliers(data, quantile_range = 2.5):
    '''Removes outliers more than 1.5IQR below Q1 or above Q3
    '''
    data = data.copy()
    quantiles = data.apply(lambda x: x.dropna().quantile([.25,.5,.75])).T
    lowlimit = np.array(quantiles.iloc[:,1] - quantile_range*(quantiles.iloc[:,2] - quantiles.iloc[:,0]))
    highlimit = np.array(quantiles.iloc[:,1] + quantile_range*(quantiles.iloc[:,2] - quantiles.iloc[:,0]))
    data_mat = data.values
    data_mat[np.logical_or((data_mat<lowlimit), (data_mat>highlimit))] = np.nan
    data = pd.DataFrame(data=data_mat, index=data.index, columns=data.columns)
    return data
    
def save_task_data(data_loc, data):
    path = os.path.join(data_loc,'Individual_Measures')
    if not os.path.exists(path):
        os.makedirs(path)
    for exp_id in np.sort(data.experiment_exp_id.unique()):
        print('Saving %s...' % exp_id)
        extract_experiment(data,exp_id).to_csv(os.path.join(path, exp_id + '.csv.gz'), compression = 'gzip')

def transform_remove_skew(data, threshold=1, 
                          positive_skewed=None,
                          negative_skewed=None,
                          drop_failed=True,
                          verbose=True):
    data = data.copy()
    if positive_skewed is None:
        positive_skewed = data.skew()>threshold
    if negative_skewed is None:
        negative_skewed = data.skew()<-threshold
    positive_subset = data.loc[:,positive_skewed]
    negative_subset = data.loc[:,negative_skewed]
    # transform variables
    # log transform for positive skew
    shift = pd.Series(0, index=positive_subset.columns)
    shift_variables = positive_subset.min()<=0
    shift[shift_variables] -= (positive_subset.min()[shift_variables]-1)
    positive_subset = np.log(positive_subset+shift)
    # remove outliers
    positive_tmp = remove_outliers(positive_subset)
    if drop_failed:
        successful_transforms = positive_subset.loc[:,abs(positive_tmp.skew())<threshold]
    else:
        successful_transforms = positive_subset
    if verbose:
        print('*'*40)
        print('** Successfully transformed %s positively skewed variables:' % len(successful_transforms.columns))
        print('\n'.join(successful_transforms.columns))
        print('*'*40)
    dropped_vars = set(positive_subset)-set(successful_transforms)
    # replace transformed variables
    data.drop(positive_subset, axis=1, inplace = True)
    successful_transforms.columns = [i + '.logTr' for i in successful_transforms]
    if verbose:
        print('*'*40)
        print('Dropping %s positively skewed data that could not be transformed successfully:' % len(dropped_vars))
        print('\n'.join(dropped_vars))
        print('*'*40)
    data = pd.concat([data, successful_transforms], axis = 1)
    # reflected log transform for negative skew
    negative_subset = np.log(negative_subset.max()+1-negative_subset)
    negative_tmp = remove_outliers(negative_subset)
    if drop_failed:
        successful_transforms = negative_subset.loc[:,abs(negative_tmp.skew())<threshold]
    else:
        successful_transforms = negative_subset
    if verbose:
        print('*'*40)
        print('** Successfully transformed %s negatively skewed variables:' % len(successful_transforms.columns))
        print('\n'.join(successful_transforms.columns))
        print('*'*40)
    dropped_vars = set(negative_subset)-set(successful_transforms)
    # replace transformed variables
    data.drop(negative_subset, axis=1, inplace = True)
    successful_transforms.columns = [i + '.ReflogTr' for i in successful_transforms]
    if verbose:
        print('*'*40)
        print('Dropping %s negatively skewed data that could not be transformed successfully:' % len(dropped_vars))
        print('\n'.join(dropped_vars))
        print('*'*40)
    data = pd.concat([data, successful_transforms], axis=1)
    return data.sort_index(axis = 1)
