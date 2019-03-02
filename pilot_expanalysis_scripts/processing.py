"""
analysis/processing.py: part of expfactory package
functions for automatically cleaning and manipulating experiments by operating
on an expanalysis Result.data dataframe
"""
from expanalysis.experiments.jspsych_processing import adaptive_nback_post, ANT_post, ART_post, \
    CCT_hot_post, choice_reaction_time_post, cognitive_reflection_post, dietary_decision_post, directed_forgetting_post, \
    DPX_post, hierarchical_post, IST_post, keep_track_post, local_global_post, \
    probabilistic_selection_post, PRP_post, recent_probes_post,  shift_post, span_post, \
    stop_signal_post, TOL_post, threebytwo_post, two_stage_decision_post, \
    calc_adaptive_n_back_DV, calc_ANT_DV, calc_ART_sunny_DV, calc_CCT_cold_DV, \
    calc_CCT_hot_DV, calc_choice_reaction_time_DV, calc_cognitive_reflection_DV, \
    calc_dietary_decision_DV, calc_digit_span_DV, calc_directed_forgetting_DV, calc_DPX_DV,\
    calc_go_nogo_DV, calc_hierarchical_rule_DV, calc_IST_DV, calc_keep_track_DV, \
    calc_local_global_DV, calc_probabilistic_selection_DV, calc_recent_probes_DV, \
    calc_ravens_DV, calc_shape_matching_DV, calc_shift_DV, \
    calc_simon_DV, calc_simple_RT_DV, calc_spatial_span_DV, calc_stop_signal_DV, \
    calc_stroop_DV, calc_threebytwo_DV, calc_TOL_DV, calc_two_stage_decision_DV
from expanalysis.experiments.survey_processing import \
    calc_bis11_DV, calc_bis_bas_DV, calc_brief_DV, calc_dickman_DV, \
    calc_dospert_DV, calc_eating_DV, calc_erq_DV, calc_five_facet_mindfulness_DV, \
    calc_future_time_perspective_DV, calc_grit_DV, calc_i7_DV, \
    calc_leisure_time_DV, calc_maas_DV, calc_mpq_control_DV, \
    calc_SOC_DV, calc_SSRQ_DV, calc_SSS_DV, calc_ten_item_personality_DV, \
    calc_theories_of_willpower_DV, calc_time_perspective_DV, calc_upps_DV
from expanalysis.experiments.utils import get_data, lookup_val, select_experiment, drop_null_cols
import pandas
import numpy
import os
import time

def clean_data(df, exp_id = None, apply_post = True, drop_columns = None, lookup = True):
    '''clean_df returns a pandas dataset after removing a set of default generic 
    columns. Optional variable drop_cols allows a different set of columns to be dropped
    :df: a pandas dataframe
    :param experiment: a string identifying the experiment used to automatically drop unnecessary columns. df should not have multiple experiments if this flag is set!
    :param apply_post: bool, if True apply post-processig function retrieved using apply_post
    :param drop_columns: a list of columns to drop. If not specified, a default list will be used from utils.get_dropped_columns()
    :param lookup: bool, default true. If True replaces all values in dataframe using the lookup_val function
    :param return_reject: bool, default false. If true returns a dataframe with rejected experiments
    '''
    if apply_post:
        # apply post processing 
        df = post_process_exp(df, exp_id)
    # Drop unnecessary columns
    if drop_columns == None:
        drop_columns = get_drop_columns()   
    df.drop(drop_columns, axis=1, inplace=True, errors='ignore')
    if exp_id != None:
        assert sum(df['experiment_exp_id'] == exp_id) == len(df), \
            "An experiment was specified, but the dataframe has other experiments!"      
        drop_rows = get_drop_rows(exp_id)
        # Drop unnecessary rows, all null rows
        for key in drop_rows.keys():
            df = df.query('%s not in  %s' % (key, drop_rows[key]))
    df = df.dropna(how = 'all')
    if lookup == True:
        #convert vals based on lookup
        for col in df.columns:
            df.loc[:,col] = df[col].map(lookup_val)
    #drop columns with only null values
    drop_null_cols(df)
    return df



def get_drop_columns():
    return ['view_history', 'stimulus', 'trial_index', 'internal_node_id', 
           'stim_duration', 'block_duration', 'feedback_duration','timing_post_trial', 'exp_id']
           
def get_drop_rows(exp_id):
    '''Function used by clean_df to drop rows from dataframes with one experiment
    :experiment: experiment key used to look up which rows to drop from a dataframe
    '''
    gen_cols = ['welcome', 'text','instruction', 'attention_check','end', 'post task questions', 'fixation', \
                'practice_intro', 'test_intro'] #generic_columns to drop
    lookup = {'adaptive_n_back': {'trial_id': gen_cols + ['update_target', 'update_delay', 'delay_text']},
                'angling_risk_task_always_sunny': {'trial_id': gen_cols + ['test_intro','intro','ask_fish','set_fish']}, 
                'attention_network_task': {'trial_id': gen_cols + ['spatialcue', 'centercue', 'doublecue', 'nocue', 'rest block', 'intro']}, 
                'bickel_titrator': {'trial_id': gen_cols + ['update_delay', 'update_mag', 'gap']}, 
                'choice_reaction_time': {'trial_id': gen_cols + ['practice_intro', 'reset trial']}, 
                'columbia_card_task_cold': {'trial_id': gen_cols + ['calculate_reward','reward','end_instructions']}, 
                'columbia_card_task_hot': {'trial_id': gen_cols + ['calculate_reward', 'reward', 'test_intro']}, 
                'dietary_decision': {'trial_id': gen_cols + ['start_taste', 'start_health']}, 
                'digit_span': {'trial_id': gen_cols + ['start_reverse', 'stim', 'feedback']},
                'directed_forgetting': {'trial_id': gen_cols + ['ITI_fixation', 'intro_test', 'stim', 'cue']},
                'dot_pattern_expectancy': {'trial_id': gen_cols + ['rest', 'cue', 'feedback']},
                'go_nogo': {'trial_id': gen_cols + ['reset_trial']},
                'hierarchical_rule': {'trial_id': gen_cols + ['feedback', 'test_intro']},
                'information_sampling_task': {'trial_id': gen_cols + ['DW_intro', 'reset_round']},
                'keep_track': {'trial_id': gen_cols + ['practice_end', 'stim', 'wait', 'prompt']},
                'kirby': {'trial_id': gen_cols + ['prompt', 'wait']},
                'local_global_letter': {'trial_id': gen_cols + []},
                'motor_selective_stop_signal': {'trial_id': gen_cols + ['prompt_fixation', 'feedback']},
                'probabilistic_selection': {'trial_id': gen_cols + ['first_phase_intro', 'second_phase_intro']},
                'psychological_refractory_period_two_choices': {'trial_id': gen_cols + ['feedback']},
                'recent_probes': {'trial_id': gen_cols + ['intro_test', 'ITI_fixation', 'stim']},
                'shift_task': {'trial_id': gen_cols + ['rest', 'alert', 'feedback']},
                'simple_reaction_time': {'trial_id': gen_cols + ['reset_trial']},
                'spatial_span': {'trial_id': gen_cols + ['start_reverse_intro', 'stim', 'feedback']},
                'stim_selective_stop_signal': {'trial_id': gen_cols + ['feedback']},
                'stop_signal': {'trial_id': gen_cols + ['reset', 'feedback']},
                'stroop': {'trial_id': gen_cols + []}, 
                'simon':{'trial_id': gen_cols + ['reset_trial']}, 
                'threebytwo': {'trial_id': gen_cols + ['cue', 'gap', 'set_stims']},
                'tower_of_london': {'trial_id': gen_cols + ['feedback', 'advance', 'practice']},
                'two_stage_decision': {'trial_id': gen_cols + ['wait', 'first_stage_selected', 'second_stage_selected', 'wait_update_fb', 'wait_update_FB','change_phase']},
                'willingness_to_wait': {'trial_id': gen_cols + []},
                'writing_task': {}}    
    to_drop = lookup.get(exp_id, {})
    return to_drop

def post_process_exp(df, exp_id):
    '''Function used to post-process a dataframe extracted via extract_row or extract_experiment
    :exp_id: experiment key used to look up appropriate grouping variables
    '''
    lookup = {'adaptive_n_back': adaptive_nback_post,
              'angling_risk_task': ART_post,
              'angling_risk_task_always_sunny': ART_post,
              'attention_network_task': ANT_post,
              'choice_reaction_time': choice_reaction_time_post,
              'cognitive_reflection_survey': cognitive_reflection_post,
              'columbia_card_task_hot': CCT_hot_post,
              'dietary_decision': dietary_decision_post,
              'digit_span': span_post,
              'directed_forgetting': directed_forgetting_post,
              'dot_pattern_expectancy': DPX_post,
              'hierarchical_rule': hierarchical_post,
              'information_sampling_task': IST_post,
              'keep_track': keep_track_post,
              'local_global_letter': local_global_post,
              'motor_selective_stop_signal': stop_signal_post,
              'probabilistic_selection': probabilistic_selection_post,
              'psychological_refractory_period_two_choices': PRP_post,
              'recent_probes': recent_probes_post,
              'shift_task': shift_post,
              'spatial_span': span_post,
              'stim_selective_stop_signal': stop_signal_post,
              'stop_signal': stop_signal_post,
              'tower_of_london': TOL_post,
              'threebytwo': threebytwo_post,
              'two_stage_decision': two_stage_decision_post}     
                
    fun = lookup.get(exp_id, lambda df: df)
    return fun(df).sort_index(axis = 1)

def post_process_data(data):
    """ applies post_process_exp to an entire dataset
    """
    time_taken = {}
    post_processed = []
    for i,row in data.iterrows():
        if (i%100 == 0):
            print i
        exp_id = row['experiment_exp_id']
        df = extract_row(row, clean = False)
        tic = time.time()
        df = post_process_exp(df,exp_id)
        toc = time.time() - tic
        time_taken.setdefault(exp_id,[]).append(toc)
        
        post_processed.append(df.to_dict())
    for key in time_taken.keys():
        time_taken[key] = numpy.mean(time_taken[key])
    print time_taken
    data.loc[:,'data'] = post_processed
    data.loc[:,'process_stage'] = 'post'
    
def extract_row(row, clean = True, apply_post = True, drop_columns = None):
    '''Returns a dataframe that has expanded the data of one row of a results object
    :row:  one row of a Results data dataframe
    :param clean: boolean, if true call clean_df on the data
    :param drop_columns: list of columns to pass to clean_df
    :param drop_na: boolean to pass to clean_df
    :return df: dataframe containing the extracted experiment
    '''
    exp_id = row['experiment_exp_id']
    if row.get('process_stage') == 'post':
        df = pandas.DataFrame(row['data'])
        df['sort']=[int(i.split('_')[-1]) for i in df.index]
        df.sort_values(by = 'sort',inplace = True)
        df.drop('sort',axis = 1, inplace = True)
        if clean == True:
            df = clean_data(df, row['experiment_exp_id'], False, drop_columns)
    else:
        exp_data = get_data(row)
        for trial in exp_data:
            trial['battery_name'] = row['battery_name']
            trial['experiment_exp_id'] = row['experiment_exp_id']
            trial['worker_id'] = row['worker_id']
            trial['finishtime'] = row['finishtime']
        df = pandas.DataFrame(exp_data)
        trial_index = ["%s_%s" % (exp_id,x) for x in range(len(exp_data))]
        df.index = trial_index
        if clean == True:
            df = clean_data(df, row['experiment_exp_id'], apply_post, drop_columns)
    return df  

def extract_experiment(data, exp_id, clean = True, apply_post = True, drop_columns = None, return_reject = False, clean_fun = clean_data):
    '''Returns a dataframe that has expanded the data column of the results object for the specified experiment.
    Each row of this new dataframe is a data row for the specified experiment.
    :data: the data from an expanalysis Result object
    :experiment: a string identifying one experiment
    :param clean: boolean, if true call clean_df on the data
    :param drop_columns: list of columns to pass to clean_df
    :param return_reject: bool, default false. If true returns a dataframe with rejected experiments
    :param clean_fun: an alternative "clean" function. Must return a dataframe of the cleaned data
    :return df: dataframe containing the extracted experiment
    '''
    trial_index = []
    df = select_experiment(data, exp_id)
    if 'flagged' in df.columns:
        df_reject = df.query('flagged == True')
        df = df.query('flagged == False')
        if len(df) == 0:
            print('All %s datasets were flagged')
            return df,df_reject
    #ensure there is only one dataset for each battery/experiment/worker combination
    assert sum(df.groupby(['battery_name', 'experiment_exp_id', 'worker_id']).size()>1)==0, \
        "More than one dataset found for at least one battery/experiment/worker combination"
    if numpy.unique(df.get('process_stage'))=='post':
        group_df = pandas.DataFrame()
        for i,row in df.iterrows():
            tmp_df = extract_row(row, clean, False, drop_columns)
            group_df = pandas.concat([group_df, tmp_df ])
            insert_i = tmp_df.index[0].rfind('_')
            trial_index += [x[:insert_i] + '_%s' % i + x[insert_i:] for x in tmp_df.index]
        df = group_df
        df.index = trial_index
        #sort_df
        df['sort']=[(int(i.split('_')[-2]),int(i.split('_')[-1])) for i in df.index]
        df.sort_values(by = 'sort',inplace = True)
        df.drop('sort',axis = 1, inplace = True)
    else:
        trial_list = []
        for i,row in df.iterrows():
            exp_data = get_data(row)
            for trial in exp_data:
                trial['battery_name'] = row['battery_name']
                trial['experiment_exp_id'] = row['experiment_exp_id']
                trial['worker_id'] = row['worker_id']
                trial['finishtime'] = row['finishtime']
            trial_list += exp_data
            trial_index += ["%s_%s_%s" % (exp_id,i,x) for x in range(len(exp_data))]
        df = pandas.DataFrame(trial_list)
        df.index = trial_index
        if clean == True:
            df = clean_fun(df, exp_id, apply_post, drop_columns)
    if return_reject:
        return df, df_reject
    else:
        return df

def export_experiment(filey, data, exp_id, clean = True):
    """ Exports data from one experiment to path specified by filey. Must be .csv, .pkl or .json
    :filey: path to export data
    :data: the data from an expanalysis Result object
    :experiment: experiment to export
    :param clean: boolean, default True. If true cleans the experiment df before export
    """
    df = extract_experiment(data, exp_id, clean)
    file_name,ext = os.path.splitext(filey)
    if ext.lower() == ".csv":
        df.to_csv(filey)
    elif ext.lower() == ".pkl":
        df.to_pickle(filey)
    elif ext.lower() == ".json":
        df.to_json(filey)
    else:
        print "File extension not recognized, must be .csv, .pkl, or .json." 

    
def get_DV(data, exp_id, use_check = True):
    '''Function used by clean_df to post-process dataframe
    :experiment: experiment key used to look up appropriate grouping variables
    :param use_check: bool, if True exclude dataframes that have "False" in a 
    passed_check column, if it exists. Passed_check would be defined by a post_process
    function specific to that experiment
    '''
    lookup = {'adaptive_n_back': calc_adaptive_n_back_DV,
              'angling_risk_task_always_sunny': calc_ART_sunny_DV,
              'attention_network_task': calc_ANT_DV,
              'bis11_survey': calc_bis11_DV,
              'bis_bas_survey': calc_bis_bas_DV,
              'brief_self_control_survey': calc_brief_DV,
              'choice_reaction_time': calc_choice_reaction_time_DV,
              'columbia_card_task_cold': calc_CCT_cold_DV,
              'columbia_card_task_hot': calc_CCT_hot_DV,
              'cognitive_reflection_survey': calc_cognitive_reflection_DV,
              'dietary_decision': calc_dietary_decision_DV,
              'dickman_survey': calc_dickman_DV,
              'digit_span': calc_digit_span_DV,
              'directed_forgetting': calc_directed_forgetting_DV,
              'dospert_eb_survey': calc_dospert_DV,
              'dospert_rp_survey': calc_dospert_DV,
              'dospert_rt_survey': calc_dospert_DV,
              'dot_pattern_expectancy': calc_DPX_DV,
              'eating_survey': calc_eating_DV,
              'five_facet_mindfulness_survey': calc_five_facet_mindfulness_DV,
              'future_time_perspective_survey': calc_future_time_perspective_DV,
              'go_nogo': calc_go_nogo_DV,
              'grit_scale_survey': calc_grit_DV,
              'hierarchical_rule': calc_hierarchical_rule_DV,
              'impulsive_venture_survey': calc_i7_DV,
              'information_sampling_task': calc_IST_DV,
              'keep_track': calc_keep_track_DV,
              'leisure_time_activity_survey': calc_leisure_time_DV,
              'local_global_letter': calc_local_global_DV,
              'mindful_attention_awareness_survey': calc_maas_DV,
              'mpq_control_survey': calc_mpq_control_DV,
              'probabilistic_selection': calc_probabilistic_selection_DV,
              'ravens': calc_ravens_DV,
              'recent_probes': calc_recent_probes_DV,
              'selection_optimization_compensation_survey': calc_SOC_DV,
              'self_regulation_survey': calc_SSRQ_DV,
              'sensation_seeking_survey': calc_SSS_DV,
              'simon': calc_simon_DV,
              'simple_reaction_time': calc_simple_RT_DV,
              'shape_matching_task': calc_shape_matching_DV,
              'shift_task': calc_shift_DV,
              'spatial_span': calc_spatial_span_DV,
              'stop_signal': calc_stop_signal_DV,
              'stroop': calc_stroop_DV,
              'ten_item_personality_survey': calc_ten_item_personality_DV,
              'theories_of_willpower': calc_theories_of_willpower_DV,
              'time_perspective_survey': calc_time_perspective_DV,
              'threebytwo': calc_threebytwo_DV,
              'tower_of_london': calc_TOL_DV,
              'two_stage_decision': calc_two_stage_decision_DV,
              'upps_impulsivity_survey': calc_upps_DV}   
    fun = lookup.get(exp_id, None)
    if fun:
        df = extract_experiment(data,exp_id)
        return fun(df, use_check)
    else:
        return {},''
    
    
def calc_DVs(data, use_check = True):
    """Calculate DVs for each experiment
    :data: the data dataframe of a expfactory Result object
    :param use_check: bool, if True exclude dataframes that have "False" in a 
    passed_check column, if it exists. Passed_check would be defined by a post_process
    function specific to that experiment
    """
    data.loc[:,'DV_val'] = numpy.nan
    data.loc[:,'DV_val'] = data['DV_val'].astype(object)
    data.loc[:,'DV_description'] = ''
    for exp_id in numpy.unique(data['experiment_exp_id']):
        tic = time.time()
        subset = data[data['experiment_exp_id'] == exp_id]
        dvs, description = get_DV(subset,exp_id, use_check) 
        subset = subset.query('worker_id in %s' % dvs.keys())
        if len(dvs) == len(subset):
            data.loc[subset.index,'DV_val'] = dvs.values()
            data.loc[subset.index,'DV_description'] = description
        toc = time.time() - tic
        print exp_id, ':', toc
        
def extract_DVs(data, use_check = True):
    """Calculate if necessary and extract DVs into a new dataframe where rows
    are workers and columns are DVs
    :data: the data dataframe of a expfactory Result object
    :param use_check: bool, if True exclude dataframes that have "False" in a 
    passed_check column, if it exists. Passed_check would be defined by a post_process
    function specific to that experiment
    """
    if not 'DV_val' in data.columns:
        calc_DVs(data, use_check)
    data = data[data['DV_val'].isnull()==False]
    DV_list = []
    for worker in numpy.unique(data['worker_id']):
        DV_dict = {'worker_id': worker}
        subset = data.query('worker_id == "%s"' % worker)
        for i,row in subset.iterrows():
            DVs = row['DV_val'].copy()
            exp_id = row['experiment_exp_id']
            for key in DVs:
                DV_dict[exp_id +'.' + key] = DVs[key]
        DV_list.append(DV_dict)
    df = pandas.DataFrame(DV_list) 
    df.set_index('worker_id' ,inplace = True)
    return df

def generate_reference(data, file_base):
    """ Takes a results data frame and returns an experiment dictionary with
    the columsn and column types for each experiment (after apply post_processing)
    :data: the data dataframe of a expfactory Result object
    :file_base:
    """
    exp_dic = {}
    for exp_id in numpy.unique(data['experiment_exp_id']):
        exp_dic[exp_id] = {}
        df = extract_experiment(data,exp_id, clean = False)
        col_types = df.dtypes
        exp_dic[exp_id] = col_types
    pandas.to_pickle(exp_dic, file_base + '.pkl')
    
def flag_data(data, reference_file):
    """ function to flag data for rejections by checking the columns of the 
    extracted data against a reference file generated by generate_reference.py
    :data: the data dataframe of a expfactory Result object
    :exp_id: an expfactory exp_id corresponding to the df
    :save_post_process: bool, if True replaces the data in each row with post processed data
    :reference file: a pickle follow created by generate_reference
    
    """
    flagged = []
    lookup_dic = pandas.read_pickle(reference_file)
    for i,row in data.iterrows():
        exp_id = row['experiment_exp_id']
        df = extract_row(row, clean = False)
        col_types = df.dtypes
        lookup = lookup_dic[exp_id]
        #drop unimportant cols which are sometimes not recorded
        for col in ['trial_num', 'responses', 'credit_var', 'performance_var']:
            if col in lookup:
                lookup.drop(col, inplace = True)
            if col in col_types:
                col_types.drop(col, inplace = True)
        flag = not ((col_types.index.tolist() == lookup.index.tolist()) and \
            (col_types.tolist() == lookup.tolist()))
        flagged.append(flag)
    data['flagged'] = flagged
