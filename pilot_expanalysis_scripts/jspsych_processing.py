"""
analysis/experiments/jspsych_processing.py: part of expfactory package
functions for automatically cleaning and manipulating jspsych experiments
"""
import re
import pandas
import numpy
import hddm
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import zscore
import json
from math import factorial

"""
Generic Functions
"""

def EZ_diffusion(df):
    assert 'correct' in df.columns, 'Could not calculate EZ DDM'
    pc = df['correct'].mean()
    vrt = numpy.var(df.query('correct == True')['rt'])
    mrt = numpy.mean(df.query('correct == True')['rt'])
    drift, thresh, non_dec = hddm.utils.EZ(pc, vrt, mrt)
    return {'EZ_drift': drift, 'EZ_thresh': thresh, 'EZ_non_decision': non_dec}
 
def multi_worker_decorate(func):
    """Decorator to ensure that dv functions have only one worker
    """
    def multi_worker_wrap(group_df, use_check = True):
        group_dvs = {}
        if len(group_df) == 0:
            return group_dvs, ''
        if 'passed_check' in group_df.columns and use_check:
            group_df = group_df[group_df['passed_check']]
        for worker in pandas.unique(group_df['worker_id']):
            df = group_df.query('worker_id == "%s"' %worker)
            try:
                group_dvs[worker], description = func(df)
            except:
                print('DV calculated failed for worker: %s' % worker)
        return group_dvs, description
    return multi_worker_wrap

def calc_common_stats(df):
    dvs = {}
    dvs['avg_rt'] = df['rt'].median()
    dvs['std_rt'] = df['rt'].std()
    if 'correct' in df.columns:
        dvs['overall_accuracy'] = df['correct'].mean()
        if 'possible_responses' in df.columns:
            df = df.query('possible_responses == possible_responses')
            possible_responses = numpy.unique(df['possible_responses'].map(sorted))
            if (len(possible_responses) == 1 and \
                len(possible_responses[0]) == 2 ):
                try:
                    diffusion_params = EZ_diffusion(df)
                    dvs.update(diffusion_params)
                except ValueError:
                    pass
    return dvs

    
"""
Post Processing functions
"""

def adaptive_nback_post(df):
    if df.query('trial_id == "stim"').iloc[0]['possible_responses'] == [37,40]:
        response_dict = {True: 37, False: 40}
    else:
        response_dict = {True: 37, False: -1}
    if 'correct_response' not in df.columns:
        df.loc[:,'correct_response'] = numpy.nan
    nan_index = df.query('target == target and correct_response != correct_response').index
    hits = df.loc[nan_index, 'stim'].str.lower() == df.loc[nan_index,'target'].str.lower()
    df.loc[nan_index,'correct_response'] = hits.map(lambda x: response_dict[x])
    df.loc[nan_index,'correct'] = df.loc[nan_index,'correct_response'] == df.loc[nan_index,'key_press']
    df.loc[:,'correct'] = df['correct'].map(lambda x: float(x) if x==x else numpy.nan)
    if 'feedback_duration' in df.columns:    
        df.drop('feedback_duration', axis = 1, inplace = True)
    df.loc[:,'correct'] = df['correct'].map(lambda x: float(x) if x==x else numpy.nan)
    if 'block_num' not in df.columns:
        subset = df[(df['trial_id'].apply(lambda x: x in ['delay_text', 'stim'])) & (df['exp_stage'] != "control")][1:]
        block_num = 0
        block_nums = []
        for row in subset['trial_id']:
            block_nums.append(block_num)
            if row == 'delay_text':
                block_num += 1
            if block_num == 20:
                block_num = 0
        df.loc[subset.index,'block_num'] = block_nums
    return df
    
    
def ANT_post(df):
    correct = df['correct_response'] == df['key_press']
    if 'correct' in df.columns:
        df.loc[:,'correct'] = correct
    else:
        df.insert(0,'correct',correct)
    df.loc[:,'correct'] = df['correct'].map(lambda x: float(x) if x==x else numpy.nan)
    df=df.dropna(subset = ['possible_responses'])
    return df
    
def ART_post(df):
    round_over_list = df.query('trial_id == "round_over"').index
    if 'caught_blue' not in df.columns:
        df.loc[:,'caught_blue'] = numpy.nan
        df['caught_blue'] = df['caught_blue'].astype(object)
    for i in round_over_list:
        if pandas.isnull(df.loc[i]['caught_blue']):
            index = df.index.get_loc(i)
            caught_blue = df.iloc[index-1]['mouse_click'] == 'goFish'
            df.set_value(i,'caught_blue', caught_blue)
        if pandas.isnull(df.loc[i]['weather']):
            index = df.index.get_loc(i)
            weather = df.iloc[index-1]['weather']
            release = df.iloc[index-1]['release']
            df.set_value(i,'weather', weather)
            df.set_value(i,'release', release)
    df.loc[:,'caught_blue'] = df['caught_blue'].map(lambda x: float(x) if x==x else numpy.nan)
    return df

def CCT_hot_post(df):
    if 'whichButtonWasClicked' in df.columns:
        df = df.drop('whichButtonWasClicked', axis = 1)
    subset = df[df['mouse_click'] == "collectButton"]
    def getNumRounds(a,b):
        return a-1 if b else a-2
    total_cards = subset.apply(lambda row: getNumRounds(row['num_click_in_round'], row['clicked_on_loss_card']), axis = 1)
    df.insert(0,'total_cards', total_cards)
    df.loc[:,'clicked_on_loss_card'] = df['clicked_on_loss_card'].astype(float) 
    return df
    
def choice_reaction_time_post(df):
    for worker in numpy.unique(df['worker_id']):
        subset = df.query('worker_id == "%s" and exp_stage == "practice"' %worker)
        response_dict = subset.groupby('stim_id')['correct_response'].mean().to_dict()
        test_index = df.query('exp_stage == "test"').index      
        df.loc[test_index, 'correct_response'] = df.loc[test_index,'stim_id'].map(lambda x: response_dict[x] if x == x else numpy.nan)
    correct = df['correct_response'] == df['key_press']
    if 'correct' in df.columns:
        df.loc[:,'correct'] = correct
    else:
        df.loc[:,'correct'] = correct
    df.loc[:,'correct'] = df['correct'].map(lambda x: float(x) if x==x else numpy.nan)
    return df
       
def cognitive_reflection_post(df):
    correct_responses = ['3', '15', '4', '29', '20', 'c'] * (len(df)/6)
    intuitive_responses = ['6', '20', '9', '30', '10', 'b'] * (len(df)/6)
    df.loc[:,'correct_response'] = correct_responses
    df.loc[:,'intuitive_response'] = intuitive_responses
    df.loc[:,'correct'] = df['correct_response'] == df['response']
    df.loc[:,'correct'] = df['correct'].map(lambda x: float(x) if x==x else numpy.nan)
    df.loc[:,'responded_intuitively'] = df['intuitive_response'] == df['response']
    df.loc[:,'responded_intuitively'] = df['responded_intuitively'].map(lambda x: float(x) if x==x else numpy.nan)    
    return df
    
def dietary_decision_post(df):
    df.loc[df['mouse_click'] == 'Strong_Yes','coded_response'] = 2
    df['stim_rating'] = df['stim_rating'].apply(lambda x: json.loads(x) if x==x else numpy.nan)
    df['reference_rating'] = df['reference_rating'].apply(lambda x: json.loads(x) if x==x else numpy.nan)
    # subset list to only decision trials where the item was rated on both health and taste
    group_subset = df[df['stim_rating'].apply(lambda lst: all(isinstance(x, int) for x in list(lst.values())) if lst == lst else False)]
    for finishtime in group_subset['finishtime']:
        subset = group_subset[group_subset['finishtime'] == finishtime]
        reference = numpy.unique(subset['reference_rating'])
        assert len(reference) == 1, "More than one reference rating found"
        reference = reference[0]
        subset.insert(0,'health_diff',subset['stim_rating'].apply(lambda x: x['health'] - reference['health']))
        subset.insert(0,'taste_diff', subset['stim_rating'].apply(lambda x: x['taste'] - reference['taste']))
        labels = []
        for i,row in subset.iterrows():
            if row['health_diff'] > 0:
                health_label = 'Healthy'
            elif row['health_diff'] < 0:
                health_label = 'Unhealthy'
            else:
                health_label = 'Neutral'
    
            if row['taste_diff'] > 0:
                taste_label = 'Liked'
            elif row['taste_diff'] < 0:
                taste_label = 'Disliked'
            else:
                taste_label = 'Neutral'
            labels.append(taste_label + '-' + health_label)
        subset.insert(0,'decision_label', labels)
        if 'decision_label' not in df.columns:
            df = df.join(subset[['health_diff','taste_diff','decision_label']])
        else:
            df.loc[subset.index, ['health_diff', 'taste_diff', 'decision_label']] = subset[['health_diff','taste_diff','decision_label']]
    df['coded_response'] = df['coded_response'].astype(float)
    return df
    
def directed_forgetting_post(df):
    if 'probeType' in df.columns:
        df['probe_type'] = df['probe_type'].fillna(df['probeType'])
        df.drop('probeType',axis = 1, inplace = True)
    if 'cue' not in df.columns:
        df.loc[:,'cue'] = numpy.nan
    if 'stim' in df.columns:
        df['cue'] = df['cue'].fillna(df['stim'])
        df.drop('stim',axis = 1, inplace = True)
    df['stim_bottom'] = df['stim_bottom'].fillna(df['stim_bottom'].shift(3))
    df['stim_top'] = df['stim_top'].fillna(df['stim_bottom'].shift(3))
    df['cue'] = df['cue'].fillna(df['cue'].shift(2))
    df.loc[:,'correct'] = df.correct.astype(float)
    return df

def DPX_post(df):
    if not 'correct' in df.columns:
        df.insert(0,'correct',numpy.nan)
        df.loc[:,'correct'] = df['correct'].astype(object)
    subset = df.query('trial_id == "probe" and correct != correct and rt != -1')
    for i,row in subset.iterrows():
        correct = ((row['condition'] == 'AX' and row['key_press'] == 37) or \
            (row['condition'] != 'AX' and row['key_press'] == 40))
        df.set_value(i, 'correct', correct)
    return df
    
def hierarchical_post(df):
    if numpy.sum(~pandas.isnull(df['correct_response'])) != numpy.sum(~pandas.isnull(df.get('correct'))):
        correct =  [float(trial['correct_response'] == trial['key_press']) if not pandas.isnull(trial['correct_response']) else numpy.nan for i, trial in df.iterrows()]
    else:
        correct = df['correct'].astype(float)
    if 'correct' in df.columns:
        df = df.loc[:,df.columns != 'correct']
    df.insert(0, 'correct', correct)
    return df

def IST_post(df):
    if 'trial_num' not in df.columns:
        df = df.drop('box_id', axis = 1)
        tmp = df['mouse_click'].apply(lambda x: 'choice' if x in ['26','27'] else numpy.nan)
        df.loc[:,'trial_id'] = tmp.fillna(df['trial_id'])
        subset=df[df['trial_id'].apply(lambda x: x in ['choice','stim'])][1:]['trial_id']
        trial_num = 0
        trial_nums = []
        for row in subset:
            trial_nums.append(trial_num)
            if row == "choice":
                trial_num+=1
            if trial_num == 10:
                trial_num = 0
        df.loc[subset.index,'trial_num'] = trial_nums
        df.rename(columns = {'clicked_on': 'color_clicked'}, inplace = True)
    # Add in correct column   
    subset = df[(df['trial_id'] == 'choice') & (df['exp_stage'] != 'practice')]
    if 'correct' not in df:
        correct = (subset['color_clicked'] == subset['correct_response']).astype(float)
        df.insert(0, 'correct', correct)
    else:
        df.loc[:,'correct'] = df['correct'].map(lambda x: float(x) if not pandas.isnull(x) else numpy.nan)
    # Add chosen and total boxes clicked to choice rows and score
    final_choices = subset[['worker_id','exp_stage','color_clicked','trial_num']]
    stim_subset = df[(df['trial_id'] == 'stim') & (df['exp_stage'] != 'practice')]
    try:
        box_clicks = stim_subset.groupby(['worker_id','exp_stage','trial_num'])['color_clicked'].value_counts()
        counts = []
        for i,row in final_choices.iterrows():
            try:
                index = row[['worker_id','exp_stage','trial_num']].tolist()
                chosen_count = box_clicks[index[0], index[1], index[2]].get(row['color_clicked'],0)
                counts.append(chosen_count)
            except KeyError:
                counts.append(0)
        df.insert(0,'chosen_boxes_clicked',pandas.Series(index = final_choices.index, data = counts))
        df.insert(0,'clicks_before_choice', pandas.Series(index = final_choices.index, data =  subset['which_click_in_round']-1))    
        df.insert(0,'points', df['reward'].shift(-1))
        # calculate probability of being correct
        def get_prob(boxes_opened,chosen_boxes_opened):
            if boxes_opened == boxes_opened:
                z = 25-int(boxes_opened)
                a = 13-int(chosen_boxes_opened)
                if a < 0:
                    return 1.0
                else:
                    return numpy.sum([factorial(z)/float(factorial(k)*factorial(z-k)) for k in range(a,z+1)])/2**z
            else:
                return numpy.nan
        probs=numpy.vectorize(get_prob)(df['clicks_before_choice'],df['chosen_boxes_clicked'])
        df.insert(0,'P_correct_at_choice', probs)
    except IndexError:
        print(('Workers: %s did not open any boxes ' % df.worker_id.unique()))
    return df
        
def keep_track_post(df):
    for i,row in df.iterrows():
        if not pandas.isnull(row['responses']) and row['trial_id'] == 'response':
            response = row['responses']
            response = response[response.find('":"')+3:-2]
            response = re.split(r'[,; ]+', response)
            response = [x.lower().strip() for x in response]
            df.set_value(i,'responses', response)
    if 'correct_responses' in df.columns:
        df.loc[:,'possible_score'] = numpy.nan
        df.loc[:,'score'] = numpy.nan
        subset = df[[isinstance(i,dict) for i in df['correct_responses']]]
        for i,row in subset.iterrows():
            targets = list(row['correct_responses'].values())
            response = row['responses']
            score = sum([word in targets for word in response])
            df.set_value(i, 'score', score)
            df.set_value(i, 'possible_score', len(targets))
    return df

def local_global_post(df):
    subset = df[df['trial_id'] == "stim"]
    df.loc[subset.index,'switch'] = abs(subset['condition'].apply(lambda x: 1 if x=='local' else 0).diff())
    df.loc[subset.index,'correct'] = (subset['key_press'] == subset['correct_response'])
    df.loc[:,'correct'] = df['correct'].map(lambda x: float(x) if (x==x) else numpy.nan)
    conflict = (df['local_shape']==df['global_shape']).apply(lambda x: 'congruent' if x else 'incongruent')
    neutral = (df['local_shape'].isin(['o']) | df['global_shape'].isin(['o']))
    df.loc[conflict.index, 'conflict_condition'] = conflict
    df.loc[neutral,'conflict_condition'] = 'neutral'
    return df
    
def probabilistic_selection_post(df):
    if (numpy.sum(pandas.isnull(df.query('exp_stage == "test"')['correct']))>0):
        def get_correct_response(stims):
            if stims[0] > stims[1]:
                return 37
            else:
                return 39
        df.replace(to_replace = 'practice', value = 'training', inplace = True)
        subset = df.query('exp_stage == "test"')
        correct_responses = subset['condition'].map(lambda x: get_correct_response(x.split('_')))
        correct = correct_responses == subset['key_press']
        df.loc[correct_responses.index,'correct_response'] = correct_responses
        df.loc[correct.index,'correct'] = correct
    if ('optimal_response' not in df.columns):
        df.loc[:,'optimal_response'] = df['condition'].map(lambda x: [37,39][numpy.diff([int(a) for a in x.split('_')])[0]>0] if x == x else numpy.nan)
    
    # add FB column
    if 'feedback' not in df.columns:
        df.loc[:,'feedback'] = df[df['exp_stage'] == "training"]['correct']
        df.loc[:,'correct'] = df['key_press'] == df['optimal_response']
    df.loc[:,'feedback'] = df['feedback'].astype(float)
    df.loc[:,'correct'] = df['correct'].astype(float)
        
    df = df.drop('optimal_response', axis = 1)
    # add condition collapsed column
    df.loc[:,'condition_collapsed'] = df['condition'].map(lambda x: '_'.join(sorted(x.split('_'))) if x == x else numpy.nan)
    # add column indicating stim chosen
    choices = [[37,39].index(x) if x in [37,39] else numpy.nan for x in df['key_press']]
    stims = df['condition'].apply(lambda x: x.split('_') if x==x else numpy.nan)
    df.loc[:,'stim_chosen'] = [s[c] if c==c else numpy.nan for s,c in zip(stims,choices)]
    #learning check - ensure during test that worker performed above chance on easiest training pair
    passed_workers = df.query('exp_stage == "test" and condition_collapsed == "20_80"').groupby('worker_id')['correct'].agg(lambda x: (numpy.mean(x)>.5) and (len(x) == 6)).astype('bool')
    if numpy.sum(passed_workers) < len(passed_workers):
        print("Probabilistic Selection: %s failed the manipulation check" % list(passed_workers[passed_workers == False].index))    
    passed_workers = list(passed_workers[passed_workers].index) 
    df.loc[:,"passed_check"] = df['worker_id'].map(lambda x: x in passed_workers)
    return df
    
def PRP_post(df):
    df['trial_id'].replace(to_replace = '', value = 'stim', inplace = True)
    def remove_nan(lst):
        return list(lst[~numpy.isnan(lst)])
    choice1_stims = remove_nan(df['choice1_stim'].unique())
    choice2_stims = remove_nan(df['choice2_stim'].unique())
    df.loc[:,'choice1_stim'] = df['choice1_stim'].map(lambda x: choice1_stims.index(x) if x==x else numpy.nan)
    df.loc[:,'choice2_stim'] = df['choice2_stim'].map(lambda x: choice2_stims.index(x) if x==x else numpy.nan)
    # separate choice and rt for the two choices
    df.loc[:,'key_presses'] = df['key_presses'].map(lambda x: json.loads(x) if x==x else x)
    df.loc[:,'rt'] = df['rt'].map(lambda x: json.loads(x) if isinstance(x,str) else x)
    subset = df[(df['trial_id'] == "stim") & (~pandas.isnull(df['stim_durations']))]
    # separate rt
    df.insert(0, 'choice1_rt', pandas.Series(index = subset.index, data = [x[0] for x in subset['rt']]))
    df.insert(0, 'choice2_rt', pandas.Series(index = subset.index, data = [x[1] for x in subset['rt']]) - subset['ISI'])
    df = df.drop('rt', axis = 1)
    # separate key press
    df.insert(0, 'choice1_key_press', pandas.Series(index = subset.index, data = [x[0] for x in subset['key_presses']]))
    df.insert(0, 'choice2_key_press', pandas.Series(index = subset.index, data = [x[1] for x in subset['key_presses']]))
    df = df.drop('key_presses', axis = 1)
    # calculate correct
    choice1_correct = df['choice1_key_press'] == df['choice1_correct_response']
    choice2_correct = df['choice2_key_press'] == df['choice2_correct_response']
    df.insert(0,'choice1_correct', pandas.Series(index = subset.index, data = choice1_correct))
    df.insert(0,'choice2_correct', pandas.Series(index = subset.index, data = choice2_correct))
    return df

def recent_probes_post(df):
    df['correct'] = df['correct'].astype(float)
    df['stim'] = df['stim'].fillna(df['stim'].shift(2))
    df['stims_1back'] = df['stims_1back'].fillna(df['stims_1back'].shift(2))
    df['stims_2back'] = df['stims_2back'].fillna(df['stims_2back'].shift(2))
    return df
    
    
def shift_post(df):
    if not 'shift_type' in df.columns:
        df.loc[:,'shift_type'] = numpy.nan
        df['shift_type'] = df['shift_type'].astype(object)
        last_feature = ''
        last_dim = ''
        for i,row in df.iterrows():
            if row['trial_id'] == 'stim':
                if last_feature == '':
                    shift_type = 'stay'
                elif row['rewarded_feature'] == last_feature:
                    shift_type = 'stay'
                elif row['rewarded_dim'] == last_dim:
                    shift_type = 'intra'
                else:
                    shift_type = 'extra'
                last_feature = row['rewarded_feature']
                last_dim = row['rewarded_dim']
                df.set_value(i,'shift_type', shift_type)
            elif row['trial_id'] == 'feedback':
                df.set_value(i,'shift_type', shift_type)
    if 'FB' in df.columns:
        df.loc[:,'feedback'] = df['FB']
        df = df.drop('FB', axis = 1)
    df.loc[:,'choice_stim'] = [json.loads(i) if isinstance(i,str) else numpy.nan for i in df['choice_stim']]
    
    if not 'correct' in df.columns:
        # Get correct choices
        def get_correct(x):
            if isinstance(x['choice_stim'], dict):
                return float(x['choice_stim'][x['rewarded_dim'][:-1]] == x['rewarded_feature'])
            else:
                return numpy.nan
        correct=df.apply(get_correct,axis = 1)
        df.insert(0,'correct',correct)    
    else:
        df.loc[:,'correct'] = df['correct'].map(lambda x: float(x) if x==x else numpy.nan)
    return df
    
def span_post(df):
    df = df[df['rt'].map(lambda x: isinstance(x,int))]
    correct_col = pandas.Series(index = df.index)
    if 'correct' in df.columns:
        correct_col.fillna(df['correct'].astype(float), inplace = True)
        df = df.loc[:,df.columns != 'correct']
    if 'feedback' in df.columns:
        correct_col.fillna(df['feedback'].astype(float), inplace = True)
        df = df.loc[:,df.columns != 'feedback']
    df.insert(0,'correct',correct_col)
    return df
    
def stop_signal_post(df):
    df.insert(0,'stopped',df['key_press'] == -1)
    df.loc[:,'correct'] = df['key_press'] == df['correct_response']
    if 'SSD' in df.columns:
        df.drop('SSD',inplace = True, axis = 1)
    return df  

def threebytwo_post(df):
    for worker in numpy.unique(df['worker_id']):
        correct_responses = {}
        subset = df.query('trial_id == "stim" and worker_id == "%s"' % worker)
        if (numpy.sum(pandas.isnull(subset.query('exp_stage == "test"')['correct']))>0):
            correct_responses['color'] = subset.query('task == "color"').groupby('stim_color')['correct_response'].mean().to_dict()
            correct_responses['parity'] = subset.query('task == "parity"').groupby(subset['stim_number']%2 == 1)['correct_response'].mean().to_dict()
            correct_responses['magnitude'] = subset.query('task == "magnitude"').groupby(subset['stim_number']>5)['correct_response'].mean().to_dict()
            color_responses = (subset.query('task == "color"')['stim_color']).map(lambda x: correct_responses['color'][x])
            parity_responses = (subset.query('task == "parity"')['stim_number']%2==1).map(lambda x: correct_responses['parity'][x])
            magnitude_responses = (subset.query('task == "magnitude"')['stim_number']>5).map(lambda x: correct_responses['magnitude'][x])
            df.loc[color_responses.index,'correct_response'] = color_responses
            df.loc[parity_responses.index,'correct_response'] = parity_responses
            df.loc[magnitude_responses.index,'correct_response'] = magnitude_responses
            df.loc[subset.index,'correct'] =df.loc[subset.index,'key_press'] == df.loc[subset.index,'correct_response']
    df.loc[:,'correct'] = df['correct'].map(lambda x: float(x) if x==x else numpy.nan)
    df.insert(0, 'CTI', pandas.Series(data = df[df['trial_id'] == "cue"].block_duration.tolist(), \
                                        index = df[df['trial_id'] == "stim"].index))
    return df
        
def TOL_post(df):
    labels = ['practice'] + list(range(12))
    if 'problem_id' not in df.columns:
        df_index = df.query('(target == target and rt != -1) or trial_id == "feedback"').index
        problem_time = 0
        move_stage = 'to_hand'
        problem_id = 0
        for loc in df_index:
            if df.loc[loc,'trial_id'] != 'feedback':
                df.loc[loc,'trial_id'] = move_stage
                df.loc[loc,'problem_id'] = labels[problem_id%13]
                if move_stage == 'to_board':
                    move_stage = 'to_hand'
                else:
                    move_stage = 'to_board'
                problem_time += df.loc[loc,'rt']
            else:
                df.loc[loc,'problem_time'] = problem_time
                problem_time = 0
                problem_id += 1
    # Change current position type to list if necessary
    index = [not isinstance(x,list) and x==x for x in df['current_position']]
    df.loc[index,'current_position'] = df.loc[index,'current_position'].map(lambda x: [x['0'], x['1'], x['2']])
    if 'correct' not in df:
        df.loc[:,'correct'] = (df['current_position'] == df['target'])
    else:
        subset = df.query('trial_id != "feedback"').index
        df.loc[subset,'correct'] = (df.loc[subset,'current_position'] == df.loc[subset,'target'])
    df.loc[:,'correct'] = df['correct'].map(lambda x: float(x) if x==x else numpy.nan)
    return df
    

def two_stage_decision_post(df):
    group_df = pandas.DataFrame()
    trials = df.groupby('exp_stage')['trial_num'].max()
    for worker_i, worker in enumerate(numpy.unique(df['worker_id'])):
        try:
            rows = []
            worker_df = df[df['worker_id'] == worker]
            for stage in ['practice', 'test']:
                stage_df = worker_df[worker_df['exp_stage'] == stage]
                for i in range(int(trials[stage]+1)):
                    trial = stage_df.loc[df['trial_num'] == i]
                    #set row to first stage
                    row = trial.iloc[0].to_dict()  
                    ss,fb = {}, {}
                    row['trial_id'] = 'incomplete_trial'
                    if len(trial) >= 2:
                        ss = trial.iloc[1]
                        row['time_elapsed'] = ss['time_elapsed']
                    if len(trial) == 3:
                        fb = trial.iloc[2]
                        row['time_elapsed'] = fb['time_elapsed']
                        row['trial_id'] = 'complete_trial'
                    row['rt_first'] = row.pop('rt')
                    row['rt_second'] = ss.get('rt',-1)
                    row['stage_second'] = ss.get('stage',-1)
                    row['stim_order_first'] = row.pop('stim_order')
                    row['stim_order_second'] = ss.get('stim_order_second',-1)
                    row['stim_selected_first'] = row.pop('stim_selected')
                    row['stim_selected_second'] = ss.get('stim_selected',-1)
                    row['stage_transition'] = ss.get('stage_transition',numpy.nan)
                    row['feedback'] = fb.get('feedback',numpy.nan)
                    row['FB_probs'] = fb.get('FB_probs',numpy.nan)
                    rows.append(row)
            worker_df = pandas.DataFrame(rows)
            trial_index = ["%s_%s_%s" % ('two_stage_decision',worker_i,x) for x in range(len(worker_df))]
            worker_df.index = trial_index
            #manipulation check
            win_stay = 0.0
            subset = worker_df[worker_df['exp_stage']=='test']
            for stage in numpy.unique(subset['stage_second']):
                stage_df=subset[subset['stage_second']==stage][['feedback','stim_selected_second']]
                stage_df.insert(0, 'next_choice', stage_df['stim_selected_second'].shift(-1))
                stage_df.insert(0, 'stay', stage_df['stim_selected_second'] == stage_df['next_choice'])
                win_stay+= stage_df[stage_df['feedback']==1]['stay'].sum()
            win_stay_proportion = win_stay/subset['feedback'].sum()
            if win_stay_proportion > .5:
                worker_df.loc[:,'passed_check'] = True
            else:
                worker_df.loc[:,'passed_check'] = False
                print('Two Stage Decision: Worker %s failed manipulation check. Win stay = %s' % (worker, win_stay_proportion))
            group_df = pandas.concat([group_df,worker_df])
        except:
            print(('Could not process two_stage_decision dataframe with worker: %s' % worker))
    if (len(group_df)>0):
        group_df.insert(0, 'switch', group_df['stim_selected_first'].diff()!=0)
        group_df.insert(0, 'stage_transition_last', group_df['stage_transition'].shift(1))
        group_df.insert(0, 'feedback_last', group_df['feedback'].shift(1))
        df = group_df
    return df
    
 
"""
DV functions
"""

@multi_worker_decorate
def calc_adaptive_n_back_DV(df):
    """ Calculate dv for adaptive_n_back task. Maximum load
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('exp_stage != "practice"')
    dvs = {'mean_load': df['load'].mean()}
    dvs['max_load'] =  df['load'].max()
    description = 'max load'
    return dvs, description
 
@multi_worker_decorate
def calc_ANT_DV(df):
    """ Calculate dv for attention network task: Accuracy and average reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1').reset_index()
    dvs = calc_common_stats(df)
    cue_rt = df.groupby(['cue'])[['rt','correct']].agg(['median','mean'])
    flanker_rt = df.groupby(['flanker_type'])[['rt','correct']].agg(['median','mean'])
    dvs['alerting_rt'] = (cue_rt.loc['nocue'] - cue_rt.loc['double'])['rt']['median']
    dvs['orienting_rt'] = (cue_rt.loc['center'] - cue_rt.loc['spatial'])['rt']['median']
    dvs['conflict_rt'] = (flanker_rt.loc['incongruent'] - flanker_rt.loc['congruent'])['rt']['median']
    dvs['alerting_accuracy'] = (cue_rt.loc['nocue'] - cue_rt.loc['double'])['correct']['mean']
    dvs['orienting_accuracy'] = (cue_rt.loc['center'] - cue_rt.loc['spatial'])['correct']['mean']
    dvs['conflict_accuracy'] = (flanker_rt.loc['incongruent'] - flanker_rt.loc['congruent'])['correct']['mean']
    dvs['missed_percent'] = missed_percent
    description = """
    DVs for "alerting", "orienting" and "conflict" attention networks are of primary
    interest for the ANT task, all concerning differences in RT. 
    Alerting is defined as nocue - double cue trials. Positive values
    indicate the benefit of an alerting double cue. Orienting is defined as center - spatial cue trials.
    Positive values indicate the benefit of a spatial cue. Conflict is defined as
    incongruent - congruent flanker trials. Positive values indicate the benefit of
    congruent trials (or the cost of incongruent trials). RT measured in ms and median
    RT are used for all comparisons.
    """
    return dvs, description
    
@multi_worker_decorate
def calc_ART_sunny_DV(df):
    """ Calculate dv for choice reaction time: Accuracy and average reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and key_press != -1').reset_index()
    dvs = calc_common_stats(df)
    dvs['missed_percent'] = missed_percent
    scores = df.groupby('release').max()['tournament_bank']
    clicks = df.groupby('release').mean()['trial_num']
    dvs['Keep_score'] = scores['Keep']    
    dvs['Release_score'] = scores['Release']  
    dvs['Keep_clicks'] = clicks['Keep']    
    dvs['Release_clicks'] = clicks['Release']  
    description = 'DVs are the total tournament score for each condition and the average number of clicks per condition'  
    return dvs, description

@multi_worker_decorate
def calc_CCT_cold_DV(df):
    """ Calculate dv for ccolumbia card task, cold version
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('exp_stage != "practice"').reset_index()
    df['num_loss_cards'] = df['num_loss_cards'].astype('float')
    rs = smf.ols(formula = 'num_cards_chosen ~ gain_amount + loss_amount + num_loss_cards', data = df).fit()
    dvs = {}
    dvs['avg_cards_chosen'] = df['num_cards_chosen'].mean()
    dvs['gain_sensitivity'] = rs.params['gain_amount']
    dvs['loss_sensitivity'] = rs.params['loss_amount']
    dvs['probability_sensitivity'] = rs.params['num_loss_cards']
    dvs['information_use'] = numpy.sum(rs.pvalues[1:]<.05)
    description = """
        Avg_cards_chosen is a measure of risk ttaking
        gain sensitivity: beta value for regression predicting number of cards
            chosen based on gain amount on trial
        loss sensitivty: as above for loss amount
        probability sensivitiy: as above for number of loss cards
        information use: ranges from 0-3 indicating how many of the sensivitiy
            parameters significantly affect the participant's 
            choices at p < .05
    """
    return dvs, description


@multi_worker_decorate
def calc_CCT_hot_DV(df):
    """ Calculate dv for ccolumbia card task, cold version
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('exp_stage != "practice" and mouse_click == "collectButton"').reset_index()
    df['num_loss_cards'] = df['num_loss_cards'].astype('float')
    subset = df[~df['clicked_on_loss_card'].astype(bool)]
    rs = smf.ols(formula = 'total_cards ~ gain_amount + loss_amount + num_loss_cards', data = subset).fit()
    dvs = {}
    dvs['avg_cards_chosen'] = subset['total_cards'].mean()
    dvs['gain_sensitivity'] = rs.params['gain_amount']
    dvs['loss_sensitivity'] = rs.params['loss_amount']
    dvs['probability_sensitivity'] = rs.params['num_loss_cards']
    dvs['information_use'] = numpy.sum(rs.pvalues[1:]<.05)
    description = """
        Avg_cards_chosen is a measure of risk ttaking
        gain sensitivity: beta value for regression predicting number of cards
            chosen based on gain amount on trial
        loss sensitivty: as above for loss amount
        probability sensivitiy: as above for number of loss cards
        information use: ranges from 0-3 indicating how many of the sensivitiy
            parameters significantly affect the participant's 
            choices at p < .05
    """
    return dvs, description


@multi_worker_decorate
def calc_choice_reaction_time_DV(df):
    """ Calculate dv for choice reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1').reset_index()
    dvs = calc_common_stats(df)
    dvs['missed_percent'] = missed_percent
    description = 'standard'  
    return dvs, description

@multi_worker_decorate
def calc_cognitive_reflection_DV(df):
    dvs = {'accuracy': df['correct'].mean(),
           'intuitive_proportion': df['responded_intuitively'].mean()
           }
    description = 'how many questions were answered correctly'
    return dvs,description

@multi_worker_decorate
def calc_dietary_decision_DV(df):
    """ Calculate dv for dietary decision task. Calculate the effect of taste and
    health rating on choice
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df[~ pandas.isnull(df['taste_diff'])].reset_index()
    rs = smf.ols(formula = 'coded_response ~ health_diff + taste_diff', data = df).fit()
    dvs = {}
    dvs['health_sensitivity'] = rs.params['health_diff']
    dvs['taste_sensitivity'] = rs.params['taste_diff']
    description = """
        Both taste and health sensitivity are calculated based on the decision phase.
        On each trial the participant indicates whether they would prefer a food option
        over a reference food. Their choice is regressed on the subjective health and
        taste difference between that option and the reference item. Positive values
        indicate that the option's higher health/taste relates to choosing the option
        more often
    """
    return dvs,description
    
@multi_worker_decorate
def calc_digit_span_DV(df):
    """ Calculate dv for digit span: forward and reverse span
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('exp_stage != "practice" and rt != -1').reset_index()
    dvs = calc_common_stats(df)
    span = df.groupby(['condition'])['num_digits'].mean()
    dvs['forward_span'] = span['forward']
    dvs['reverse_span'] = span['reverse']
    description = 'standard'  
    return dvs, description

@multi_worker_decorate
def calc_directed_forgetting_DV(df):
    """ Calculate dv for directed forgetting
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('exp_stage != "practice" and rt != -1').reset_index()
    df_correct = df.query('correct == True').reset_index()
    dvs = calc_common_stats(df)
    rt_contrast = df_correct.groupby('probe_type').rt.median()
    acc_contrast = df.groupby('probe_type').correct.mean()
    dvs['proactive_inteference_rt'] = rt_contrast['neg'] - rt_contrast['con']
    dvs['proactive_inteference_acc'] = acc_contrast['neg'] - acc_contrast['con']
    description = """
    Each DV contrasts trials where subjects were meant to forget the letter vs.
    trials where they had never seen the letter. On both types of trials the
    subject is meant to respond that the letter was not in the memory set. RT
    contrast is only computed for correct trials
    """ 
    return dvs, description

@multi_worker_decorate
def calc_DPX_DV(df):
    """ Calculate dv for dot pattern expectancy task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1').reset_index()
    df_correct = df.query('correct == True').reset_index()
    dvs = calc_common_stats(df)
    contrast_df = df_correct.groupby('condition')['rt'].median()
    dvs['AY_diff'] = contrast_df['AY'] - df['rt'].median()
    dvs['BX_diff'] = contrast_df['BX'] - df['rt'].median()
    dvs['missed_percent'] = missed_percent
    description = 'standard'  
    return dvs, description


@multi_worker_decorate
def calc_go_nogo_DV(df):
    """ Calculate dv for go-nogo task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('exp_stage != "practice"').reset_index()
    dvs = {}
    dvs['overall_accuracy'] = df['correct'].mean()
    dvs['go_accuracy'] = df[df['condition'] == 'go']['correct'].mean()
    dvs['nogo_accuracy'] = df[df['condition'] == 'nogo']['correct'].mean()
    dvs['go_rt'] = df[(df['condition'] == 'go') & (df['rt'] != -1)]['rt'].median()
    description = """
        Calculated accuracy for go/stop conditions. 75% of trials are go
    """
    return dvs, description


@multi_worker_decorate
def calc_hierarchical_rule_DV(df):
    """ Calculate dv for hierarchical learning task. 
    DVs
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1').reset_index()
    dvs = calc_common_stats(df)
    dvs['score'] = df['correct'].sum()
    dvs['missed_percent'] = missed_percent
    description = 'average reaction time'  
    return dvs, description

@multi_worker_decorate
def calc_IST_DV(df):
    """ Calculate dv for information sampling task
    DVs
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('exp_stage != "practice"').reset_index()
    dvs = {}
    latency_df = df[df['trial_id'] == "stim"].groupby('exp_stage')['rt'].median()
    points_df = df[df['trial_id'] == "choice"].groupby('exp_stage')['points'].sum()
    contrast_df = df[df['trial_id'] == "choice"].groupby('exp_stage')['correct','P_correct_at_choice','clicks_before_choice'].mean()
    for condition in ['Decreasing Win', 'Fixed Win']:
        dvs[condition + '_rt'] = latency_df.get(condition,numpy.nan)
        dvs[condition + '_total_points'] = points_df.loc[condition]
        dvs[condition + '_boxes_opened'] = contrast_df.loc[condition,'clicks_before_choice']
        dvs[condition + '_accuracy'] = contrast_df.loc[condition, 'correct']
        dvs[condition + '_P_correct'] = contrast_df.loc[condition, 'P_correct_at_choice']
    description = """ Each dependent variable is calculated for the two conditions:
    DW (Decreasing Win) and FW (Fixed Win). "RT" is the median rt over every choice to open a box,
    "boxes opened" is the mean number of boxes opened before choice, "accuracy" is the percent correct
    over trials and "P_correct" is the P(correct) given the number and distribution of boxes opened on that trial
    """
    return dvs, description

@multi_worker_decorate
def calc_keep_track_DV(df):
    """ Calculate dv for choice reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('exp_stage != "practice" and rt != -1').reset_index()
    score = df['score'].sum()/df['possible_score'].sum()
    dvs = {}
    dvs['score'] = score
    description = 'percentage of items remembered correctly'  
    return dvs, description

@multi_worker_decorate
def calc_local_global_DV(df):
    """ Calculate dv for hierarchical learning task. 
    DVs
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1').reset_index()
    df_correct = df.query('correct == True').reset_index()
    rt_contrast = df_correct.groupby('conflict_condition').rt.median()
    acc_contrast = df.groupby('conflict_condition').correct.mean()
    dvs = calc_common_stats(df)
    dvs['congruent_facilitation_rt'] = (rt_contrast['neutral'] - rt_contrast['congruent'])
    dvs['incongruent_harm_rt'] = (rt_contrast['incongruent'] - rt_contrast['neutral'])
    dvs['congruent_facilitation_accuracy'] = (acc_contrast['congruent'] - acc_contrast['neutral'])
    dvs['incongruent_harm_accuracy'] = (acc_contrast['neutral'] - acc_contrast['incongruent'])
    switch_rt = df_correct.groupby('switch').rt.median()
    switch_acc = df.groupby('switch').correct.mean()
    dvs['switch_cost_rt'] = (switch_rt[1] - switch_rt[0])
    dvs['switch_cost_accuracy'] = (switch_acc[1] - switch_acc[0])
    dvs['missed_percent'] = missed_percent
    description = """
        local-global incongruency effect calculated for accuracy and RT. 
        Facilitation for RT calculated as neutral-congruent. Positive values indicate speeding on congruent trials.
        Harm for RT calculated as incongruent-neutral. Positive values indicate slowing on incongruent trials
        Facilitation for accuracy calculated as congruent-neutral. Positives values indicate higher accuracy for congruent trials
        Harm for accuracy calculated as neutral - incongruent. Positive values indicate lower accuracy for incongruent trials
        Switch costs calculated as switch-stay for rt and stay-switch for accuracy. Thus positive values indicate slowing and higher
        accuracy on switch trials. Expectation is positive rt switch cost, and negative accuracy switch cost
        RT measured in ms and median RT is used for comparison.
        """
    return dvs, description
    
@multi_worker_decorate
def calc_probabilistic_selection_DV(df):
    """ Calculate dv for probabilistic selection task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    def get_value_diff(lst, values):
        return abs(values[lst[0]] - values[lst[1]])
    def get_value_sum(lst,values):
        return values[lst[0]] + values[lst[1]]
    missed_percent = (df['rt']==-1).mean()
    df = df[df['rt'] != -1].reset_index()
    
    #Calculate regression DVs
    train = df.query('exp_stage == "training"')
    values = train.groupby('stim_chosen')['feedback'].mean()
    df.loc[:,'value_diff'] = df['condition_collapsed'].apply(lambda x: get_value_diff(x.split('_'), values) if x==x else numpy.nan)
    df.loc[:,'value_sum'] =  df['condition_collapsed'].apply(lambda x: get_value_sum(x.split('_'), values) if x==x else numpy.nan)  
    test = df.query('exp_stage == "test"')
    rs = smf.glm(formula = 'correct ~ value_diff*value_sum', data = test, family = sm.families.Binomial()).fit()
    
    #Calculate non-regression, simpler DVs
    pos_subset = test[test['condition_collapsed'].map(lambda x: '20' not in x)]
    neg_subset = test[test['condition_collapsed'].map(lambda x: '80' not in x)]
    chose_A = pos_subset[pos_subset['condition_collapsed'].map(lambda x: '80' in x)]['stim_chosen']=='80'
    chose_C = pos_subset[pos_subset['condition_collapsed'].map(lambda x: '70' in x and '80' not in x and '30' not in x)]['stim_chosen']=='70'
    pos_acc = (numpy.sum(chose_A) + numpy.sum(chose_C))/float((len(chose_A) + len(chose_C)))
    
    avoid_B = neg_subset[neg_subset['condition_collapsed'].map(lambda x: '20' in x)]['stim_chosen']!='20'
    avoid_D = neg_subset[neg_subset['condition_collapsed'].map(lambda x: '30' in x and '20' not in x and '70' not in x)]['stim_chosen']!='30'
    neg_acc = (numpy.sum(avoid_B) + numpy.sum(avoid_D))/float((len(avoid_B) + len(avoid_D)))
    
    dvs = calc_common_stats(df)
    dvs['reg_value_sensitivity'] = rs.params['value_diff']
    dvs['reg_positive_learning_bias'] = rs.params['value_diff:value_sum']
    dvs['positive_accuracy'] = pos_acc
    dvs['negative_accuracy'] = neg_acc
    dvs['positive_learning_bias'] = pos_acc/neg_acc
    dvs['overall_test_acc'] = test['correct'].mean()
    dvs['missed_percent'] = missed_percent
    description = """
        The primary DV in this task is whether people do better choosing
        positive stimuli or avoiding negative stimuli. Two different measurements
        are calculated. The first is a regression that predicts participant
        accuracy based on the value difference between the two options (defined by
        the participant's actual experience with the two stimuli) and the sum of those
        values. A significant effect of value difference would say that participants
        are more likely to be correct on easier trials. An interaction between the value
        difference and value-sum would say that this effect (the relationship between
        value difference and accuracy) differs based on the sum. A positive learning bias
        would say that the relationship between value difference and accuracy is greater 
        when the overall value is higher.
        
        Another way to calculate a similar metric is to calculate participant accuracy when 
        choosing the two most positive stimuli over other novel stimuli (not the stimulus they 
        were trained on). Negative accuracy can similarly be calculated based on the 
        probability the participant avoided the negative stimuli. Bias is calculated as
        their positive accuracy/negative accuracy. Thus positive values indicate that the
        subject did better choosing positive stimuli then avoiding negative ones. 
        Reference: http://www.sciencedirect.com/science/article/pii/S1053811914010763
    """
    return dvs, description

@multi_worker_decorate
def calc_ravens_DV(df):
    """ Calculate dv for ravens task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('stim_response == stim_response').reset_index()
    dvs = calc_common_stats(df)
    dvs['score'] = df['score_response'].sum()
    description = 'Score is the number of correct responses out of 18'
    return dvs,description    

@multi_worker_decorate
def calc_recent_probes_DV(df):
    """ Calculate dv for recent_probes
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('exp_stage != "practice" and rt != -1').reset_index()
    df_correct = df.query('correct == True').reset_index()
    dvs = calc_common_stats(df)
    rt_contrast = df_correct.groupby('probeType').rt.median()
    acc_contrast = df.groupby('probeType').correct.mean()
    dvs['proactive_inteference_rt'] = rt_contrast['rec_neg'] - rt_contrast['xrec_neg']
    dvs['proactive_inteference_acc'] = acc_contrast['rec_neg'] - acc_contrast['xrec_neg']
    description = """
    proactive interference defined as the difference in reaction time and accuracy
    for negative trials (where the probe was not part of the memory set) between
    "recent" trials (where the probe was part of the previous trial's memory set)
    and "non-recent trials" where the probe wasn't.
    """ 
    return dvs, description
    
@multi_worker_decorate
def calc_shift_DV(df):
    """ Calculate dv for shift task. I
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1').reset_index()
    dvs = calc_common_stats(df)
    dvs['missed_percent'] = missed_percent
    
    rs = smf.glm('correct ~ trials_since_switch', data = df, family = sm.families.Binomial()).fit()
    dvs['learning_rate'] = rs.params['trials_since_switch']    
    description = """
        Shift task has a complicated analysis. Right now just using accuracy and 
        slope of learning after switches (which I'm calling "learning rate")
        """
    return dvs, description
    
@multi_worker_decorate
def calc_simon_DV(df):
    """ Calculate dv for simon task. Incongruent-Congruent, median RT and Percent Correct
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df_correct = df.query('correct == True').reset_index()
    dvs = calc_common_stats(df)
    rt_contrast = df_correct.groupby('condition').rt.median()
    acc_contrast = df.groupby('condition').correct.mean()
    dvs['simon_rt'] = rt_contrast['incongruent']-rt_contrast['congruent']
    dvs['simon_accuracy'] = acc_contrast['incongruent']-acc_contrast['congruent']
    dvs['missed_percent'] = missed_percent
    description = """
        simon effect calculated for accuracy and RT: incongruent-congruent.
        RT measured in ms and median RT is used for comparison.
        """
    return dvs, description
    
@multi_worker_decorate
def calc_simple_RT_DV(df):
    """ Calculate dv for simple reaction time. Average Reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1').reset_index()
    dvs = calc_common_stats(df)
    dvs['avg_rt'] = df['rt'].median()
    dvs['missed_percent'] = missed_percent
    description = 'average reaction time'  
    return dvs, description

@multi_worker_decorate
def calc_shape_matching_DV(df):
    """ Calculate dv for shape_matching task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1').reset_index()
    dvs = calc_common_stats(df)
    dvs['missed_percent'] = missed_percent
    contrast = df.groupby('condition').rt.median()
    dvs['stimulus_interference'] = contrast['SDD'] - contrast['SNN']
    description = 'standard'  
    return dvs, description
    
@multi_worker_decorate
def calc_spatial_span_DV(df):
    """ Calculate dv for spatial span: forward and reverse mean span
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('exp_stage != "practice" and rt != -1').reset_index()
    dvs = calc_common_stats(df)
    span = df.groupby(['condition'])['num_spaces'].mean()
    dvs['forward_span'] = span['forward']
    dvs['reverse_span'] = span['reverse']
    description = 'standard'  
    return dvs, description
    
@multi_worker_decorate
def calc_stroop_DV(df):
    """ Calculate dv for stroop task. Incongruent-Congruent, median RT and Percent Correct
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1').reset_index()
    df_correct = df.query('correct == True').reset_index()
    dvs = calc_common_stats(df)
    rt_contrast = df_correct.groupby('condition').rt.median()
    acc_contrast = df.groupby('condition').correct.mean()
    dvs['stroop_rt'] = rt_contrast['incongruent']-rt_contrast['congruent']
    dvs['stroop_accuracy'] = acc_contrast['incongruent']-acc_contrast['congruent']
    dvs['missed_percent'] = missed_percent
    description = """
        stroop effect calculated for accuracy and RT: incongruent-congruent.
        RT measured in ms and median RT is used for comparison.
        """
    return dvs, description

@multi_worker_decorate
def calc_stop_signal_DV(df):
    """ Calculate dv for stop signal task. Common states like rt, correct and
    DDM parameters are calculated on go trials only
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice" and SS_trial_type == "go"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and ((SS_trial_type == "stop") or (SS_trial_type == "go" and rt != -1))').reset_index()
    dvs = calc_common_stats(df.query('SS_trial_type == "go"'))
    dvs = {'go_' + key: dvs[key] for key in list(dvs.keys())}
    dvs['SSRT'] = df.query('SS_trial_type == "go"')['rt'].median()-df['SS_delay'].median()
    dvs['stop_success'] = df.query('SS_trial_type == "stop"')['stopped'].mean()
    dvs['stop_avg_rt'] = df.query('SS_trial_type == "stop" and rt > 0')['rt'].median()
    dvs['missed_percent'] = missed_percent
    description = """ SSRT calculated as the difference between median go RT
    and median SSD. Missed percent calculated on go trials only.
    """
    return dvs, description

@multi_worker_decorate
def calc_threebytwo_DV(df):
    """ Calculate dv for 3 by 2 task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1').reset_index()
    dvs = calc_common_stats(df)
    dvs['cue_switch_cost'] = df.query('task_switch == "stay"').groupby('cue_switch')['rt'].median().diff()['switch']
    dvs['task_switch_cost'] = df.groupby(df['task_switch'].map(lambda x: 'switch' in x))['rt'].median().diff()[True]
    dvs['task_inhibition_of_return'] =  df[['switch' in x for x in df['task_switch']]].groupby('task_switch')['rt'].median().diff()['switch_old']
    dvs['missed_percent'] = missed_percent
    description = """ Task switch cost defined as rt difference between task "stay" trials
    and both task "switch_new" and "switch_old" trials. Cue Switch cost is defined only on 
    task stay trials. Inhibition of return is defined as the difference in reaction time between
    task "switch_old" and task "switch_new" trials. Positive values indicate higher RTs (cost) for
    task switches, cue switches and switch_old
    """
    return dvs, description



@multi_worker_decorate
def calc_TOL_DV(df):
    df = df.query('exp_stage == "test" and rt != -1').reset_index()
    dvs = {}
    # When they got it correct, did they make the minimum number of moves?
    dvs['num_optimal_solutions'] =  numpy.sum(df.query('correct == 1')[['num_moves_made','min_moves']].diff(axis = 1)['min_moves']==0)
    # how long did it take to make the first move?    
    dvs['planning_time'] = df.query('num_moves_made == 1 and trial_id == "to_hand"')['rt'].median()
    # how long did it take on average to take an action    
    dvs['avg_move_time'] = df.query('trial_id in ["to_hand", "to_board"]')['rt'].median()
    # how many moves were made overall
    dvs['total_moves'] = numpy.sum(df.groupby('problem_id')['num_moves_made'].max())
    dvs['num_correct'] = numpy.sum(df['correct']==1)
    description = 'many dependent variables related to tower of london performance'
    return dvs, description
    
    
    
@multi_worker_decorate
def calc_two_stage_decision_DV(df):
    """ Calculate dv for choice reaction time: Accuracy and average reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['trial_id']=="incomplete_trial").mean()
    df = df.query('exp_stage != "practice" and trial_id == "complete_trial"').reset_index()
    rs = smf.glm(formula = 'switch ~ feedback_last * stage_transition_last', data = df, family = sm.families.Binomial()).fit()
    rs.summary()
    dvs = {}
    dvs['avg_rt'] = numpy.mean(df[['rt_first','rt_second']].mean())
    dvs['model_free'] = rs.params['feedback_last']
    dvs['model_based'] = rs.params['feedback_last:stage_transition_last[T.infrequent]']
    dvs['missed_percent'] = missed_percent
    description = 'standard'  
    return dvs, description
    
    
@multi_worker_decorate
def calc_generic_dv(df):
    """ Calculate dv for choice reaction time: Accuracy and average reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1').reset_index()
    dvs = calc_common_stats(df)
    dvs['missed_percent'] = missed_percent
    description = 'standard'  
    return dvs, description
    
    