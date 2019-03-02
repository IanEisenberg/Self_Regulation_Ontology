"""
analysis/experiments/survey_processing.py: part of expfactory package
functions for automatically cleaning and manipulating surveys
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

"""
DV functions
"""

@multi_worker_decorate
def calc_bis11_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = {
        'attention': [6,10,12,21,29],
        'cognitive_stability': [7,25,27],
        'motor': [3,4,5,18,20,23,26],
        'perseverance': [17,22,24,31],
        'self-control': [2,8,9,13,14,15],
        'cognitive_complexity': [11,16,19,28,30],
    }
    DVs = {}
    firstorder = {}
    for score,subset in list(scores.items()):
         firstorder[score] = df.query('question_num in %s' % subset).numeric_response.sum()
    DVs['Attentional'] = firstorder['attention'] + firstorder['cognitive_stability']
    DVs['Motor'] = firstorder ['motor'] + firstorder['perseverance']
    DVs['Nonplanning'] = firstorder['self-control'] + firstorder['cognitive_complexity']
    DVs['first_order_factors'] = firstorder
    description = """
        Score for bis11. Higher values mean
        greater expression of that factor. "Attentional", "Motor" and "Nonplanning"
        are second-order factors, while the other 6 are first order factors.
    """
    return DVs,description

@multi_worker_decorate
def calc_bis_bas_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = {
        'BAS_drive': [4,10,13,22],
        'BAS_fun_seeking': [6,11,16,21],
        'BAS_reward_responsiveness': [5,8,15,19,24],
        'BIS': [3,9,14,17,20,23,25]

    }
    DVs = {}
    for score,subset in list(scores.items()):
         DVs[score] = df.query('question_num in %s' % subset).numeric_response.sum()
    description = """
        Score for bias/bas. Higher values mean
        greater expression of that factor. BAS: "behavioral approach system",
        BIS: "Behavioral Inhibition System"
    """
    return DVs,description

@multi_worker_decorate
def calc_brief_DV(df):
    DVs = {'self_control': df['response'].astype(float).sum()}
    description = """
        Grit level. Higher means more gritty
    """
    return DVs,description

@multi_worker_decorate
def calc_dickman_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = {
        'dysfunctional': [2,5,8,10,11,14,15,18,19,22,23,24],
        'functional': [3,4,6,7,9,12,13,16,17,20,21]

    }
    DVs = {}
    for score,subset in list(scores.items()):
         DVs[score] = df.query('question_num in %s' % subset).numeric_response.sum()
    description = """
        Score for all dickman impulsivity survey. Higher values mean
        greater expression of that factor. 
    """
    return DVs,description
    
@multi_worker_decorate
def calc_dospert_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = {
        'ethical': [7,10,11,17,30,31],
        'financial': [4,5,9,13,15,19],
        'health/safety': [6,16,18,21,24,27],
        'recreational': [3,12,14,20,25,26],
        'social': [2,8,22,23,28,29]

    }
    DVs = {}
    for score,subset in list(scores.items()):
         DVs[score] = df.query('question_num in %s' % subset).numeric_response.sum()
    description = """
        Score for all dospert scales. Higher values mean
        greater expression of that factor. 
    """
    return DVs,description
    
@multi_worker_decorate
def calc_eating_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = {
        'cognitive_restraint': [3,12,13,16,17,19],
        'uncontrolled_eating': [2,5,6,8,9,10,14,15,18],
        'emotional_eating': [4,7,11]
    }
    DVs = {}
    for score,subset in list(scores.items()):
         raw_score = df.query('question_num in %s' % subset).numeric_response.sum()
         normalized_score = (raw_score-len(subset))/(len(subset)*3)*100
         DVs[score] = normalized_score
    description = """
        Score for three eating components. Higher values mean
        greater expression of that value
    """
    return DVs,description
    
@multi_worker_decorate
def calc_erq_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = {
        'reappraisal': [2,4,6,8,9,11],
        'suppression': [3,5,7,10]
    }
    DVs = {}
    for score,subset in list(scores.items()):
        DVs[score] = df.query('question_num in %s' % subset).numeric_response.mean()
    description = """
        Score for different emotion regulation strategies. Higher values mean
        greater expression of that strategy
    """
    return DVs,description
    
@multi_worker_decorate
def calc_five_facet_mindfulness_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = {
        'observe': [2,7,12,16,21,27,32,37],
        'describe': [3,8,13,17,23,28,33,38],
        'act_with_awareness': [6,9,14,19,23,29,35,39],
        'nonjudge': [4,11,15,18,26,31,36,39],
        'nonreact': [5,10,20,22,25,30,34]
    }
    DVs = {}
    for score,subset in list(scores.items()):
         DVs[score] = df.query('question_num in %s' % subset).numeric_response.sum()
    description = """
        Score for five factors mindfulness. Higher values mean
        greater expression of that value
    """
    return DVs,description

@multi_worker_decorate
def calc_future_time_perspective_DV(df):
    DVs = {'future_time_perspective': df['response'].astype(float).sum()}
    description = """
        Future time perspective (FTP) level. Higher means being more attentive/
        influenced by future states
    """
    return DVs,description
    
@multi_worker_decorate
def calc_grit_DV(df):
    DVs = {'grit': df['response'].astype(float).sum()}
    description = """
        Grit level. Higher means more gritty
    """
    return DVs,description

@multi_worker_decorate
def calc_i7_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = {
        'impulsiveness': [6,7,8,11,13,15,16,17,18,21,22,24,26,29,30,31],
        'venturesomeness': [2,3,4,5,9,10,12,14,19,20,23,25,27,28,32]
    }
    DVs = {}
    for score,subset in list(scores.items()):
         DVs[score] = df.query('question_num in %s' % subset).numeric_response.sum()
    description = """
        Score for i7. Higher values mean
        greater expression of that value. One question was removed from the original
        survey for venturesomeness: "Would you like to go pot-holing"
    """
    return DVs,description
    
@multi_worker_decorate
def calc_leisure_time_DV(df):
    DVs = {'activity_level': float(df.iloc[0]['response'])}
    description = """
        Exercise level. Higher means more exercise
    """
    return DVs,description

@multi_worker_decorate
def calc_maas_DV(df):
    DVs = {'mindfulness': df['response'].astype(float).mean()}
    description = """
        mindfulness level. Higher levels means higher levels of "dispositional mindfulness"
    """
    return DVs,description

@multi_worker_decorate
def calc_mpq_control_DV(df):
    DVs = {'control': df['response'].astype(float).sum()}
    description = """
        control level. High scorers on this scale describe themselves as:
            Reflective; cautious, careful, plodding; rational, 
            sensible, level-headed; liking to plan activities in detail.
    """
    return DVs,description

@multi_worker_decorate
def calc_SOC_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = {
        'elective_selection': list(range(2,14)),
        'loss-based_selection': list(range(14,26)),
        'optimization': list(range(26,38)),
        'compensation': list(range(38,50))
    }
    DVs = {}
    for score,subset in list(scores.items()):
        DVs[score] = df.query('question_num in %s' % subset).numeric_response.mean()
    description = """
        Score for five different personality measures. Higher values mean
        greater expression of that personality
    """
    return DVs,description
    
@multi_worker_decorate
def calc_SSRQ_DV(df):
    DVs = {'control': df['response'].astype(float).sum()}
    description = """
        control level. High scorers means higher level of endorsement
    """
    return DVs,description

@multi_worker_decorate
def calc_SSS_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = {
        'boredom_susceptibility': [3,6,8,9,16,25,28,32,35,40],
        'disinhibition': [2,13,14,26,30,31,33,34,36,37],
        'experience_seeking': [5,7,10,11,15,19,20,23,27,38],
        'thrill_adventure_seeking': [4,12,17,18,21,22,24,29,39,41]
    }
    DVs = {}
    for score,subset in list(scores.items()):
        DVs[score] = df.query('question_num in %s' % subset).numeric_response.mean()
    description = """
        Score for SSS-V. Higher values mean
        greater expression of that trait
    """
    return DVs,description
    
@multi_worker_decorate
def calc_ten_item_personality_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = {
        'extraversion': [3,8],
        'agreeableness': [4, 9],
        'conscientiousness': [5,10],
        'emotional_stability': [6,11],
        'openness': [7,12]
    }
    DVs = {}
    for score,subset in list(scores.items()):
        DVs[score] = df.query('question_num in %s' % subset).numeric_response.mean()
    description = """
        Score for five different personality measures. Higher values mean
        greater expression of that personality
    """
    return DVs,description

@multi_worker_decorate
def calc_theories_of_willpower_DV(df):
    DVs = {'endorse_limited_resource': df['response'].astype(float).sum()}
    description = """
        Higher values on this survey indicate a greater endorsement of a 
        "limited resource" theory of willpower
    """
    return DVs,description

@multi_worker_decorate
def calc_time_perspective_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = {
        'past_negative': [5,6,17,23,28,34,35,37,51,55],
        'present_hedonistic': [2,9,13,18,20,24,27,29,32,33,43,45,47,49,56],
        'future': [7,10,11,14,19,22,25,31,41,14,46,52,57],
        'past_positive': [3,8,11,16,21,26,30,42,50],
        'present_fatalistic': [4,15,36,38,39,40,48,53,54],

    }
    DVs = {}
    for score,subset in list(scores.items()):
        DVs[score] = df.query('question_num in %s' % subset).numeric_response.mean()
    description = """
        Score for five different time perspective factors. High values indicate 
        higher expression of that value
    """
    return DVs,description
    
@multi_worker_decorate
def calc_upps_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = {
        'negative_urgency': [3,8,13,18,23,30,35,40,45,51,54,59],
        'lack_of__premeditation': [2,7,12,17,22,29,34,39,44,49,56],
        'lack_of_perseverance': [5,10,15,20,25,28,33,38,43,48],
        'sensation_seeking': [4,9,14,19,24,27,32,37,42,47,52,57],
        'positive_urgency': [6,11,16,21,26,31,36,41,46,50,53,55,58,60]
    }
    DVs = {}
    for score,subset in list(scores.items()):
        DVs[score] = df.query('question_num in %s' % subset).numeric_response.sum()
    description = """
        Score for five different upps+p measures. Higher values mean
        greater expression of that factor
    """
    return DVs,description
    