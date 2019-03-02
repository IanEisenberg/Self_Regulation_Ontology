# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 13:57:56 2016

@author: ian
"""
import pandas
import numpy 
import hddm

def EZ_diffusion(df, condition = None):
    assert 'correct' in df.columns, 'Could not calculate EZ DDM'
    df = df.copy()
    # convert reaction time to seconds to match with HDDM
    df['rt'] = df['rt']/1000
    # ensure there are no missed responses or extremely short responses (to fit with EZ)
    df = df.query('rt > .01')
    # convert any perfect accuracies to .95
    
    EZ_dvs = {}
    # calculate EZ params for each condition
    if condition:
        conditions = df[condition].unique()
        conditions = conditions[~pandas.isnull(conditions)]
        for c in conditions:
            subset = df[df[condition] == c]
            pc = subset['correct'].mean()
            # edge case correct
            if pc == 1:
                pc = 1-(1.0/(2*len(subset)))
            vrt = numpy.var(subset.query('correct == True')['rt'])
            mrt = numpy.mean(subset.query('correct == True')['rt'])
            try:
                drift, thresh, non_dec = hddm.utils.EZ(pc, vrt, mrt)
                EZ_dvs['EZ_drift_' + c] = {'value': drift, 'valence': 'Pos'}
                EZ_dvs['EZ_thresh_' + c] = {'value': thresh, 'valence': 'Pos'}
                EZ_dvs['EZ_non_decision_' + c] = {'value': non_dec, 'valence': 'Neg'}
            except ValueError:
                continue
    else:
        # calculate EZ params
        try:
            pc = df['correct'].mean()
            # edge case correct
            if pc == 1:
                pc = 1-(1.0/(2*len(df)))
            vrt = numpy.var(df.query('correct == True')['rt'])
            mrt = numpy.mean(df.query('correct == True')['rt'])
            drift, thresh, non_dec = hddm.utils.EZ(pc, vrt, mrt)
            EZ_dvs['EZ_drift'] = {'value': drift, 'valence': 'Pos'}
            EZ_dvs['EZ_thresh'] = {'value': thresh, 'valence': 'Pos'}
            EZ_dvs['EZ_non_decision'] = {'value': non_dec, 'valence': 'Neg'}
        except ValueError:
            return {}
    return EZ_dvs

def fit_HDDM(df, response_col = 'correct', condition = None, fixed= ['t','a'], estimate_task_vars = True):
    """ fit_HDDM is a helper function to run hddm analyses.
    :df: that dataframe to perform hddm analyses on
    :response_col: a column of correct/incorrect values
    :condition: optional, categoricla variable to use to separately calculated ddm parameters
    :fixed: a list of ddm parameters (e.g. ['a', 't']) where 'a' is threshold, 'v' is drift and 't' is non-decision time
        to keep fixed when using the optional condition argument
    :estimate_task_vars: bool, if True estimate DDM vars using the entire task in addition to conditional vars
    """
    assert estimate_task_vars or condition != None, "Condition must be defined or estimate_task_vars must be set to true"
    variable_conversion = {'a': ('thresh', 'Pos'), 'v': ('drift', 'Pos'), 't': ('non_decision', 'NA')}
    # set up condition variables
    if condition:
        condition_vars = [var for var in ['a','v','t'] if var not in fixed]
        depends_dict = {var: 'condition' for var in condition_vars}
    else:
        condition_vars = []
        depends_dict = {}
    # set up data
    data = (df.loc[:,'rt']/1000).astype(float).to_frame()
    data.insert(0, 'response', df[response_col].astype(float))
    if condition:
        data.insert(0, 'condition', df[condition])
        conditions = [i for i in data.condition.unique() if i]
        
    # add subject ids 
    data.insert(0,'subj_idx', df['worker_id'])
    # remove missed responses and extremely short response
    data = data.query('rt > .01')
    subj_ids = data.subj_idx.unique()
    ids = {subj_ids[i]:int(i) for i in range(len(subj_ids))}
    data.replace(subj_ids, [ids[i] for i in subj_ids],inplace = True)
    
    # extract dvs
    group_dvs = {}
    dvs = {}
    if estimate_task_vars:
        # run hddm
        m = hddm.HDDM(data)
        # find a good starting point which helps with the convergence.
        m.find_starting_values()
        # start drawing 10000 samples and discarding 1000 as burn-in
        m.sample(2500, burn=500)
        dvs = {var: m.nodes_db.loc[m.nodes_db.index.str.contains(var + '_subj'),'mean'] for var in ['a', 'v', 't']}  
    
    if len(depends_dict) > 0:
        # run hddm
        m_depends = hddm.HDDM(data, depends_on=depends_dict)
        # find a good starting point which helps with the convergence.
        m_depends.find_starting_values()
        # start drawing 10000 samples and discarding 1000 as burn-in
        m_depends.sample(2500, burn=500)
    for var in depends_dict.keys():
        dvs[var + '_conditions'] = m_depends.nodes_db.loc[m_depends.nodes_db.index.str.contains(var + '_subj'),'mean']
    
    for i,subj in enumerate(subj_ids):
        group_dvs[subj] = {}
        hddm_vals = {}
        for var in ['a','v','t']:
            var_name, var_valence = variable_conversion[var]
            if var in list(dvs.keys()):
                hddm_vals.update({'hddm_' + var_name: {'value': dvs[var][i], 'valence': var_valence}})
            if var in condition_vars:
                for c in conditions:
                    try:
                        hddm_vals.update({'hddm_' + var_name + '_' + c: {'value': dvs[var + '_conditions'].filter(regex = '\(' + c + '\)', axis = 0)[i], 'valence': var_valence}})
                    except IndexError:
                        print('%s failed on condition %s for var: %s' % (subj, c, var_name))                
        group_dvs[subj].update(hddm_vals)
    return group_dvs
    
def get_post_error_slow(df):
    """df should only be one subject's trials where each row is a different trial. Must have at least 4 suitable trials
    to calculate post-error slowing
    """
    index = [(j-1, j+1) for j in [df.index.get_loc(i) for i in df.query('correct == False and rt != -1').index] if j not in [0,len(df)-1]]
    post_error_delta = []
    for i,j in index:
        pre_rt = df.iloc[i]['rt']
        post_rt = df.iloc[j]['rt']
        if pre_rt != -1 and post_rt != -1 and df.iloc[i]['correct'] and df.iloc[j]['correct']:
            post_error_delta.append(post_rt - pre_rt) 
    if len(post_error_delta) >= 4:
        return numpy.mean(post_error_delta)
    else:
        return numpy.nan

    
def group_decorate(group_fun = None):
    """ Group decorate is a wrapper for multi_worker_decorate to pass an optional group level
    DV function
    :group_fun: a function to apply to the entire group that returns a dictionary with DVs
    for each subject (i.e. fit_HDDM)
    """
    def multi_worker_decorate(fun):
        """Decorator to ensure that dv functions (i.e. calc_stroop_DV) have only one worker
        :func: function to apply to each worker individuals
        """
        def multi_worker_wrap(group_df, use_check = False, use_group_fun = True):
            exps = group_df.experiment_exp_id.unique()
            group_dvs = {}
            if len(group_df) == 0:
                return group_dvs, ''
            if len(exps) > 1:
                print('Error - More than one experiment found in dataframe. Exps found were: %s' % exps)
                return group_dvs, ''
            # remove practice trials
            group_df = group_df.query('exp_stage != "practice"')
            # remove workers who haven't passed some check
            if 'passed_check' in group_df.columns and use_check:
                group_df = group_df[group_df['passed_check']]
            # apply group func if it exists
            if group_fun and use_group_fun:
                group_dvs = group_fun(group_df)
            # apply function on individuals
            for worker in pandas.unique(group_df['worker_id']):
                df = group_df.query('worker_id == "%s"' %worker)
                dvs = group_dvs.get(worker, {})
                worker_dvs, description = fun(df, dvs)
                group_dvs[worker] = worker_dvs
            return group_dvs, description
        return multi_worker_wrap
    return multi_worker_decorate

def threebytwo_HDDM(df):
    group_dvs = fit_HDDM(df)
    for CTI in df.CTI.unique():
        CTI_df = df.query('CTI == %s' % CTI)
        CTI_df.loc[:,'cue_switch_binary'] = CTI_df.cue_switch.map(lambda x: ['cue_stay','cue_switch'][x!='stay'])
        CTI_df.loc[:,'task_switch_binary'] = CTI_df.task_switch.map(lambda x: ['task_stay','task_switch'][x!='stay'])
        
        cue_switch = fit_HDDM(CTI_df.query('cue_switch in ["switch","stay"]'), condition = 'cue_switch_binary', estimate_task_vars = False)
        task_switch = fit_HDDM(CTI_df, condition = 'task_switch_binary', estimate_task_vars = False)
        for key in cue_switch.keys():   
            if key not in group_dvs.keys():
                group_dvs[key] = {}
            cue_dvs = cue_switch[key]
            task_dvs = task_switch[key]
            for ckey in list(cue_dvs.keys()):
                cue_dvs[ckey + '_%s' % CTI] = cue_dvs.pop(ckey)
            for tkey in list(task_dvs.keys()):
                task_dvs[tkey + '_%s' % CTI] = task_dvs.pop(tkey)
            group_dvs[key].update(cue_dvs)
            group_dvs[key].update(task_dvs)
    return group_dvs
@group_decorate(group_fun = threebytwo_HDDM)
def calc_threebytwo_DV(df, dvs = {}):
    """ Calculate dv for 3 by 2 task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # add columns for shift effect
    df.insert(0,'correct_shift', df.correct.shift(1))
    df.insert(0, 'task_switch_shift', df.task_switch.shift(1))

    # post error slowing
    post_error_slowing = get_post_error_slow(df)
    
    missed_percent = (df['rt']==-1).mean()
    df = df.query('rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True and correct_shift == True').reset_index(drop = True)
    # make dataframe for EZ_DDM comparisons
    df_EZ = df.query('correct_shift == 1 and rt > .01')
    # convert reaction time to seconds to match with HDDM
    df_EZ = df_EZ.rename(columns = {'rt':'old_rt'})
    df_EZ['rt'] = df_EZ['old_rt']/1000
    # get DDM across all trials
    ez_alltrials = EZ_diffusion(df)
    dvs.update(ez_alltrials)
    
    # Calculate basic statistics - accuracy, RT and error RT
    dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'}
    dvs['avg_rt_error'] = {'value':  df.query('correct == False').rt.median(), 'valence': 'NA'}
    dvs['std_rt_error'] = {'value':  df.query('correct == False').rt.std(), 'valence': 'NA'}
    dvs['avg_rt'] = {'value':  df_correct.rt.median(), 'valence': 'Neg'}
    dvs['std_rt'] = {'value':  df_correct.rt.std(), 'valence': 'NA'}
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
    dvs['post_error_slowing'] = {'value':  post_error_slowing, 'valence': 'Pos'}
    
    # calculate task set inhibition (difference between CBC and ABC)
    selection = ['switch' in x.task_switch and 'stay' not in x.task_switch_shift for i,x in df_correct.iterrows()]
    task_inhibition_contrast =  df_correct[selection].groupby(['CTI','task','task_switch']).rt.median().diff()

    #switch costs
    for CTI in df.CTI.unique():
        dvs['task_inhibition_%s' % CTI] = {'value':  task_inhibition_contrast.reset_index().query('task_switch == "switch_old" and CTI == %s' % CTI).mean().rt, 'valence': 'Neg'} 
        CTI_df = df_correct.query('CTI == %s' % CTI)
        CTI_df_EZ = df_EZ.query('CTI == %s' % CTI)
        dvs['cue_switch_cost_rt_%s' % CTI] = {'value':  CTI_df.groupby('cue_switch')['rt'].median().diff()['switch'], 'valence': 'Neg'} 
        task_switch_cost = CTI_df.groupby(CTI_df['task_switch'].map(lambda x: 'switch' in x)).rt.median().diff()[True]
        dvs['task_switch_cost_rt_%s' % CTI] = {'value':  task_switch_cost - dvs['cue_switch_cost_rt_%s' % CTI]['value'], 'valence': 'Neg'} 
        
        # DDM equivalents
        
        # calculate EZ_diffusion for cue switchin and task switching
        # cue switch
        for c in ['switch', 'stay']:
            try:
                subset = CTI_df_EZ[CTI_df_EZ['cue_switch'] == c]
                pc = subset['correct'].mean()
                # edge case correct
                if pc == 1:
                    pc = 1-(1.0/(2*len(subset)))
                vrt = numpy.var(subset.query('correct == True')['rt'])
                mrt = numpy.mean(subset.query('correct == True')['rt'])
                drift, thresh, non_dec = hddm.utils.EZ(pc, vrt, mrt)
                dvs['EZ_drift_cue_' + c + '_%s' % CTI] = {'value': drift, 'valence': 'Pos'}
                dvs['EZ_thresh_cue_' + c + '_%s' % CTI] = {'value': thresh, 'valence': 'NA'}
                dvs['EZ_non_decision_cue_' + c + '_%s' % CTI] = {'value': non_dec, 'valence': 'Neg'}
            except ValueError:
                continue
            
        # task switch
        for c in [['stay'],['switch_old','switch_new']]:
            try:
                subset = CTI_df_EZ[CTI_df_EZ['task_switch'].isin(c)]
                pc = subset['correct'].mean()
                # edge case correct
                if pc == 1:
                    pc = 1-(1.0/(2*len(subset)))
                vrt = numpy.var(subset.query('correct == True')['rt'])
                mrt = numpy.mean(subset.query('correct == True')['rt'])
                drift, thresh, non_dec = hddm.utils.EZ(pc, vrt, mrt)
                dvs['EZ_drift_task_' + c[0].split('_')[0] + '_%s' % CTI] = {'value': drift, 'valence': 'Pos'}
                dvs['EZ_thresh_task_' + c[0].split('_')[0] + '_%s' % CTI] = {'value': thresh, 'valence': 'NA'}
                dvs['EZ_non_decision_task_' + c[0].split('_')[0] + '_%s' % CTI] = {'value': non_dec, 'valence': 'Neg'}
            except ValueError:
                continue
            
        param_valence = {'drift': 'Pos', 'thresh': 'Pos', 'non_decision': 'NA'}
        for param in ['drift','thresh','non_decision']:
            if set(['EZ_' + param + '_cue_switch'  + '_%s' % CTI, 'EZ_' + param + '_cue_stay' + '_%s' % CTI]) <= set(dvs.keys()):
                dvs['cue_switch_cost_EZ_' + param + '_%s' % CTI] = {'value':  dvs['EZ_' + param + '_cue_switch' + '_%s' % CTI]['value'] - dvs['EZ_' + param + '_cue_stay' + '_%s' % CTI]['value'], 'valence': param_valence[param]}
                if set(['EZ_' + param + '_task_switch' + '_%s' % CTI, 'EZ_' + param + '_task_stay' + '_%s' % CTI]) <= set(dvs.keys()):
                    dvs['task_switch_cost_EZ_' + param + '_%s' % CTI] = {'value':  dvs['EZ_' + param + '_task_switch' + '_%s' % CTI]['value'] - dvs['EZ_' + param + '_task_stay' + '_%s' % CTI]['value'] - dvs['cue_switch_cost_EZ_' + param + '_%s' % CTI]['value'], 'valence': param_valence[param]}
        for param in ['drift','thresh','non_decision']:
            if set(['hddm_' + param + '_cue_switch' + '_%s' % CTI, 'hddm_' + param + '_cue_stay' + '_%s' % CTI]) <= set(dvs.keys()):
                dvs['cue_switch_cost_hddm_' + param + '_%s' % CTI] = {'value':  dvs['hddm_' + param + '_cue_switch' + '_%s' % CTI]['value'] - dvs['hddm_' + param + '_cue_stay' + '_%s' % CTI]['value'], 'valence': param_valence[param]}
                if set([ 'hddm_' + param + '_task_switch' + '_%s' % CTI, 'hddm_' + param + '_task_stay' + '_%s' % CTI]) <= set(dvs.keys()):
                    dvs['task_switch_cost_hddm_' + param + '_%s' % CTI] = {'value':  dvs['hddm_' + param + '_task_switch' + '_%s' % CTI]['value'] - dvs['hddm_' + param + '_task_stay' + '_%s' % CTI]['value']  - dvs['cue_switch_cost_hddm_' + param + '_%s' % CTI]['value'], 'valence': param_valence[param]}
             
    description = """ Task switch cost defined as rt difference between task "stay" trials
    and both task "switch_new" and "switch_old" trials. Cue Switch cost is defined only on 
    task stay trials. Inhibition of return is defined as the difference in reaction time between
    task "switch_old" and task "switch_new" trials (ABC vs CBC). The value is the mean over the three tasks. 
    Positive values indicate higher RTs (cost) for
    task switches, cue switches and switch_old
    """
    return dvs, description

 
df = pandas.DataFrame.from_csv('../Data/Local/Discovery_11-13-2016/Individual_Measures/threebytwo.csv')
df = df.query('exp_stage != "practice"')
DVs, description = calc_threebytwo_DV(df, use_group_fun = True, use_check = True)
#transform DVs into nicer format
for key,val in DVs.items():
    for subj_key in val.keys():
        val[subj_key]=val[subj_key]['value']
DVs = pandas.DataFrame.from_dict(DVs).T
DVs.to_csv('../Data/Local/threebytwo_DVs.csv')