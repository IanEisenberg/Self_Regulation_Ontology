from os import path
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp
import seaborn as sns
from selfregulation.utils.utils import get_behav_data
from selfregulation.utils.data_preparation_utils import drop_vars


DV_df = get_behav_data(file = 'variables_exhaustive.csv')
subset = drop_vars(DV_df, saved_vars = ['\.std_rt$','\.avg_rt$','\.acc$'])

ddm_tasks = np.unique([c.split('.')[0] for c in DV_df.filter(regex = 'EZ').columns])
ddm_regex = '(' + '|'.join(ddm_tasks) + ')'

drift = subset.filter(regex = ddm_regex + '.*(drift|\.(avg|std)_rt$|\.acc)')
thresh = subset.filter(regex = ddm_regex + '.*(thresh|\.(avg|std)_rt$|\.acc)')
non_decision = subset.filter(regex = ddm_regex + '.*(non_decision|\.(avg|std)_rt$|\.acc)')

# ************************   
# helper functions
# ************************   
def reorder_cols(param, df):
    # sort by variable
    column_order = np.argsort([c.replace('EZ','hddm') for c in df.columns])
    df = df.iloc[:,column_order]
    # move ddm parameters to beginning
    columns = df.columns
    column_order = np.append(np.where([param in c for c in columns])[0],np.where([param not in c for c in columns])[0])
    df = df.iloc[:,column_order]
    return df
# ************************   
# ddm exploration
# ************************   
#ddm correlations across all subject/experiments
DDM_params = {}
for method in ['hddm','EZ']:
    for param in ['drift','thresh','non_decision']:
        DDM_params['%s_%s' % (method, param)] = DV_df.filter(regex = '\.%s_%s$' % (method, param)).as_matrix().flatten()
DDM_params = pd.DataFrame(DDM_params)
    
# correlate non decision with accuracy and rt
for param in ['drift','non_decision','thresh']:
    param_rt = DV_df.filter(regex = ddm_regex + '.*(hddm_%s$|(go|avg)_rt$)' % param)    
    correlations = np.diag(param_rt.corr(),1)[0::2]
    print('%s correlations with RT: %s' % (param, np.mean(correlations)))
    param_acc = DV_df.filter(regex = ddm_regex + '.*(hddm_%s$|(go_acc|\.acc$))' % param)    
    correlations = np.diag(param_acc.corr(),1)[0::2]
    print('%s correlations with Acc: %s' % (param, np.mean(correlations)))

# generate figs
figs = {}
for param, df in [('drift', drift), ('thresh', thresh), ('non_decision', non_decision)]:
    param_task_correlations = []
    param_figs = []
    for task in ddm_tasks:
        task_subset = df.filter(regex = '^%s' % task)
        task_subset = reorder_cols(param, task_subset)
        corr_mat = task_subset.corr()
        fig = sns.plt.figure()
        sns.heatmap(corr_mat)
        param_figs.append(fig)
        sns.plt.close(fig)
    figs[param] = param_figs
    
# look at between method correlations
"""
task_correlations = {}
correlations = []
for c in task_subset.filter(regex = 'hddm').columns:
    c_ez = c.replace('hddm', 'EZ')
    try:
        correlations.append(corr_mat.loc[c,c_ez])
    except KeyError:
        print('Problem with %s' % c)    
mean_reliability = np.mean([np.mean(i) for i in task_correlations])
print('Correlation across measures: %s' % mean_reliability)
"""

# non decision correlations
sns.heatmap(DV_df.filter(regex = '\.(EZ|hddm)_non_decision$').corr())

# thresh correlations
sns.heatmap(DV_df.filter(regex = '\.(EZ|hddm)_thresh$').corr())

# droft correlations
sns.heatmap(DV_df.filter(regex = '\.(EZ|hddm)_drift$').corr())

#variability
ddm_std = DV_df.filter(regex = '\.(hddm|EZ)').std()
ddm_std=ddm_std.groupby([lambda x: 'EZ' in x,lambda x: ['non_decision','drift','thresh'][('drift' in x) + ('thresh' in x)*2]]).mean().reset_index()
ddm_std.columns = ['Routine','Param','STD']
ddm_std.replace({False: 'hddm', True: 'EZ'}, inplace = True)

# ddm parameters contrasts
EZnon_decision_contrasts = {x[0]: ttest_1samp(x[1].dropna(),0).statistic for x in DV_df.filter(regex = '\.[a-z]+.*EZ_non_decision$').iteritems()}
EZdrift_contrasts = {x[0]: ttest_1samp(x[1].dropna(),0).statistic for x in DV_df.filter(regex = '\.[a-z]+.*EZ_drift$').iteritems()}
EZthresh_contrasts = {x[0]: ttest_1samp(x[1].dropna(),0).statistic for x in DV_df.filter(regex = '\.[a-z]+.*EZ_thresh$').iteritems()}
print('Mean EZ Non Decision Contrast: %s' % np.mean(list(EZnon_decision_contrasts.values())))
print('Mean EZ Drift Contrast: %s' % np.mean(list(EZdrift_contrasts.values())))
print('Mean EZ Thresh Contrast: %s' % np.mean(list(EZthresh_contrasts.values())))


Hdrift_contrasts = {x[0]: ttest_1samp(x[1].dropna(),0).statistic for x in DV_df.filter(regex = '\.[a-z]+.*hddm_drift$').iteritems()}
Hthresh_contrasts = {x[0]: ttest_1samp(x[1].dropna(),0).statistic for x in DV_df.filter(regex = '\.[a-z]+.*hddm_thresh$').iteritems()}
print('Mean hddm Drift Contrast: %s' % np.mean(list(Hdrift_contrasts.values())))
print('Mean hddm Thresh Contrast: %s' % np.mean(list(Hthresh_contrasts.values())))


# ******************************
# Try out regression
# ******************************
import hddm 

df = pd.DataFrame.from_csv('../Data/Local/Discovery_10-14-2016/Individual_Measures/simon.csv')

#convert to hddm format
condition = 'condition'
response_col = 'correct'
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

# run hddm.

# no parameters allowed to change
depends_dict = {}
m = hddm.HDDM(data, depends_on=depends_dict)
# find a good starting point which helps with the convergence.
m.find_starting_values()
# start drawing 10000 samples and discarding 1000 as burn-in
m.sample(2500, burn=500)
dvs = {var: m.nodes_db.loc[m.nodes_db.index.str.contains(var + '_subj'),'mean'] for var in ['a', 'v', 't']}  

# drift changes based on condition
depends_dict = {'v': 'condition'}
m_condition = hddm.HDDM(data, depends_on=depends_dict)
# find a good starting point which helps with the convergence.
m_condition.find_starting_values()
# start drawing 10000 samples and discarding 1000 as burn-in
m_condition.sample(2500, burn=500)
dvs_condition = {var: m_condition.nodes_db.loc[m_condition.nodes_db.index.str.contains(var + '_subj'),'mean'] for var in ['a', 'v', 't']}  

# regression model
m_within = hddm.HDDMRegressor(data, "v ~ C(condition, Treatment('congruent'))")
# find a good starting point which helps with the convergence.
m_within.find_starting_values()
# start drawing 10000 samples and discarding 1000 as burn-in
m_within.sample(2500, burn=500)
dvs_within = {var: m_within.nodes_db.loc[m_within.nodes_db.index.str.contains(var + '.*_subj'),'mean'] for var in ['a', 'v', 't']}  

# plot regression model
v_congruent, v_incongruent = m_within.nodes_db.ix[["v_Intercept",
                                              "v_C(condition, Treatment('congruent'))[T.incongruent]"], 'node']
hddm.analyze.plot_posterior_nodes([v_congruent, v_incongruent])

# comparisons of ddm params
thresh_corr = np.corrcoef(dvs['a'].tolist(), dvs_condition['a'].tolist())[0,1]
print('thresholds are correlated r = %s' % thresh_corr)

non_decision_corr = np.corrcoef(dvs['t'].tolist(), dvs_condition['t'].tolist())[0,1]
print('non-decision are correlated r = %s' % non_decision_corr)

halfway = int(len(dvs_condition['v'])/2)
drift_corr = np.corrcoef(dvs['v'].tolist(), dvs_condition['v'].tolist()[0:halfway])[0,1]
print('drift are correlated r = %s for congruent' % drift_corr)

halfway = int(len(dvs_condition['v'])/2)
drift_corr = np.corrcoef(dvs['v'].tolist(), dvs_condition['v'].tolist()[halfway:])[0,1]
print('drift are correlated r = %s for incongruent' % drift_corr)

