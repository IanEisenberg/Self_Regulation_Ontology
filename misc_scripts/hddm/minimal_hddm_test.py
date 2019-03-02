# -*- coding: utf-8 -*-

from expanalysis.experiments.ddm_utils import get_HDDM_fun, load_model
from selfregulation.utils.utils import get_behav_data
import hddm
import numpy as np

# test HDDM calculation from processed task
task = 'stroop'
df = get_behav_data(file='Individual_Measures/%s.csv.gz' % task)
df = df.query('worker_id in %s' % list(df.worker_id.unique()[0:15]))


fun = get_HDDM_fun(task, samples=20, burn=10, outfile = '/home/ian/tmp/stroop', parallel=True)
out = fun(df)

acc = df.groupby('worker_id').correct.mean()
rt = df.groupby('worker_id').rt.median()
for var in ['hddm_drift', 'hddm_thresh', 'hddm_non_decision']:
    ddm_vars = [out[k][var]['value'] for k in sorted(out.keys())]
    print(var)
    print('Correlation with Acc: ', np.corrcoef(acc,ddm_vars)[0,1])
    print('Correlation with RT: ', np.corrcoef(rt,ddm_vars)[0,1])



    
samples = 20; burn = 10; thin = 1; 
response_col = 'correct'
# massage data into HDDM format
response_col='correct'
# set up data
data = (df.loc[:,'rt']/1000).astype(float).to_frame()
data.insert(0, 'response', df[response_col].astype(float))

extra_cols = ['conflict_condition', 'switch']
for col in extra_cols:
    data.insert(0, col, df[col])
    
# add subject ids 
data.insert(0,'subj_idx', df['worker_id'])
# remove missed responses and extremely short response
data = data.query('rt > .05')
subj_ids = data.subj_idx.unique()
ids = {subj_ids[i]:int(i) for i in range(len(subj_ids))}
data.replace(subj_ids, [ids[i] for i in subj_ids],inplace = True)


# set HDDM params
burn = 1000
thin = 2
samples = 4000

# run a bunch of models

# basic HDDM
np.random.seed(5)
m_nothing =  hddm.HDDM(data)
# find a good starting point which helps with the convergence.
m_nothing.find_starting_values()
# start drawing 10000 samples and discarding 1000 as burn-in
m_nothing.sample(samples, burn=burn, thin=thin)

# Conflict Condition, effect coding
np.random.seed(5)
# separate regaressors for each variable
mc_effect = hddm.models.HDDMRegressor(data, 'v ~ C(conflict_condition, Sum)', group_only_regressors=False)
mc_effect .find_starting_values()
# start drawing 10000 samples and discarding 1000 as burn-in
mc_effect .sample(samples, burn=burn, thin=thin)

# Conflict Condition, dummy coding
np.random.seed(5)
# separate regaressors for each variable
mc = hddm.models.HDDMRegressor(data, 'v ~ C(conflict_condition, Treatment("neutral"))', group_only_regressors=False)
mc.find_starting_values()
# start drawing 10000 samples and discarding 1000 as burn-in
mc.sample(samples, burn=burn, thin=thin)

# Switch Condition, effect coding
np.random.seed(5)
# separate regaressors for each variable
ms_effect = hddm.models.HDDMRegressor(data, 'v ~ C(switch, Sum)', group_only_regressors=False)
ms_effect .find_starting_values()
# start drawing 10000 samples and discarding 1000 as burn-in
ms_effect .sample(samples, burn=burn, thin=thin)

# Switch Condition, dummy coding
np.random.seed(5)
# separate regaressors for each variable
ms = hddm.models.HDDMRegressor(data, 'v ~ C(switch)', group_only_regressors=False)
ms.find_starting_values()
# start drawing 10000 samples and discarding 1000 as burn-in
ms.sample(samples, burn=burn, thin=thin)

# both, effect coding
np.random.seed(5)
# effect coding
regressor = 'C(' + ', Sum)+C('.join(extra_cols) + ', Sum)'
m2_effect = hddm.models.HDDMRegressor(data, 'v ~%s' % regressor, group_only_regressors=False)
m2_effect.find_starting_values()
# start drawing 10000 samples and discarding 1000 as burn-in
m2_effect.sample(samples, burn=burn, thin=thin)

# both, dummy coding
np.random.seed(5)
# dummy coding
regressor = 'C(' + ')+C('.join(extra_cols) + ')'
m2 = hddm.models.HDDMRegressor(data, 'v ~%s' % regressor, group_only_regressors=False)
m2.find_starting_values()
# start drawing 10000 samples and discarding 1000 as burn-in
m2.sample(samples, burn=burn, thin=thin)

# posterior predictive check: http://ski.clps.brown.edu/hddm_docs/tutorial_post_pred.html#summary-statistics-relating-to-outside-variables
# simulate data from a model
ppc_data = hddm.utils.post_pred_gen(m2_effect)
ppc_compare = hddm.utils.post_pred_stats(data, ppc_data)






def not_regex(txt):
    return '^((?!%s).)*$' % txt

ms.nodes_db.filter(regex='v',axis=0)['mean'].filter(regex=not_regex('subj'), axis=0)
ms_effect.nodes_db.filter(regex='v',axis=0)['mean'].filter(regex=not_regex('subj'), axis=0)


np.corrcoef(ms.nodes_db.filter(regex='v',axis=0)['mean'].filter(regex='switch.*subj',axis=0).tolist(),ms_effect.nodes_db.filter(regex='v',axis=0)['mean'].filter(regex='stay.*subj',axis=0).tolist())


np.corrcoef(m2_effect.nodes_db.filter(regex='^a_subj', axis=0)['mean'].tolist(),
            data.groupby('subj_idx').rt.median())


np.corrcoef(mc_effect.nodes_db.filter(regex='Intercept_subj', axis=0)['mean'].tolist(),
            m_nothing.nodes_db.filter(regex='^v_subj', axis=0)['mean'].tolist())

























