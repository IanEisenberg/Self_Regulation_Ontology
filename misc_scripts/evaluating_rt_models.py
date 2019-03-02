import pandas as pd
import statsmodels.formula.api as smf
from selfregulation.utils.utils import get_behav_data
data = get_behav_data()
problem_subj = data.filter(regex='simon').isnull().iloc[:,0]
df = get_behav_data(file='Individual_Measures/simon.csv.gz')
df = df.query('exp_stage == "test" and rt==rt')

params = {}
for worker in df.worker_id.unique():
    if problem_subj.loc[worker] == True:
        continue
    subset = df.query('worker_id=="%s"' % worker)
    acc_contrast = subset.groupby('condition').correct.mean()
    acc_diff = rt_contrast['incongruent']-rt_contrast['congruent']
    subset = subset.query('correct == True')
    rs = smf.ols(formula = 'rt ~ C(condition, Sum)', data = subset).fit()
    params[worker] = rs.params.tolist()
    params[worker][1]*=-2
    rt_contrast = subset.groupby('condition').rt.median()
    diff = rt_contrast['incongruent']-rt_contrast['congruent']
    params[worker].append(diff)
    params[worker].append(acc_diff)
    
    
DVs = pd.DataFrame(params, index=['Intercept','model_diff', 'diff', 'acc_diff']).T


