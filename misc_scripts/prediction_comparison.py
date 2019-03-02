import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from dimensional_structure.prediction_utils import run_prediction
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_recent_dataset, get_behav_data

results = load_results(get_recent_dataset())
data = get_behav_data(file='variables_exhaustive.csv')

# get demographics
full_demog = results['task'].DA.get_scores()
full_demog.columns = ['full_' + c for c in full_demog.columns]
demog = pd.read_csv('/home/ian/Downloads/demog_fa_scores_t1.csv', index_col=0)

# get predictors
ddm_factors = pd.read_csv('/home/ian/Downloads/ez_t1_fa_3_scores.csv', index_col=0)
ontology_factors = results['task'].EFA.get_scores()
ontology_ddm_factors = ontology_factors[['Speeded IP', 'Caution', 'Perc / Resp']]

#
# compare demographics
diff = pd.DataFrame(demog.values - full_demog.loc[demog.index].values,
                    index=demog.index, columns=demog.columns)
corr = demog.join(full_demog).corr().iloc[:len(demog.columns), 
                                         len(demog.columns):]

# EZ vars 
EZ_vars = data.filter(regex='EZ_(non_decision|drift|thresh)$')
hddm_vars = data.filter(regex='hddm_(non_decision|drift|thresh)$')
EZ_lookup = [i.replace('hddm', 'EZ') for i in hddm_vars.columns]
overlap = sorted(list(set(EZ_lookup) & set(EZ_vars)))
hddm_lookup = [i.replace('EZ', 'hddm') for i in overlap]
combined = pd.concat([data[overlap], data[hddm_lookup]], axis=1)
corr = combined.corr().iloc[:len(overlap), len(overlap):]
diag = np.diag(corr)

# reproduce zeynep analysis "exactly"
ddm_out = run_prediction(ddm_factors, demog, save=False, classifier=LinearRegression())
ddm_scores = {k: v[0]['R2'] for k,v in ddm_out.scores.items()}

# predictions
scores= {'zeynep_demo': {},
         'full_demo': {},
         'subset_demo': {}}

# use zeyneps demo
targets = demog
ddm_out = run_prediction(ddm_factors, targets, save=False, classifier='ridge')
ontology_out = run_prediction(ontology_ddm_factors.loc[ddm_factors.index], 
                              targets, save=False, classifier='ridge')

scores['zeynep_demo']['ddm'] = {k: v[0]['R2'] for k,v in ddm_out.scores.items()}
scores['zeynep_demo']['ddm_ont']  = {k: v[0]['R2'] for k,v in ontology_out.scores.items()}
scores['zeynep_demo']['diff'] = {}
for key in scores['zeynep_demo']['ddm'].keys():
    scores['zeynep_demo']['diff'][key] = scores['zeynep_demo']['ddm'][key] \
                                    - scores['zeynep_demo']['ddm_ont'][key]

# use subset demo
targets = full_demog.loc[demog.index]
ddm_out = run_prediction(ddm_factors, targets, save=False, classifier='ridge')
ontology_out = run_prediction(ontology_ddm_factors.loc[ddm_factors.index], 
                              targets, save=False, classifier='ridge')

scores['subset_demo']['ddm'] = {k: v[0]['R2'] for k,v in ddm_out.scores.items()}
scores['subset_demo']['ddm_ont']  = {k: v[0]['R2'] for k,v in ontology_out.scores.items()}
scores['subset_demo']['diff'] = {}
for key in scores['subset_demo']['ddm'].keys():
    scores['subset_demo']['diff'][key] = scores['subset_demo']['ddm'][key] \
                                    - scores['subset_demo']['ddm_ont'][key]
# use full demo
targets = full_demog
ontology_out = run_prediction(ontology_ddm_factors, 
                              targets, save=False, classifier='ridge')

scores['full_demo']['ddm_ont']  = {k: v[0]['R2'] for k,v in ontology_out.scores.items()}



