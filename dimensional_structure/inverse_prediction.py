import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
import seaborn as sns

from dimensional_structure.prediction_utils import run_prediction
from dimensional_structure.utils import abs_pdist, hierarchical_cluster
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_recent_dataset

dataset = get_recent_dataset()
results = load_results(dataset)

demographics = results['survey'].DA.get_scores()
result = results['survey']
c = result.EFA.results['num_factors']

# predict data using demographics to give a "demographic fingerprint"
out = run_prediction(demographics, result.data, 
               '/home/ian/tmp', classifier='ridge',
               binarize=False, save=False)
demog_importances = pd.DataFrame({k:v[0] for k,v in out.importances.items()},
                            index=out.behavdata.columns).T


# predicting other things
raw_predictions = result.load_prediction_object(EFA=False, classifier='ridge')['data']
EFA_predictions = result.load_prediction_object(EFA=True, classifier='ridge')['data']

importances = {}
for outcome, prediction in raw_predictions.items():
    print(prediction['scores_cv'])
    print('EFA: ' + str(EFA_predictions[outcome]['scores_cv'][0]['R2']))
    imp = prediction['importances'][0]
    predvars = prediction['predvars']
    importances[outcome] = {p:i for p,i in zip(predvars, imp)}


prediction_fingerprints = pd.DataFrame(importances)
loading = result.EFA.get_loading()

# get clusterings
result_cluster = result.HCA.results['EFA%s_oblimin' % c]
demog_cluster = hierarchical_cluster(demog_importances, pdist_kws={'metric': 'abscorrelation'})

# combine 
combined = prediction_fingerprints.join(loading)
combined_cluster = hierarchical_cluster(combined, pdist_kws={'metric': 'abscorrelation'})

# get distances
raw_dist = abs_pdist(result.data.T)
demog_dist = squareform(demog_cluster['distance_df'])
loading_dist = squareform(result_cluster['distance_df'])
combined_dist = squareform(combined_cluster['distance_df'])

dists = pd.DataFrame(np.vstack([raw_dist, 
                                demog_dist, 
                                loading_dist, 
                                combined_dist]),
                     index=['raw', 'demog', 'loading', 'combined'])
sns.heatmap(dists.T.corr())

