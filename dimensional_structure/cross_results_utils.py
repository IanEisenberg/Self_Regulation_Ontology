from itertools import permutations
import numpy as np
from os import path
import pandas as pd
import pickle
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from dimensional_structure.prediction_utils import run_prediction

def run_cross_prediction(all_results, verbose=False, save=True):
    # within
    CV = KFold(10, shuffle=True)
    within_predictions = {}
    for name, predicting_set in all_results.items():
        within_predictions[name] = {}
        all_predictors = predicting_set.data
        for target, vals in all_predictors.iteritems():
            predictors = all_predictors.drop(target, axis=1)
            pipe = Pipeline(steps=[('scale', StandardScaler()),
                                   ('clf', RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)))])
            score = np.mean(cross_val_score(pipe, predictors, vals, cv=CV))
            within_predictions[name][target] = score
    # across
    across_predictions = {}
    for pname, tname in permutations(all_results.keys(),2):
        across_predictions['%s_to_%s' % (pname, tname)] = {}
        predictors = all_results[pname].data
        targets = all_results[tname].data
        for target, vals in targets.iteritems():
            pipe = Pipeline(steps=[('scale', StandardScaler()),
                                   ('clf', RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)))])
            score = np.mean(cross_val_score(pipe, predictors, vals, cv=CV))
            across_predictions['%s_to_%s' % (pname, tname)] [target] = score
    if save:
        save_loc = path.join(path.dirname(all_results['task'].get_output_dir()), 
                         'cross_prediction.pkl')
        pickle.dump({'within': within_predictions,
                     'across': across_predictions},
                    open(save_loc, 'wb'))
    
def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = pearsonr(df[r], df[c])[1]
    return pvalues

def FDR_correction(pval_df, alpha=.01):
    pvals = squareform(pval_df)
    sort_indices = np.argsort(pvals)
    sorted_pvals = pvals[sort_indices]
    thresholds = [i/len(sorted_pvals)*alpha for i in range(len(sorted_pvals))]
    highest_k = np.where(sorted_pvals<=thresholds)[0][-1]
    significant_values = np.zeros(len(pvals), dtype=np.int8)
    significant_values[sort_indices[:highest_k]]=1
    return pd.DataFrame(squareform(significant_values), 
                        columns=pval_df.columns[:],
                        index=pval_df.index[:])
    
def calc_survey_task_relationship(all_results, EFA=False, alpha=.01):
    def get_EFA_HCA(results, EFA):
        if EFA == False:
            return results.HCA.results['data']
        else:
            c = results.EFA.results['num_factors']
            return results.HCA.results['EFA%s_oblimin' % c]
    survey_order = get_EFA_HCA(all_results['survey'], EFA)['reorder_vec']
    task_order = get_EFA_HCA(all_results['task'], EFA)['reorder_vec']
    
    if EFA == False:
        all_data = pd.concat([all_results['task'].data.iloc[:, task_order], 
                              all_results['survey'].data.iloc[:, survey_order]], 
                            axis=1)
    else:
        all_data = pd.concat([all_results['task'].EFA.get_loading().T.iloc[:, task_order], 
                              all_results['survey'].EFA.get_loading().T.iloc[:, survey_order]], 
                            axis=1)
        
    pvals = calculate_pvalues(all_data)
    corrected_results = {}
    for alpha in [.05, .01]:
        corrected_results=FDR_correction(pvals, alpha)
    return corrected_results
