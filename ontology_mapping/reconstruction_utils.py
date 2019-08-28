import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from selfregulation.utils.r_to_py_utils import psychFA

# *****************************************************************************  
#utils for loading reconstruction files
# *****************************************************************************  
def load_files(reconstruction_files, query_string=None):
    out = {}
    for f in reconstruction_files:
        tmp = pd.read_pickle(f)
        if query_string:
            tmp = tmp.query(query_string)
        name = f.split('-')[-1][:-4]
        out[name] = tmp
    return out

def update_files(old, new):
    for k, df in old.items():
        if k in new.keys():
            add = new.pop(k)
            add = add.query('label == "partial_reconstruct"')
            add.loc[:,'rep'] += df.rep.max()
            old[k] = pd.concat([df, add], sort=False).reset_index(drop=True)
    old.update(new)
        
def combine_files(reconstruction_files, query_string=None):
    if type(reconstruction_files) != dict:
        out = load_files(reconstruction_files, query_string)
    else:
        out = reconstruction_files
    return pd.concat(out, sort=False).reset_index(drop=True)

# *****************************************************************************  
# helper functions for reconstrucionts
# *****************************************************************************  

def normalize_reconstruction(reconstruction, c, inplace=True):
    """ Ensures reconstructions lie on the unit circle """
    if not inplace:
        reconstruction = reconstruction.copy()
    normed = normalize(reconstruction.iloc[:,:c])
    reconstruction.iloc[:,:c] = normed
    if not inplace:
        return reconstruction
    
def reorder_FA(ref_FA, new_FA, thresh=.9):
    """ Reorder FA to correspond to old FA, and check that there is such a correspondence"""
    c = len(ref_FA.columns)
    corr = pd.concat([ref_FA, new_FA], axis=1, sort=False).corr().iloc[c:, :c]
    new_FA = new_FA.loc[:,abs(corr).idxmax()]
    new_FA.columns = ref_FA.columns
    # if the correlation is low, the factors are completely off
    if corr.max().min() < thresh:
        return None
    else:
        return new_FA
    
# *****************************************************************************   
# utils for deriving and evaluating ontological factors for out-of-model tasks
# *****************************************************************************  

def run_linear(scores, test_vars, clf=LinearRegression(fit_intercept=False)):
    """
    Run Knearest neighbor using precomputed distances to create an ontological mapping
    
    Args:
        scores: ontological scores
        test_vars: variable to reconstruct
        clf: linear model that returns coefs
    """
    clf.fit(scores, scale(test_vars))
    out = clf.coef_
    if len(out.shape)==1:
        out = out.reshape(1,-1)
    out = pd.DataFrame(out, columns=scores.columns)
    out['var'] = test_vars.columns
    return out

def linear_reconstruction(results, drop_regex, 
                          pseudo_pop_size=60, n_reps=100, 
                          clf=LinearRegression(fit_intercept=False),
                          EFA_rotation='oblimin', 
                          independent_EFA=False,
                          verbose=True):
    def run_EFA(data, c, rotation, orig_scores):
        fa, out = psychFA(data, c, rotate=EFA_rotation)
        scores = pd.DataFrame(out['scores'], index=data.index)
        scores = reorder_FA(orig_scores, scores)
        return scores
    
    data = results.data
    c = results.EFA.results['num_factors']
    orig_scores = results.EFA.get_scores(c, rotate=EFA_rotation)
    # refit an EFA model without variable    
    drop_vars = list(data.filter(regex=drop_regex).columns)
    subset = data.drop(drop_vars, axis=1)
    scores = run_EFA(subset, c, EFA_rotation, orig_scores)
    # check to see if scores are problematic (not highly correlated with original scores)
    if scores is None:
        return None, None
    if verbose:
        print('*'*79)
        print('Reconstructing', drop_vars)
        print('*'*79)
        
    if verbose: print('Starting full reconstruction')
    full_reconstruction = run_linear(scores, data.loc[:, drop_vars], clf)
    full_reconstruction.reset_index(drop=True, inplace=True)

    if verbose: print('Starting partial reconstruction, pop size:', pseudo_pop_size)
    estimated_loadings = pd.DataFrame()
    for rep in range(n_reps):
        if verbose and rep%100==0: 
            print('Rep', rep)
        random_subset = np.random.choice(data.index,pseudo_pop_size, replace=False)
        if independent_EFA:
            tmp_subset = subset.drop(random_subset.index)
            scores = run_EFA(tmp_subset, c, EFA_rotation, orig_scores)
        out = run_linear(scores.loc[random_subset], data.loc[random_subset, drop_vars], clf)
        out['rep'] = rep+1
        estimated_loadings = pd.concat([estimated_loadings, out], sort=False)
    estimated_loadings.reset_index(drop=True)
    return estimated_loadings, full_reconstruction

    
def run_kNeighbors(distances, loadings, test_vars, 
                   weightings=('uniform',), k_list=(3)):
    """
    Run Knearest neighbor using precomputed distances to create an ontological mapping
    
    Args:
        distances: square distance matrix to pass to KNeighborsRegressors
        loadings: loading matrix for training
        test_vars: variable to reconstruct
        weightings: (optional) list of weightings to pass to KNeighbors
        k_list: list of k values to pass to KNeighbors as n_neighbors
    """
    train_distances = distances.loc[loadings.index, loadings.index]
    test_distances = distances.loc[test_vars, loadings.index]
    to_return = pd.DataFrame()
    for weighting in weightings:
        for k in k_list:
            clf = KNeighborsRegressor(metric='precomputed', n_neighbors=k, weights=weighting)
            clf.fit(train_distances, loadings)
            out = clf.predict(test_distances)
            out = pd.DataFrame(out, columns=loadings.columns)
            out['var'] = test_vars
            out['k'] = k
            out['weighting'] = weighting
            # add neighbors and distances
            neighbors = clf.kneighbors(test_distances)
            out['distances'] = tuple(neighbors[0])
            out['neighbors'] = tuple(test_distances.columns[neighbors[1]])
            to_return = pd.concat([to_return, out], sort=False)
    return to_return
    
def k_nearest_reconstruction(results, drop_regex, num_available_measures=None,
                             pseudo_pop_size=60, n_reps=100, 
                             k_list=None, EFA_rotation='oblimin', 
                             metric='correlation',
                             independent_EFA=False,
                             verbose=True,
                             weightings = ['uniform', 'distance']):
    def run_EFA(data, c, rotation, orig_loading):
        fa, out = psychFA(data, c, rotate=EFA_rotation)
        loadings = pd.DataFrame(out['loadings'], index=data.columns)
        loadings = reorder_FA(orig_loadings, loadings)
        return loadings
    
    if k_list is None:
        k_list = [3]
    data = results.data
    c = results.EFA.results['num_factors']
    orig_loadings = results.EFA.get_loading(c, rotate=EFA_rotation)
    # refit an EFA model without variable    
    drop_vars = list(data.filter(regex=drop_regex).columns)
    subset = data.drop(drop_vars, axis=1)
    loadings = run_EFA(subset, c, EFA_rotation, orig_loadings)
    # check to see if loadings are problematic (not highly correlated with original scores)
    if loadings is None:
        return None, None
    if num_available_measures is not None:
        measures = np.unique([i.split('.')[0] for i in results.data.columns])
        measures = [x for x in measures if x not in drop_regex]
        available_measures =  np.random.choice(measures, num_available_measures, replace=False)
        available_vars = results.data.filter(regex='^'+'|^'.join(available_measures)).columns
        data = data.loc[:, set(available_vars) | set(drop_vars)]
        loadings = loadings.loc[available_vars,:]
        k_list = [k for k in k_list if k<len(available_vars)]
        if len(k_list) == 0:
            k_list = [len(available_vars)-1]
    if verbose:
        print('*'*79)
        print('Reconstructing', drop_vars)
        if num_available_measures is not None:
            print('*'*79)
            print('Using subset of measures:', available_measures)
        print('*'*79)
    if verbose: print('Starting full reconstruction')
    distances = pd.DataFrame(squareform(pdist(data.T, metric=metric)), 
                             index=data.columns, 
                             columns=data.columns).drop(drop_vars, axis=1)

    full_reconstruction = run_kNeighbors(distances, loadings, drop_vars, weightings, k_list)
    full_reconstruction.reset_index(drop=True, inplace=True)

    if verbose: print('Starting partial reconstruction, pop size:', pseudo_pop_size)
    estimated_loadings = pd.DataFrame()
    reps_remaining = n_reps
    while reps_remaining > 0:
        rep = n_reps-reps_remaining
        if verbose and rep%50==0: 
            print('Rep', rep)
        random_subset = data.sample(pseudo_pop_size)
        if independent_EFA:
            tmp_subset = subset.drop(random_subset.index)
            loadings = run_EFA(tmp_subset, c, EFA_rotation, orig_loadings)
            if loadings is None:
                continue
        distances = pd.DataFrame(squareform(pdist(random_subset.T, metric=metric)), 
                                 index=random_subset.columns, 
                                 columns=random_subset.columns).drop(drop_vars, axis=1)
        out = run_kNeighbors(distances, loadings, drop_vars, weightings, k_list)
        out['rep'] = rep+1
        out
        estimated_loadings = pd.concat([estimated_loadings, out], sort=False)
        reps_remaining -= 1
    estimated_loadings.reset_index(drop=True)
    # add columns to dataframes
    if num_available_measures is not None:
        estimated_loadings['num_available_measures'] = num_available_measures
        estimated_loadings['available_vars'] = ','.join(available_vars)
        full_reconstruction['num_available_measures'] = num_available_measures
        full_reconstruction['available_vars'] = ','.join(available_vars)
    return estimated_loadings, full_reconstruction

def corr_scoring(organized_results):
    for v, group in organized_results.groupby('var'):
        corr_scores = np.corrcoef(x=group.iloc[:,:5].astype(float))[:,0]
        organized_results.loc[group.index, 'corr_score'] = corr_scores
        
def organize_reconstruction(reconstruction_results, scoring_funs=None):
    # organize the output from the simulations
    reconstruction_df = reconstruction_results.pop('true')
    reconstruction_df.loc[:,'label'] = 'true'
    for pop_size, (estimated, full) in reconstruction_results.items():
        combined = pd.concat([full, estimated], sort=False)
        combined.reset_index(drop=True, inplace=True)
        labels = ['full_reconstruct']
        if len(full.shape) == 2:
            labels += ['full_reconstruct']*(full.shape[0]-1)
        labels += ['partial_reconstruct']*estimated.shape[0]
        combined.loc[:, 'label'] = labels
        combined.loc[:, 'pop_size'] = pop_size
        reconstruction_df = pd.concat([reconstruction_df, combined], sort=False)
    reconstruction_df = reconstruction_df.infer_objects().reset_index(drop=True)
    # drop redundant reconstructions
    pop_sizes = reconstruction_df.pop_size.dropna().unique()[1:]
    drop_indices = reconstruction_df[(reconstruction_df.label=="full_reconstruct") & 
                                     (reconstruction_df.pop_size.isin(pop_sizes))].index
    reconstruction_df.drop(drop_indices, inplace=True)
    reconstruction_df.loc[(reconstruction_df.label=="full_reconstruct"), 'pop_size'] = np.nan
    if scoring_funs:
        for fun in scoring_funs:
            fun(reconstruction_df)
    return reconstruction_df

def get_reconstruction_results(results, measure_list, pop_sizes=(100,200), 
                               EFA_rotation='oblimin',
                               recon_fun=linear_reconstruction, 
                               scoring_funs=(corr_scoring,), 
                               **kwargs):
    loadings = results.EFA.get_loading(rotate=EFA_rotation, c=results.EFA.results['num_factors'])
    out = {}
    # convert list of measures to a regex lookup
    for measure in measure_list:
        reconstruction_results = {}
        for pop_size in pop_sizes:  
            estimated, full = recon_fun(results, drop_regex=measure, 
                                        pseudo_pop_size=pop_size, 
                                        EFA_rotation=EFA_rotation, **kwargs)
            if estimated is None:
                break
            reconstruction_results[pop_size] = [estimated, full]
        if len(reconstruction_results) > 0:
            true = loadings.loc[set(full['var'])]
            true.loc[:,'var'] = true.index
            reconstruction_results['true'] = true
            organized = organize_reconstruction(reconstruction_results, scoring_funs=scoring_funs)
            out[measure.lstrip('^')] = organized  
    return out
    
# other evaluations
def CV_predict(reconstruction, labels, cv=10, clf=LinearSVC(), test_set=None):
    """
    Run cross-validated classification of reconstruction
    Args:
        reconstruction: a reconstruction created by get_reconstruction_results
        cv: an int or sklearn CV object
        clf: a sklearn multilabel classifier
        test_set: separate test_set to be used after fitting the classifier on all of the data
    
    """
    if type(cv) == int:
        cv = StratifiedKFold(n_splits=cv)
    c = reconstruction.columns.get_loc('var')
    le = LabelEncoder()
    embedding = reconstruction.iloc[:,:c].values
    encoded_labels = le.fit_transform(labels)
    scores = {'precision': [],
              'recall': [],
              'f1': [],
              'confusion': []}
    for train_ind, test_ind in cv.split(embedding, encoded_labels):
        X = embedding[train_ind]
        y = encoded_labels[train_ind]
        clf.fit(X,y)
        X_test = embedding[test_ind]
        y_test = encoded_labels[test_ind]
        predicted = clf.predict(X_test)
        # scoring
        scores['precision'].append(precision_score(y_test, predicted, average='macro'))
        scores['recall'].append(recall_score(y_test, predicted, average='macro'))
        scores['f1'].append(f1_score(y_test, predicted, average='macro'))
        scores['confusion'].append(confusion_matrix(y_test, predicted))
    for key, val in scores.items():
        scores[key] = np.mean(val,0)
    if test_set:
        clf.fit(embedding, encoded_labels)
        predicted = clf.predict(test_set[0])
        scores['true_confusion'] = confusion_matrix(le.transform(test_set[1]), predicted)
    return scores

def summarize_k(k_reconstructions):
    """ Takes dictionary of k reconstructions and outputs a summary"""
    var_summary = pd.DataFrame()
    for measure, reconstruction in k_reconstructions.items():
        tmp_summary = reconstruction.query('label=="partial_reconstruct"') \
                        .groupby(['pop_size', 'k', 'weighting','var'])['corr_score'].mean().reset_index()
        var_summary = pd.concat([var_summary, tmp_summary])
    
    k_summary = var_summary.groupby(['pop_size', 'k', 'weighting']).mean()
    # summarize further
    # determine best parameters
    k_best_params = {}
    for pop_size in k_summary.index.unique(level='pop_size'):
        tmp=k_summary.query('pop_size == %s' % pop_size)
        best_params = tmp.idxmax()[0]
        best_val = tmp.loc[best_params][0]
        k_best_params[pop_size] = {'k': best_params[1], 
                                   'weighting': best_params[2],
                                   'best_val': best_val}
    # extract values using best parameters
    reconstruction_list = []
    for reconstruction in k_reconstructions.values():
        true = reconstruction.query('label == "true"')
        reconstruction_list.append(true)
        for k, v in k_best_params.items():
            tmp_partial = reconstruction.query('pop_size == %s and \
                                         k == %s and \
                                         weighting == "%s"' % (k, v['k'], v['weighting']))
            full = reconstruction.query('label == "full_reconstruct" and \
                                         k == %s and \
                                         weighting == "%s"' % (v['k'], v['weighting']))
            reconstruction_list += [tmp_partial, full]

    k_best_reconstruction = pd.concat(reconstruction_list, axis=0, sort=False)
    return var_summary, k_best_params, k_best_reconstruction

def summarize_k_partial(k_reconstructions):
    """ Takes dictionary of k reconstructions and outputs a summary"""
    var_summary = pd.DataFrame()
    for measure, reconstruction in k_reconstructions.items():
        tmp_summary = reconstruction.query('label=="partial_reconstruct"') \
                        .groupby(['num_available_measures', 'pop_size', 'available_vars', 'var'])['corr_score'].mean().reset_index()
        var_summary = pd.concat([var_summary, tmp_summary])
    return var_summary

# *****************************************************************************   
# wrapper for running reconstructions with different parameters
# *****************************************************************************  

def run_reconstruction(results, measure_list, pop_sizes, n_reps, recon_fun,
                       previous_files=None, append=True, verbose=True, **kwargs):
    regex_list = ['^'+m for m in measure_list]
    if previous_files is not None and len(previous_files) > 0:
        previous_recon = load_files(previous_files)
        if not append:
            tmp_measures = set(measure_list) - set(k_reconstructions.keys())
            regex_list = ['^'+m for m in tmp_measures]
            
    # run new reconstruction
    new_recon = get_reconstruction_results(results, 
                                           regex_list, 
                                           pop_sizes,  
                                           n_reps=n_reps, 
                                           recon_fun=recon_fun, 
                                           verbose=verbose,
                                          **kwargs)
    # determine measures that have been updated
    updated = list(new_recon.keys())
    
    # update previous
    if previous_files is not None and len(previous_files) > 0:
        if append:
            update_files(previous_recon, new_recon)
        else:
            previous_recon.update(new_recon)
        to_return = previous_recon
    else:
        to_return = new_recon
    return updated, to_return