from collections import OrderedDict as odict
import fancyimpute
import functools
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage, cut_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (adjusted_mutual_info_score, r2_score, 
                             silhouette_samples, silhouette_score)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, scale

from selfregulation.utils.data_preparation_utils import (remove_outliers, 
                                                         transform_remove_skew)
from selfregulation.utils.r_to_py_utils import (get_attr, dynamicTreeCut, 
                                                missForest, psychFA)

def set_seed(seed):
    def seeded_fun_decorator(fun):
        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            np.random.seed(seed)
            out = fun(*args, **kwargs)
            np.random.seed()
            return out
        return wrapper
    return seeded_fun_decorator
    

class Imputer(object):
    """ Imputation class so that fancyimpute can be used with scikit pipeline"""
    def __init__(self, imputer=None):
        if imputer is None:
            self.imputer = fancyimpute.SimpleFill()
        else:
            self.imputer = imputer(verbose=False)
        
    def transform(self, X):
        transformed = self.imputer.fit_transform(X)
        return transformed
    
    def fit(self, X, y=None):
        return self

def distcorr(X, Y, flip=True):
    """ Compute the distance correlation function
    
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    
    Taken from: https://gist.github.com/satra/aa3d19a12b74e9ab7941
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    if flip == True:
        dcor = 1-dcor
    return dcor

def abs_pdist(mat, square=False):
    correlation_dist = pdist(mat, metric='correlation')
    correlations = 1-correlation_dist
    absolute_distance = 1-abs(correlations)
    if square == True:
        absolute_distance = squareform(absolute_distance)
    return absolute_distance

def shorten_labels(labels, conversions={}):
    lookup = []
    new_labels = []
    for label in labels:
        if label in conversions.keys():
            new_label = conversions[label]
        else:
            label.replace('-', ' ')
            new_label = ''.join([l[0:2] for l in label.split(' ')])
        new_labels.append(new_label)
        lookup.append((new_label, label))
    return new_labels, lookup

# ****************************************************************************
# helper functions for hierarchical clustering
# ****************************************************************************
def reorder_labels(labels, link):
    """ reorder labels based on a linkage matrix
    
    reorder labels based on dendrogram position
    reindex so the clusters are in order based on their proximity
    in the dendrogram
    """
    reorder_vec = leaves_list(link)
    cluster_swap = {}
    last_group = 1
    for i in labels[reorder_vec]:
        if i not in cluster_swap.keys():
            cluster_swap[i] = last_group
            last_group += 1
    cluster_reindex = np.array([cluster_swap[i] for i in labels])
    return cluster_reindex
    
def hierarchical_cluster(df, compute_dist=True,  pdist_kws=None, 
                         method='average', min_cluster_size=3,
                         cluster_kws=None):
    """
    plot hierarchical clustering and heatmap
    :df: a correlation matrix
    parse_heatmap: int (optional). If defined, devides the columns of the 
                    heatmap based on cutting the dendrogram
    """
    
    # if compute_dist = False, assume df is a distance matrix. Otherwise
    # compute distance on df rows
    if compute_dist == True:
        if pdist_kws is None:
            pdist_kws= {'metric': 'correlation'}
        if pdist_kws['metric'] == 'abscorrelation':
            # convert to absolute correlations
            dist_vec = abs_pdist(df)
        elif pdist_kws['metric'] == 'sqcorrelation':
            # convert to squared correlations
            dist_vec = squareform(1-df.T.corr()**2)
        else:
            dist_vec = pdist(df, **pdist_kws)
        dist_df = pd.DataFrame(squareform(dist_vec), 
                               index=df.index, 
                               columns=df.index)
    else:
        assert df.shape[0] == df.shape[1]
        dist_df = df
        dist_vec = squareform(df.values)
    #clustering. This works the same as hclust
    link = linkage(dist_vec, method=method)    
    #dendrogram
    # same as order.dendrogram(as.dendrogram(hclust output)) in R
    reorder_vec = leaves_list(link)
    clustered_df = dist_df.iloc[reorder_vec, reorder_vec]
    # clustering
    if cluster_kws is None:
        cluster_kws = {'minClusterSize': 3,
                       'verbose': 0,
                       'pamStage': False}
    labels = dynamicTreeCut(dist_df, func='hybrid', method=method,  **cluster_kws)
    labels = reorder_labels(labels, link)
    return {'linkage': link, 
            'distance_df': dist_df, 
            'clustered_df': clustered_df,
            'reorder_vec': reorder_vec,
            'labels': labels}

def silhouette_analysis(clustering, labels=None):
    distance_df = clustering['distance_df']
    if labels is None:
        labels = clustering['labels']
    sample_scores = silhouette_samples(distance_df, metric='precomputed', labels=labels)
    score = np.mean(sample_scores)
    return sample_scores, score

def get_constant_height_labels(clustering, n_clusters=None):
    """
    use silhouette analysis to select the best heigh to cut a linkage matrix
    :df: a correlation matrix
    parse_heatmap: int (optional). If defined, devides the columns of the 
                    heatmap based on cutting the dendrogram
    """
    N_variables = len(clustering['reorder_vec'])
    scores = []
    if n_clusters is None:
        for k_clusters in range(2,N_variables//3):
            labels = cut_tree(clustering['linkage'], n_clusters=k_clusters)
            try:
                score = silhouette_score(clustering['distance_df'], 
                                         labels.ravel(), metric='precomputed')
            except ValueError:
                continue
            scores.append((k_clusters,score))
        best_k = max(scores, key=lambda x: x[1])[0]
        labels = cut_tree(clustering['linkage'], n_clusters=best_k)

    else:
        labels = cut_tree(clustering['linkage'], n_clusters=n_clusters)
        score = silhouette_score(clustering['distance_df'], 
                                         labels, metric='precomputed')
        scores.append((n_clusters, score))
    labels = reorder_labels(labels.flatten(), clustering['linkage'])
    # comparison
    MI = adjusted_mutual_info_score(labels, clustering['labels'])
    return labels, scores, MI
    
# ****************************************************************************
# helper functions for dealing with factor analytic results
# ****************************************************************************
def corr_lower_higher(higher_dim, lower_dim, cross_only=True):
    """
    Returns a correlation matrix between factors at different dimensionalities
    cross_only: bool, if True only display the correlations between dimensions
    """
    # higher dim is the factor solution with fewer factors
    higher_dim = higher_dim.copy()
    lower_dim = lower_dim.copy()
    higher_n = higher_dim.shape[1]
    
    lower_dim.columns = ['l%s' % i  for i in lower_dim.columns]
    higher_dim.columns = ['h%s' % i for i in higher_dim.columns]
    corr = pd.concat([higher_dim, lower_dim], axis=1).corr()
    if cross_only:
        corr = corr.iloc[:higher_n, higher_n:]
    return corr

# functions to fit and extract factor analysis solutions
def find_optimal_components(data, minc=1, maxc=50, nobs=0, metric='BIC'):
    """
    Fit EFA over a range of components and returns the best c. If metric = CV
    uses sklearn. Otherwise uses psych
    metric: str, method to use for optimal components. Options 'BIC', 'SABIC',
            and 'CV'
    """
    steps_since_best = 0 # count steps since last best metric.
    metrics = {}
    maxc = min(maxc, data.shape[1])
    n_components = range(minc,maxc)
    scaler = StandardScaler()
    if metric != 'CV':
        best_metric = float("Inf")
        best_c = 0
        for c in n_components:
            out = psychFA(data, c, method='ml', nobs=nobs)
            if out is None:
                break
            fa, output = out
            curr_metric = output[metric]
            # iterate counter if new metric isn't better than previous metric
            if len(metrics) > 0:
                if curr_metric >= best_metric:
                    steps_since_best += 1
                else:
                    steps_since_best = 0
                    best_c = c
                    best_metric = curr_metric
            metrics[c] = curr_metric
            if steps_since_best > 2:
                break
    else:
        for c in n_components:
            fa = FactorAnalysis(c)
            scaler = StandardScaler()
            imputer = Imputer()
            pipe = Pipeline(steps = [('impute', imputer),
                                     ('scale', scaler),
                                     ('fa', fa)])
            cv = cross_val_score(pipe, data, cv=10)
            # iterate counter if new metric isn't better than previous metric
            if len(metrics) > 0:
                if cv < metrics[c-1]:
                    steps_since_best += 1
                else:
                    steps_since_best = 0
            metrics[c] = np.mean(cv)
            if steps_since_best > 2:
                break
        best_c = max(metrics, key=metrics.get)
    return best_c, metrics

def get_loadings(fa_output, labels, sort=False):
    """
    Takes output of psychFA, and a list of labels and returns a loadings dataframe
    """
    loading_df = pd.DataFrame(fa_output['loadings'], index=labels)
    if sort == True:
        # sort by maximum loading on surveys
        sorting_index = np.argsort(loading_df.filter(regex='survey',axis=0).abs().mean()).tolist()[::-1]
        loading_df = loading_df.loc[:,sorting_index]
        loading_df.columns = range(loading_df.shape[1])
    return loading_df

def get_top_factors(loading_df, n=4, verbose=False):
    """
    Takes output of get_loadings and prints the absolute top variables per factor
    """
    # number of variables to display
    factor_top_vars = {}
    for i,column in loading_df.iteritems():
        sort_index = np.argsort(abs(column))[::-1] # descending order
        top_vars = column[sort_index]
        factor_top_vars[i] = top_vars
        if verbose:
            print('\nFACTOR %s' % i)
            print(top_vars[0:n])
    return factor_top_vars

def reorder_data(data, groups, axis=1):
    ordered_cols = []
    for i in groups:
        ordered_cols += i[1]
    new_data = data.reindex_axis(ordered_cols, axis)
    return new_data

def create_factor_tree(data, component_range=(1,13), component_list=None,
                       rotate='oblimin'):
    """
    Runs "visualize_factors" at multiple dimensionalities and saves them
    to a pdf
    data: dataframe to run EFA on at multiple dimensionalities
    groups: group list to be passed to visualize factors
    filename: filename to save pdf
    component_range: limits of EFA dimensionalities. e.g. (1,5) will run
                     EFA with 1 component, 2 components... 5 components.
    component_list: list of specific components to calculate. Overrides
                    component_range if set
    """
    def get_similarity_order(lower_dim, higher_dim):
        "Helper function to reorder factors into correspondance between two dimensionalities"
        subset = corr_lower_higher(higher_dim, lower_dim)
        max_factors = np.argmax(abs(subset.values), axis=0)
        return np.argsort(max_factors)

    EFA_results = {}
    full_fa_results = {}
    # plot
    if component_list is None:
        components = range(component_range[0],component_range[1]+1)
    else:
        components = component_list
    for c in components:
        fa, output = psychFA(data, c, method='ml', rotate=rotate)
        tmp_loading_df = get_loadings(output, labels=data.columns)
        if (c-1) in EFA_results.keys():
            reorder_index = get_similarity_order(tmp_loading_df, EFA_results[c-1])
            tmp_loading_df = tmp_loading_df.iloc[:, reorder_index]
            tmp_loading_df.columns = sorted(tmp_loading_df.columns)
        EFA_results[c] = tmp_loading_df
        full_fa_results[c] = fa
    return EFA_results, full_fa_results

def get_factor_groups(loading_df):
    index_assignments = np.argmax(abs(loading_df).values,axis=1)
    names = loading_df.columns
    factor_groups = []
    for assignment in np.unique(index_assignments):
        name = names[assignment]
        assignment_vars = [var for i,var in enumerate(loading_df.index) if index_assignments[i] == assignment]
        # sort assignment_vars by maximum loading on their assigned factor
        assignment_vars = abs(loading_df.loc[assignment_vars, name]).sort_values()
        assignment_vars = list(assignment_vars.index)[::-1] # get names
        factor_groups.append([name, assignment_vars])
    return factor_groups


def get_scores_from_subset(data, fa_output, task_subset):
    match_cols = []
    for i, c in enumerate(data.columns):
        if np.any([task in c for task in task_subset]):
            match_cols.append(i)

    weights_subset = fa_output['weights'][match_cols,:]
    data_subset = scale(data.iloc[:, match_cols])
    subset_scores = data_subset.dot(weights_subset)

    # concat subset and full scores into one dataframe
    labels = ['%s_full' % i for i in list(range(fa_output['scores'].shape[1]))]
    labels+=[i.replace('full','subset') for i in labels]
    concat_df = pd.DataFrame(np.hstack([fa_output['scores'], subset_scores]),
                             columns = labels)
    
    # calculate variance explained by subset
    lr = LinearRegression()
    lr.fit(concat_df.filter(regex='subset'), 
           concat_df.filter(regex='full'))
    scores = r2_score(lr.predict(concat_df.filter(regex='subset')), 
                      concat_df.filter(regex='full'), 
                      multioutput='raw_values')
    return concat_df, scores


def quantify_higher_nesting(higher_dim, lower_dim):
    """
    Quantifies how well higher levels of the tree can be reconstructed from 
    lower levels
    """
    lr = LinearRegression()
    best_score = -1
    relationship = []
    # quantify how well the higher dimensional solution can reconstruct
    # the lower dimensional solution using a linear combination of two factors
    for higher_name, higher_c in higher_dim.iteritems():
        for lower_c1, lower_c2 in combinations(lower_dim.columns, 2):
            # combined prediction
            predict_mat = higher_dim.loc[:,[lower_c1, lower_c2]]
            lr.fit(predict_mat, higher_c)
            score = lr.score(predict_mat, higher_c)
            # individual correlation
            lower_subset = lower_dim.drop(higher_name, axis=1)
            higher_subset = higher_dim.drop([lower_c1, lower_c2], axis=1)
            corr = corr_lower_higher(higher_subset, lower_subset)
            if len(corr)==1:
                other_cols = [corr.iloc[0,0]]
            else:
                other_cols = corr.apply(lambda x: max(x**2)-sorted(x**2)[-2],
                                        axis=1)
            total_score = np.mean(np.append(other_cols, score))
            if total_score>best_score:
                best_score = total_score
                relationship = {'score': score,
                                'lower_factor': higher_c.name, 
                                'higher_factors': (lower_c1, lower_c2), 
                                'coefficients': lr.coef_}
    return relationship

def quantify_lower_nesting(factor_tree):
    """
    Quantifies how well lower levels of the tree can be reconstruted from
    higher levels
    """
    lr = LinearRegression()
    relationships = odict()
    for higher_c, lower_c in combinations(factor_tree.keys(), 2):
        higher_dim = factor_tree[higher_c]
        lower_dim = factor_tree[lower_c]
        lr.fit(higher_dim, lower_dim)
        scores = r2_score(lr.predict(higher_dim), 
                                 lower_dim, 
                                 multioutput='raw_values')
        relationship = {'scores': scores,
                        'coefs': lr.coef_}
        relationships[(higher_c,lower_c)] = relationship
    return relationships

# ****************************************************************************
# Helper functions for factor analysis
# ****************************************************************************
def transfer_scores(data, results, rotate='oblimin'):
    """ calculates factor scores in a new dataset based on a reference results object """
    ref_data = results.data
    EFA = results.EFA
    c = EFA.results['num_factors']
    loadings = EFA.get_loading(c=c, rotate=rotate)
    # transform data
    positive_skewed = [i.replace('.logTr', '') for i in ref_data.columns if ".logTr" in i]
    negative_skewed = [i.replace('.ReflogTr', '') for i in ref_data.columns if ".ReflogTr" in i]
    DVs = [i.replace('.logTr','').replace('.ReflogTr','') for i in ref_data.columns]
    data = data.loc[:, DVs]
    data = transform_remove_skew(data,
                                 positive_skewed=positive_skewed,
                                 negative_skewed=negative_skewed,
                                 drop_failed=False)
    data = remove_outliers(data)
    data_imputed, error = missForest(data)
    subset = data_imputed.loc[:, loadings.index]
    scaled_data = scale(subset)
    # calculate scores
    weights = get_attr(EFA.results['factor_tree_Rout_%s' % rotate][c], 'weights')
    scores = pd.DataFrame(scaled_data.dot(weights),
                          index=data_imputed.index,
                          columns=loadings.columns)
    return scores

# ****************************************************************************
# Helper functions for visualization of component loadings
# ****************************************************************************
from fancyimpute import SimpleFill
def residualize_baseline(df, baseline_vars=[]):
    if len(baseline_vars) == 0:
        baseline_vars = ['Age', 'Sex']
    # remove baseline vars
    baseline=df[baseline_vars]
    data=df.copy()
    data.drop(baseline_vars, axis=1, inplace=True)
    lr=LinearRegression()
    if data.isnull().sum().sum() > 0:
        imputed = SimpleFill().fit_transform(data)
        data = pd.DataFrame(imputed, 
                            index=data.index, 
                            columns=data.columns)
    for v in data:
        y=data[v]
        lr.fit(baseline,y)
        data[v]=y - lr.predict(baseline)
    return data




