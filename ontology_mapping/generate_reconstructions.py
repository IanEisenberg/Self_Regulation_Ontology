
# coding: utf-8

# # Reconstructing out-of-sample DVs
# 
# Given a quantitative ontology, or psychological space, that DVs can be projected into, how can we determine the embedding of new variables?
# 
# Currently, our embedding is determined by factor analysis. Thus ontological embedding are only known for the DVs entered into the original model. How could we extend this?
# 
# One possibility is measuring new variables in the same population that completed our original battery. After doing this we could either (1) run the model anew, or (2) use linear regression to map the already discovered factors onto the new variables. The former is better, but results in small changes to the actual factors with each new variable. The latter method ensures that our factors stay the same. Neither is scalable, however, as we do not, in general, have access to a constant population that can be remeasured whenever new measures come into the picture.
# 
# Another possibility that works with new populations requires that the new population completes the entire battery used to estimate the original factors, in addition to whatever new variables are of interest. Doing so allows the calculation of factor scores for this new population based on the original model, which can then be mapped to the new measures of interest. This allows researchers to capitalize on the original model (presumably fit on more subjects than the new study), while expanding the ontology. Problems exist here, however.
# - The most obvious problem is that you have to measure the new sample on the entire battery used to fit the original EFA model. Given that this takes many hours (the exact number depending on whether tasks, surveys or both are used), this is exceedingly impractical. In our case we did have our new fMRI sample take the entire battery (or at least a subset of participants), so this problem isn't as relevant
# - Still problems remain. If N is small, the estimates of the ontological embeddings for new DVs are likely unstable.
# 
# This latter problem necessitates some quantitative exploration. This notebook simulates the issue by:
# 1. Removing a DV from the original ontology dataset
# 2. Performing EFA on this subset
# 3. Using linear regression to map these EFA factors to the left out variable
# (3) is performed on smaller population sizes to reflect the reality of most studies (including ours) and is repeated to get a sense of the mapping's variability
# 
# This simulates the ideal case of mapping a new variable by measuring the entire ontological battery. We also use K-nearest-neighbor regression to map new variables into the space with fewer variables. Doing this proceeds as follows:
# 1. Remove a DV from the original ontology dataset
# 2. Perform EFA
# 3. Create a distance matrix between each DV and every other DV
# 
# 
# ### Small issues not currently addressed
# 
# - The EFA model is fit on the entire population. An even more stringent simulation would subset the subjects used in the "new study" and fit the EFA model on a completely independent group. I tried this once - the factor scores hardly differed. In addition, I want the EFA model to be as well-powered as possible, as that will be the reality for this method moving forward

# ## Imports

# In[ ]:


import argparse
from glob import glob
import numpy as np
from os import makedirs, path
import pandas as pd
import pickle
import sklearn
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import normalize
from ontology_mapping.reconstruction_utils import (load_files,
                                                   combine_files,
                                                   update_files,
                                                   normalize_reconstruction,
                                                   get_reconstruction_results, 
                                                   linear_reconstruction,
                                                   k_nearest_reconstruction,
                                                   CV_predict,
                                                   summarize_k,
                                                   run_reconstruction)
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_recent_dataset, get_info


# In[ ]:


# ignore some warnings
import warnings
warnings.filterwarnings("ignore", category=sklearn.metrics.classification.UndefinedMetricWarning)


# In[ ]:


# argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pop_sizes', nargs='+', default=[30, 50, 100, 400], type=int)
    parser.add_argument('-n_reps', default=250, type=int)
    parser.add_argument('-n_measures', default=None, type=int)
    parser.add_argument('-result_subset', default='task')
    parser.add_argument('-rerun', action='store_true')
    parser.add_argument('-no_append', action='store_true')
    parser.add_argument('-EFA_rotation', default='oblimin')
    parser.add_argument('-knn_metric', default='correlation')
    parser.add_argument('-verbose', action='store_true')
    parser.add_argument('-no_save', action='store_true')
    args, _ = parser.parse_known_args()
    pop_sizes = args.pop_sizes
    n_reps = args.n_reps
    n_measures = args.n_measures
    result_subset = args.result_subset
    rerun = args.rerun
    append = not args.no_append
    knn_metric = args.knn_metric
    EFA_rotation = args.EFA_rotation
    verbose = args.verbose
    dataset = get_recent_dataset()
    save = not args.no_save


# ## Additional Setup

# In[ ]:


# Load dataset
np.random.seed(12412)
results = load_results(dataset)[result_subset]
c = results.EFA.get_c()

# Define classifiers
classifiers = {'Ridge': Ridge(fit_intercept=False),
               'LR': LinearRegression(fit_intercept=False)}
# get output dir to store results
output_dir = path.join(get_info('results_directory'),
                       'ontology_reconstruction', dataset, results.ID, EFA_rotation)
makedirs(output_dir, exist_ok=True)
# get plot dir to store plots
plot_dir = path.join(output_dir, 'Plots')
makedirs(plot_dir, exist_ok=True)


# In[ ]:


# get a random subset of variables to perform the calculation on if n_vars is set
measures = np.unique([i.split('.')[0] for i in results.data.columns])
if n_measures is not None:
    measure_list = np.random.choice(measures, n_measures, replace=False)
else:
    measure_list = measures
# get all variables from selected tasks
var_list = results.data.filter(regex='|'.join(measure_list)).columns


# Run simulation for every variable at different population sizes. 
# 
# That is, do the following:
# 
# 1. take a variable (say stroop incongruent-congruent RT), remove it from the data matrix
# 2. Run EFA on the data matrix composes of the 522 (subject) x N-1 (variable) data matrix
# 3. Calculate factor scores for all 522 subjects
# 4. Select a subset of "pop_size" to do an "ontological mapping". That is, pretend that these subjects did the whole battery (missing the one variable) *and then* completed one more task. The idea is we want to do a mapping from those subject's factor scores to the new variable
#    1. We can do a linear mapping (regression) from the ontological scores to the output variable
#    2. We can do a k-nearest neighbor interpolation, where we say the unknown ontological factor is a blend of the "nearest" variables in the dataset
# 5. Repeat (4) a number of times to get a sense for the accuracy and variability of that mapping
# 6. Compare the estimated ontological scores for the held out var (stroop incongruent-congruent) to the original "correct" ontological mapping (that would have been obtained if the variable was included in the original data matrix

# ## Perform reconstruction

# ### K Nearest Neighbor Reconstruction

# In[ ]:


# %%time
verbose=True
for name, independent_flag in [('KNNR', False), ('KNNRind', True)]:
    k_list = list(range(1,20))
    basename = path.join(output_dir, '%s_%s-*' % (name, knn_metric))
    files = glob(basename)
    if name == "KNNRind":
        pops_to_use = [i for i in pop_sizes if i < 100]
    else:
        pops_to_use = pop_sizes
    updated, k_reconstructions = run_reconstruction(results, 
                                                   measure_list, 
                                                   pops_to_use, 
                                                   n_reps, 
                                                   k_nearest_reconstruction,
                                                   previous_files=files, 
                                                   append=append, 
                                                   verbose=verbose, 
                                                   k_list=k_list, 
                                                   metric=knn_metric,
                                                   independent_EFA=independent_flag,
                                                   EFA_rotation=EFA_rotation)
    for measure in updated:
        df = k_reconstructions[measure]
        if save:
            df.to_pickle(basename[:-1]+'%s.pkl' % measure)


# ### Linear Reconstruction

# In[ ]:


# %%time
clfs = {'Linear': LinearRegression(fit_intercept=False),
       'RidgeCV': RidgeCV(fit_intercept=False, cv=10)}
linear_reconstructions = {}
for clf_name, clf in clfs.items():
    
    basename = path.join(output_dir, 'linear-%s_reconstruct*' % clf_name)
    files = glob(basename)
    updated, reconstruction = run_reconstruction(results, 
                                                   measure_list, 
                                                   pop_sizes, 
                                                   n_reps, 
                                                   linear_reconstruction,
                                                   previous_files=files, 
                                                   append=append, 
                                                   verbose=verbose, 
                                                   clf=clf,
                                                   EFA_rotation=EFA_rotation)
    for measure in updated:
        df = reconstruction[measure]
        if save:
            df.to_pickle(basename[:-1]+'-%s.pkl' % measure)


# ### K Nearest Neighbor Partial Reconstruction

# In[ ]:


for num_available_measures in range(5,len(measures),5):
    # repeat with different samples of measures
    for sample_repeats in range(50):
        if verbose and sample_repeats%5==0:
            print('SAMPLE %s FOR %s MEASURES' % (sample_repeats, num_available_measures))
        basename = path.join(output_dir, '%s_%s-*' % ('KNNRpartial', knn_metric))
        files = glob(basename)
        k_list=[13]
        updated, k_reconstructions = run_reconstruction(results, 
                                                       measure_list, 
                                                       pop_sizes, 
                                                       n_reps=10, 
                                                       recon_fun=k_nearest_reconstruction,
                                                       previous_files=files, 
                                                       append=append, 
                                                       verbose=False, 
                                                       k_list=k_list, 
                                                       metric=knn_metric,
                                                       independent_EFA=False,
                                                       EFA_rotation=EFA_rotation,
                                                       num_available_measures=num_available_measures,
                                                       weightings=['distance'])

        for measure in updated:
            df = k_reconstructions[measure]
            if save:
                df.to_pickle(basename[:-1]+'%s.pkl' % measure)

