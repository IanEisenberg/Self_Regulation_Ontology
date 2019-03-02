
import fancyimpute
import numpy as np
import pandas as pd
from random import randint

from selfregulation.utils.utils import get_behav_data

def get_rand_index(M, n):
    subset = [(randint(0,M.shape[0]-1), randint(0,M.shape[1]-1)) for _ in range(n)]
    subset = pd.unique(subset)
    while len(subset) < n:
        new_indices = [(randint(0,M.shape[0]-1), randint(0,M.shape[1]-1)) for _ in range(n-len(subset))]
        subset = list(subset) + new_indices
        subset = pd.unique(subset)
    return list(subset)

def impute(data, method):
    sigma = data.std()
    matrix = (data/sigma).as_matrix()
    complete_matrix = method().fit_transform(matrix)*sigma.tolist()
    return pd.DataFrame(complete_matrix, index = data.index, columns = data.columns)
    
   
DV_df = get_behav_data('Discovery_9-26-16', use_EZ = True)
sigma = DV_df.std()

base_matrix = (DV_df/sigma).as_matrix()

# test different imputation methods
methods = [fancyimpute.SoftImpute, fancyimpute.IterativeSVD, fancyimpute.KNN] 
correlations = {}
percent_off = {}
for method in methods:
    print('using %s' % method)
    correlations[method] = []
    percent_off[method] = []
    for simulation in range(20):
        indices = get_rand_index(base_matrix,1000)
        originals = [base_matrix[i] for i in indices]
        missing_matrix = base_matrix.copy()
        for i in indices:
            missing_matrix[i] = np.nan
        complete_matrix = method(verbose = False).fit_transform(missing_matrix)
        imputed = [complete_matrix[i] for i in indices]
        correlations[method].append(pd.DataFrame([imputed,originals]).T.corr().iloc[0,1])
        deviation = np.mean([(abs((o-i)/o)) for o,i in zip(originals,imputed) if o == o and o > .01])
        percent_off[method].append(deviation)
        

# try different K values for KNN
for k in range(4,15):
    print('using k=%s' % k)
    key = 'KNN:K=%s' % k
    correlations[key] = []
    percent_off[key] = []
    for simulation in range(100):
        indices = get_rand_index(base_matrix,5000)
        originals = [base_matrix[i] for i in indices]
        missing_matrix = base_matrix.copy()
        for i in indices:
            missing_matrix[i] = np.nan
        complete_matrix = fancyimpute.KNN(k = k, verbose = False).fit_transform(missing_matrix)
        imputed = [complete_matrix[i] for i in indices]
        correlations[key].append(pd.DataFrame([imputed,originals]).T.corr().iloc[0,1])
        deviation = np.mean([(abs((o-i)/o)) for o,i in zip(originals,imputed) if o == o and o > .01])
        percent_off[key].append(deviation)