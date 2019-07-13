import numpy as np
from os import path
import pandas as pd
import readline
import rpy2.robjects
from rpy2.robjects import pandas2ri, Formula
from rpy2.robjects.packages import importr
from selfregulation.utils.utils import get_info
pandas2ri.activate()

def missForest(data):
    missForest = importr('missForest')
    data_complete, error = missForest.missForest(data)
    imputed_df = pd.DataFrame(np.matrix(data_complete).T, index=data.index, columns=data.columns)
    return imputed_df, error
    

def GPArotation(data, method='varimax', normalize=True):
    GPArotation = importr('GPArotation')
    rotated_data = GPArotation.GPForth(data.values, method = method, normalize=normalize)[0]
    rotated_data = pd.DataFrame(data = np.matrix(rotated_data), index=data.index, columns=data.columns)
    return rotated_data

def get_Rpsych():
    psych = importr('psych')
    return psych

def get_attr(fa, attr):
        try:
            index = list(fa.names).index(attr)
            val = list(fa.items())[index][1]
            if len(val) == 1:
                val = val[0]
            if type(val)==rpy2.robjects.vectors.Matrix:
                val = np.matrix(val)
            return val
        except ValueError:
            print('Did not pass a valid attribute')
            
def psychFA(data, n_components, return_attrs=['BIC', 'SABIC', 'RMSEA'], 
            rotate='oblimin', method='ml', nobs=0, n_iter=1, verbose=False):
    psych = importr('psych')
    if n_iter==1:
        fa = psych.fa(data, n_components, rotate=rotate, fm=method, n_obs=nobs,
                      scores='tenBerge')
    else:
        assert nobs==0
        fa = psych.fa_sapa(data, n_components, rotate=rotate, fm=method, 
                           scores='tenBerge', n_iter=n_iter, frac=.9)
    # ensure the model isn't ill-specified
    if get_attr(fa, 'dof') > 0:
        attr_dic = {}
        # loadings are the weights of the linear combination between factors and variables
        attr_dic['loadings'] = get_attr(fa, 'loadings')
        # scores are calculated if raw data is passed, rather than a correlation matrix
        if 'scores' in fa.names:
            # scores are the the factors
            attr_dic['scores'] = get_attr(fa, 'scores')
            # weights are the "mixing matrix" such that the final data is
            # S * W
            attr_dic['weights'] = get_attr(fa, 'weights')
        for attr in return_attrs:
            attr_dic[attr] = get_attr(fa, attr)
        if verbose:
            print(fa)
        return fa, attr_dic
    else:
        if verbose:  print('Too few DOF to specify model!')
        return None

def dynamicTreeCut(distance_df, func='hybrid', method='average', **cluster_kws):
    """ uses DynamicTreeCut to find clusters
    Args:
        method = "hybrid" or "dyanmic":
    """
    stats = importr('stats')
    dynamicTreeCut = importr('dynamicTreeCut')
    dist = stats.as_dist(distance_df)
    link = stats.hclust(dist, method=method)
    if func == 'hybrid':
        dist = stats.as_dist(distance_df)
        clustering = dynamicTreeCut.cutreeHybrid(link, distance_df, **cluster_kws)
        return np.array(clustering[0])
    elif func == 'dynamic':
        clustering = dynamicTreeCut.cutreeDynamic(link, **cluster_kws)
        return np.array(clustering)
    
def glmer(data, formula, verbose=False):
    base = importr('base')
    lme4 = importr('lme4')
    rs = lme4.glmer(Formula(formula), data, family = 'binomial')
    
    fixed_effects = lme4.fixed_effects(rs)
    fixed_effects = {k:v for k,v in zip(fixed_effects.names, list(fixed_effects))}
                                  
    random_effects = lme4.random_effects(rs)[0]
    random_effects = pd.DataFrame([list(lst) for lst in random_effects], index = list(random_effects.colnames)).T
    if verbose:
        print(base.summary(rs))
    return fixed_effects, random_effects

def lmer(data, formula, verbose=False):
    base = importr('base')
    lme4 = importr('lme4')
    rs = lme4.lmer(Formula(formula), data)
    
    fixed_effects = lme4.fixed_effects(rs)
    fixed_effects = {k:v for k,v in zip(fixed_effects.names, list(fixed_effects))}
                                  
    random_effects = lme4.random_effects(rs)
    random_df = pd.DataFrame()
    for re in random_effects:
        random_effects = pd.DataFrame([list(lst) for lst in re], index = list(re.colnames)).T
        random_df = pd.concat([random_df, random_effects], axis=1)
    random_variance = pandas2ri.ri2py(base.as_data_frame(lme4.VarCorr_merMod(rs)))
    if verbose:
        print(base.summary(rs))
    return rs, random_variance, fixed_effects, random_df

def M3C(data, ncores=1, iters=100, maxk=20):
    base = importr('base')
    M3C = importr('M3C')
    res = M3C.M3C(data.T, maxK=maxk, distance="abscorr",
                  clusteralg="hc", removeplots=True,
                  cores=ncores, iters=iters)
    k = np.argmax(list(res[1][3]))+2 # k is actuall k+1 because python is zero indexed
    DV_order = list(base.colnames(res[0][k-2][1]))
    labels = pd.Series(np.array(res[0][k-2][2]).squeeze(), index=DV_order)
    consensus_mat = pd.DataFrame(np.matrix(res[0][k-2][0]),
                                index=DV_order, columns=DV_order)
    return consensus_mat, labels, res

def psychICC(df):
    psych = importr('psych')
    rs = psych.ICC(df)
    return rs

def qgraph_cor(data, glasso=False, gamma=.25):
    qgraph = importr('qgraph')
    cors = qgraph.cor_auto(data)
    if glasso==True:
        EBICglasso = qgraph.EBICglasso(cors, data.shape[0],
                                       returnAllResults=True,
                                       gamma=gamma)
        # figure out the index for the lowest EBIC
        best_index = np.argmin(EBICglasso[1])
        tuning_param = EBICglasso[4][best_index]
        glasso_cors = np.array(EBICglasso[0][0])[:,:,best_index]
        glasso_cors_df = pd.DataFrame(np.matrix(glasso_cors), 
                           index=data.columns, 
                           columns=data.columns)
        return glasso_cors_df, tuning_param
    else:
        cors_df = pd.DataFrame(np.matrix(cors), 
                           index=data.columns, 
                           columns=data.columns)
        return cors_df