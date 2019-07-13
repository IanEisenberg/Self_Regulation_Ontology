from collections import OrderedDict as odict
import math
import numpy as np
from os import path
import pandas as pd
import pickle
from sklearn.metrics import adjusted_mutual_info_score
from scipy.spatial.distance import pdist, squareform

from dimensional_structure.utils import get_loadings, hierarchical_cluster
from ontology_mapping.reconstruction_utils import reorder_FA
from selfregulation.utils.r_to_py_utils import psychFA
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_recent_dataset

# helper functions
def drop_EFA(data, measures, c):
    
    to_drop = data.filter(regex='|'.join(measures)).columns
    subset = data.drop(to_drop, axis=1)
    fa, output = psychFA(subset, c, method='ml', rotate='oblimin')
    loadings = get_loadings(output, labels=subset.columns)
    return loadings

def tril(square):
    indices = np.tril_indices_from(square, -1)
    return square[indices]
    
def print_stats(consensusClust):
    m = np.mean(consensusClust.AMI_scores)
    sd = np.std(consensusClust.AMI_scores)
    print('AMI mean: %s, std: %s' % (m, sd))
    print('AMI compared to consensus: %s' % consensusClust.compare_clusters())
    
    cooccurence = consensusClust.evaluate_relative_cooccurence().values()
    within = np.mean([i[0] for i in cooccurence])
    across = np.mean([i[1] for i in cooccurence])
    closest = np.mean([i[2] for i in cooccurence])
    print('Cooccurence, within: %s, across: %s, closest: %s' % (within, across, closest))
    
class ConsensusCluster():
    def __init__(self, results, percent_vars=1,
                 method='average', distance_metric='abscorrelation'):
        self.results = results
        self.total_vars = len(results.data.columns)
        self.num_vars = math.ceil(percent_vars*len(results.data.columns))
        self.method = method
        self.distance_metric = distance_metric
        # to store consensus clustering
        self.consensus_clustering = None
        # get original clustering
        self.orig_clustering = hierarchical_cluster(results.EFA.get_loading(),
                                  method=self.method,
                                  min_cluster_size=3,
                                  pdist_kws={'metric': self.distance_metric})
        # adjusted mutual information scores across simulations
        self.AMI_scores = []
        
    def simLoading(self):
        c = self.results.EFA.get_c()
        stats = self.results.EFA.get_boot_stats(c=c)
        loadings = np.random.normal(size=stats['means'].shape)*stats['sds']+stats['means']
        return loadings
    
    def sim_cluster(self):
        loadings = self.simLoading()
        kept_vars = np.random.choice(self.total_vars, self.num_vars, replace=False)
        kept_vars = sorted(kept_vars)
        loadings = loadings.iloc[kept_vars,:]
        clustering = hierarchical_cluster(loadings,
                                  method=self.method,
                                  min_cluster_size=3,
                                  pdist_kws={'metric': self.distance_metric})
        tmp_labels = clustering['labels']
        labels = np.ones(self.total_vars)*np.nan
        for i, index in enumerate(kept_vars):
            labels[index] = tmp_labels[i]
            
        cooccurence = self.convert_cooccurence(labels)
        # calculated AMI scores ignoring nan
        orig = self.orig_clustering['labels']
        sim = labels
        not_nans = np.logical_not(np.isnan(sim))
        score = adjusted_mutual_info_score(orig[not_nans], sim[not_nans])
        self.AMI_scores.append(score)
        return cooccurence
    
    def convert_cooccurence(self, labels):
        """ Convert a set of labels to a cooccurence matrix
        
        labels: class labels
        new_index: indices of labels (could be subset of total)
        size: total size of possible cooccurence matrix
        """
        mat = np.ones((len(labels), len(labels))) * np.nan
        kept_indices = np.where(np.logical_not(np.isnan(labels)))[0]
        for i, val in zip(kept_indices, labels[kept_indices]):
            mat[i, kept_indices] = (labels[kept_indices]==val)
        return mat
        """
        # extract lower triangle
        tril_indices = np.tril_indices_from(mat, -1)
        return mat[tril_indices]
        """
    
    def sim_clusters(self, reps=100):
        """
        clusterings = np.zeros((reps, self.total_vars*(self.total_vars-1)//2))
        for i in range(reps):
            clusterings[i,:] = self.simCluster()
        """
        clusterings = np.zeros((self.total_vars, self.total_vars, reps))
        for i in range(reps):
            clusterings[:,:, i] = self.sim_cluster()
        return clusterings
    
    def calc_consensus_cluster(self, reps=100):
        sims = self.sim_clusters(reps)
        cooccurence_distance = 1-np.nanmean(sims,2)
        dist_df = pd.DataFrame(cooccurence_distance,
                               columns = self.results.data.columns,
                               index = self.results.data.columns)
        clustering = hierarchical_cluster(dist_df,
                          method=self.method,
                          min_cluster_size=3,
                          compute_dist=False)
        self.consensus_clustering = clustering
    
    def compare_clusters(self):
        # compares original to consensus clustering
        if self.consensus_clustering is None:
            print("First run consensusCluster!")
            return
        else:
            orig_labels = self.orig_clustering['labels']
            new_labels = self.consensus_clustering['labels']
        return adjusted_mutual_info_score(orig_labels, new_labels)
    
    def get_orig_cluster(self):
        return self.orig_clustering
    
    def get_consensus_cluster(self):
        return self.consensus_clustering
    
    def evaluate_relative_cooccurence(self):
        c = self.results.EFA.get_c()
        inp = 'EFA%s_oblimin' % c
        HCA = self.results.HCA
        cooccurence = self.consensus_clustering['distance_df']
        relative_cooccurence = odict({})
        for cluster, DVs in HCA.get_cluster_DVs(inp).items():
            nearest_clusters, nearest_DVs = self._get_nearest_clusters(HCA, inp, cluster)
            intra_subset = tril(cooccurence.loc[DVs, DVs].values)
            inter_subset = cooccurence.drop(DVs, axis=1).loc[DVs]
            nearest_subset = cooccurence.loc[DVs, nearest_DVs]
            relative_cooccurence[cluster] = (1-np.mean(intra_subset),
                                             1-np.mean(inter_subset.values.flatten()),
                                             1-np.mean(nearest_subset.values.flatten()))
        return relative_cooccurence
    
    def _get_nearest_clusters(self, HCA, inp, cluster):
        names, DVs = zip(*HCA.get_cluster_DVs(inp).items())
        cluster_i = names.index(cluster)
        nearest_clusters = []
        nearest_DVs = []
        i_1 = cluster_i-1 if cluster_i>0 else cluster_i+2
        i_2 = cluster_i+1 if cluster_i+1 < len(names) else cluster_i-2
        nearest_clusters.append(names[i_1])
        nearest_DVs.extend(DVs[i_1])
        nearest_clusters.append(names[i_2])
        nearest_DVs.extend(DVs[i_2])
        return nearest_clusters, nearest_DVs
    

if __name__ == '__main__':
    # EFA robustness
    # Check to see how sensitive the EFA solution is to any single measure
    results = load_results(get_recent_dataset())
    for result in results.values():
        output_dir = result.get_output_dir()
        c = result.EFA.get_c()
        orig_loadings = result.EFA.get_loading(c=c)
        measures = np.unique([c.split('.')[0] for c in result.data.columns])
        # drop a single measure
        factor_correlations = {}
        for measure in measures:
            data = result.data
            new_loadings = drop_EFA(data, [measure], c)
            new_loadings = reorder_FA(orig_loadings.loc[new_loadings.index], 
                                      new_loadings, thresh=-1)
            
            corr = pd.concat([new_loadings, orig_loadings], axis=1, sort=False) \
                    .corr().iloc[:c, c:]
            diag = {c:abs(i) for c,i in zip(new_loadings.columns, np.diag(corr))}
            factor_correlations[measure] = diag
    
        
        # save pair factor correlations
        save_file = path.join(output_dir, 'EFAdrop_robustness.pkl')
        to_save = factor_correlations
        pickle.dump(to_save, open(save_file, 'wb'))
    
    
    # cluster robustness
    # simulate based on EFA bootstraps and dropping variables
    sim_reps = 5000
    for name, result in results.items():
        output_dir = result.get_output_dir()
        save_file = path.join(output_dir, 'cluster_robustness.pkl')
        consensusClust = ConsensusCluster(result, percent_vars=.8)
        consensusClust.calc_consensus_cluster(sim_reps)
        print(consensusClust.compare_clusters())
        relative_cooccurence = consensusClust.evaluate_relative_cooccurence()
    
        # saving
        pickle.dump({'consensusClust': consensusClust}, 
            open(save_file, 'wb'))