# Defines Results and Analysis Classes to run on subsets of data

# imports
from collections import OrderedDict as odict
import glob
from os import makedirs, path
import pandas as pd
import numpy as np
import pickle
import random
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import entropy
from sklearn.preprocessing import scale

from dimensional_structure.prediction_utils import run_prediction
from dimensional_structure.utils import (
        create_factor_tree, distcorr,  find_optimal_components, 
        get_loadings, get_scores_from_subset, get_top_factors, 
        hierarchical_cluster, residualize_baseline
        )
from dimensional_structure.graph_utils import  (get_adj, Graph_Analysis)
from selfregulation.utils.utils import get_behav_data, get_demographics, get_info
from selfregulation.utils.r_to_py_utils import get_attr, get_Rpsych, psychFA

# load the psych R package
psych = get_Rpsych()

# ****************************************************************************
# Peform factor analysis
# ****************************************************************************
class EFA_Analysis:
    def __init__(self, data, data_no_impute=None, boot_iter=1000):
        self.results = {}
        self.data = data
        if data_no_impute is not None:
            self.data_no_impute = data_no_impute
        self.boot_iter=boot_iter
        # global variables to hold certain aspects of the analysis
        self.results['num_factors'] = 1

    # private methods
    def _get_attr(self, attribute, c=None, rotate='oblimin'):
        if c is None:
            c = self.get_c()
            print('# of components not specified, using BIC determined #')
        return get_attr(self.results['factor_tree_Rout_%s' % rotate][c],
                        attribute)
    
    def _thresh_loading(loading, threshold=.2):
        over_thresh = loading.max(1)>threshold
        rejected = loading.index[~over_thresh]
        return loading.loc[over_thresh,:], rejected
    
    def get_factor_reorder(self, c, rotate='oblimin'):
        # reorder factors based on correlation matrix
        phi=get_attr(self.results['factor_tree_Rout_%s' % rotate][c],'Phi')
        if phi is None:
            return list(range(c))
        new_order = list(leaves_list(linkage(squareform(np.round(1-phi,3)))))
        return new_order[::-1] # reversing because it works better for task EFA
            
    # public methods
    def adequacy_test(self, verbose=False):
        """ Determine whether data is adequate for EFA """
        data = self.data
        # KMO test should be > .6
        KMO_MSA = psych.KMO(data.corr())[0][0]
        # barlett test should be significant
        Barlett_p = psych.cortest_bartlett(data.corr(), data.shape[0])[1][0]
        adequate = KMO_MSA>.6 and Barlett_p < .05
        if verbose:
            print('Is the data adequate for factor analysis? %s' % \
                  ['No', 'Yes'][adequate])
        return adequate, {'Barlett_p': Barlett_p, 'KMO': KMO_MSA}
    
    def compute_higher_order_factors(self, c=None, rotate='oblimin'):
        """ Return higher order EFA """
        if c is None:
            c = self.get_c()
            print('# of components not specified, using BIC determined #')
        if ('factor_tree_%s' % rotate in self.results.keys() and 
            c in self.results['factor_tree_Rout_%s' % rotate].keys()):
            # get factor correlation matrix
            scores = get_attr(self.results['factor_tree_Rout_%s' % rotate][c], 'scores')
            phi = pd.DataFrame(np.corrcoef(scores.T))
            # check for correlations
            if np.mean(np.tril(phi, -1)) < 10E-5:
                return
            n_obs = self.data.shape[0]
            labels = list(self.results['factor_tree_%s' % rotate][c].columns)
            BIC_c, BICs = find_optimal_components(phi, 
                                                  metric='BIC', 
                                                  nobs=n_obs)
            if BIC_c != 0:
                if 'factor2_tree_%s' % rotate not in self.results.keys():
                    self.results['factor2_tree_%s' % rotate] = {}
                    self.results['factor2_tree_Rout_%s' % rotate] = {}
                Rout, higher_order_out = psychFA(phi, BIC_c, nobs=n_obs)
                loadings = get_loadings(higher_order_out, labels)
                self.results['factor2_tree_%s' % rotate][c] = loadings
                self.results['factor2_tree_Rout_%s' % rotate][c] = Rout
            else:
                print('Higher order factors could not be calculated')
        else:
            print('No %s factor solution computed yet!' % c)
            
            
    def create_factor_tree(self, start=1, end=None, rotate='oblimin'):
        if end is None:
            end = max(self.get_c(), start)
        ftree, ftree_rout = create_factor_tree(self.data,  (start, end),
                                               rotate=rotate)
        self.results['factor_tree_%s' % rotate] = ftree
        self.results['factor_tree_Rout_%s' % rotate] = ftree_rout
    
    def get_boot_stats(self, c=None, rotate='oblimin'):
        if c is None:
            c = self.get_c()
            print('# of components not specified, using BIC determined #')
        if c in self.results['factor_tree_Rout_%s' % rotate].keys():
            bootstrap_Rout = self.results['factor_tree_Rout_%s' % rotate][c]
            if 'cis' in bootstrap_Rout.names:
                loadings = self.get_loading(c, rotate=rotate)
                bootstrap_stats = get_attr(bootstrap_Rout, 'cis')
                means = pd.DataFrame(get_attr(bootstrap_stats,'means'), 
                                     index=loadings.index,
                                     columns=loadings.columns)
                sds = pd.DataFrame(get_attr(bootstrap_stats,'sds'), 
                                     index=loadings.index,
                                     columns=loadings.columns)
                return {'means': means, 'sds': sds}
            else:
                print('No bootstrap has been run for EFA with %s factors' % c)
                return None
        else:
            print("EFA hasn't been run for %s factors" % c)
            return None

    def get_dimensionality(self, metrics=None, verbose=False):
        """ Use multiple methods to determine EFA dimensionality
        
        Args
            Metrics: A list including a subset of the following strings:
                BIC, parallel, SABIC, and CV. Default [BIC, parallel]
        """
        if metrics is None:
            metrics = ['BIC']
        if 'BIC' in metrics:
            BIC_c, BICs = find_optimal_components(self.data, metric='BIC')
            self.results['c_metric-BIC'] = BIC_c
            self.results['cscores_metric-BIC'] = BICs
        if 'parallel' in metrics:
            # parallel analysis
            parallel_out = psych.fa_parallel(self.data, fa='fa', fm='ml',
                                             plot=False, **{'n.iter': 2})
            parallel_c = parallel_out[parallel_out.names.index('nfact')][0]
            self.results['c_metric-parallel'] = int(parallel_c)
        if 'SABIC' in metrics:
            # using SABIC
            SABIC_c, SABICs = find_optimal_components(self.data, metric='SABIC')
            self.results['c_metric-SABIC'] = SABIC_c
            self.results['cscores_metric-SABIC'] = SABICs
        if 'CV' in metrics:
            try:
                 # using CV
                CV_c, CVs = find_optimal_components(self.data_no_impute, 
                                                    maxc=50, metric='CV')
                self.results['c_metric-CV'] = CV_c
                self.results['cscores_metric-CV'] = CVs
            except AttributeError:
                print("CV dimensionality could not be calculated. " + \
                      "data_no_impute not found.")
        # record max_factors
        best_cs = {k:v for k,v in self.results.items() if 'c_metric-' in k}
        self.results['num_factors'] = BIC_c
        if verbose:
                print('Best Components: ', best_cs)
    
    def get_loading(self, c=None, bootstrap=False, rotate='oblimin',
                    recompute=False, copy=True):
        """ Return the loading for an EFA solution at the specified c """
        if c is None:
            c = self.get_c()
            print('# of components not specified, using BIC determined #')
        n_iter = 1
        if bootstrap:
            n_iter = self.boot_iter
        if 'factor_tree_%s' % rotate not in self.results.keys():
            self.results['factor_tree_%s' % rotate] = {}
            self.results['factor_tree_Rout_%s' % rotate] = {}
        if (not recompute and# recomputing isn't wanted
            c in self.results['factor_tree_%s' % rotate].keys() and # c factors have been computed
            (n_iter==1 or 'cis' in self.results['factor_tree_Rout_%s' % rotate][c].names)):
            if copy:
                return self.results['factor_tree_%s' % rotate][c].copy()
            else:
                return self.results['factor_tree_%s' % rotate][c]
        else:
            print('No %s factor solution computed yet! Computing...' % c)
            fa, output = psychFA(self.data, c, method='ml', rotate=rotate,
                                 n_iter=n_iter)
            loadings = get_loadings(output, labels=self.data.columns)
            self.results['factor_tree_%s' % rotate][c] = loadings
            self.results['factor_tree_Rout_%s' % rotate][c] = fa
            if copy:
                return loadings.copy()
            else:
                return loadings
    
    def get_loading_entropy(self, c=None, rotate='oblimin'):
        if c is None:
            c = self.get_c()
            print('# of components not specified, using BIC determined #')
        assert c>1
        loading = self.get_loading(c, rotate=rotate)
        # calculate entropy of each variable
        loading_entropy = abs(loading).apply(entropy, 1)
        max_entropy = entropy([1/loading.shape[1]]*loading.shape[1])
        return loading_entropy/max_entropy
    
    def get_null_loading_entropy(self, c=None, reps=50, rotate='oblimin'):
        if c is None:
            c = self.get_c()
            print('# of components not specified, using BIC determined #')
        assert c>1
        # get absolute loading
        loading = abs(self.get_loading(c, rotate=rotate))
        max_entropy = entropy([1/loading.shape[1]]*loading.shape[1])
        permuted_entropies = np.array([])
        for _ in range(reps):
            # shuffle matrix
            for i, col in enumerate(loading.values.T):
                shuffle_vec = np.random.permutation(col)
                loading.iloc[:, i] = shuffle_vec
            # calculate entropy of each variable
            loading_entropy = loading.apply(entropy, 1)
            permuted_entropies = np.append(permuted_entropies,
                                           (loading_entropy/max_entropy).values)
        return permuted_entropies
    
    def get_factor_entropies(self, rotate='oblimin'):
        # calculate entropy for each measure at different c's
        entropies = {}
        null_entropies = {}
        for c in self.results['factor_tree_%s' % rotate].keys():
            if c > 1:
                entropies[c] = self.get_loading_entropy(c, rotate=rotate)
                null_entropies[c] = self.get_null_loading_entropy(c, rotate=rotate)
        self.results['entropies_%s' % rotate] = pd.DataFrame(entropies)
        self.results['null_entropies_%s' % rotate] = pd.DataFrame(null_entropies)
        
    def get_metric_cs(self):
        metric_cs = {k:v for k,v in self.results.items() if 'c_metric-' in k}
        return metric_cs
    
    def get_factor_names(self, c=None, rotate='oblimin'):
        if c is None:
            c = self.get_c()
            print('# of components not specified, using BIC determined #')
        return self.get_loading(c, rotate=rotate).columns
    
    def get_c(self):
        return self.results['num_factors']
    
    def get_scores(self, c=None, rotate='oblimin'):
        if c is None:
            c = self.get_c()
            print('# of components not specified, using BIC determined #')
        scores = self._get_attr('scores', c, rotate=rotate)
        names = self.get_factor_names(c, rotate=rotate)
        scores = pd.DataFrame(scores, index=self.data.index,
                              columns=names)
        return scores
        
    def get_task_representations(self, tasks, c=None, rotate='oblimin'):
        """Take a list of tasks and reconstructs factor scores"""   
        if c is None:
            c = self.get_c()
            print('# of components not specified, using BIC determined #')         
        fa_output = self.results['factor_tree_Rout_%s' % rotate][c]
        output = {'weights': get_attr(fa_output, 'weights'),
                  'scores': get_attr(fa_output, 'scores')}
        subset_scores, r2_scores = get_scores_from_subset(self.data,
                                                          output,
                                                          tasks)
        return subset_scores, r2_scores
        
    def get_nesting_matrix(self, explained_threshold=.5, rotate='oblimin'):
        factor_tree = self.results['factor_tree_%s' % rotate]
        explained_scores = -np.ones((len(factor_tree), len(factor_tree)-1))
        sum_explained = np.zeros((len(factor_tree), len(factor_tree)-1))
        for key in self.results['lower_nesting'].keys():
            r =self.results['lower_nesting'][key]
            adequately_explained = r['scores'] > explained_threshold
            explained_score = np.mean(r['scores'][adequately_explained])
            if np.isnan(explained_score): explained_score = 0
            explained_scores[key[1]-1, key[0]-1] = explained_score
            sum_explained[key[1]-1, key[0]-1] = (np.sum(adequately_explained/key[0]))
        return explained_scores, sum_explained
    
    def name_factors(self, labels, rotate='oblimin'):
        loading = self.get_loading(len(labels), rotate=rotate, copy=False)
        loading.columns = labels
    
    def print_top_factors(self, c=None, n=5, rotate='oblimin'):
        if c is None:
            c = self.get_c()
            print('# of components not specified, using BIC determined #')
        tmp = get_top_factors(self.get_loading(c, rotate=rotate), n=n, verbose=True)
      
    def reorder_factors(self, mat, rotate='oblimin'):
        c = mat.shape[1]
        reorder_vec = self.get_factor_reorder(c, rotate=rotate)
        if type(mat) == pd.core.frame.DataFrame:
            mat = mat.iloc[:, reorder_vec]
        else:
            mat = mat[reorder_vec][:, reorder_vec]
        return mat
    
    def run(self, loading_thresh=None, rotate='oblimin', 
            bootstrap=False, verbose=False):
        if 'EFA_adequacy' not in self.results:
            # check adequacy
            adequate, adequacy_stats = self.adequacy_test(verbose)
            assert adequate, "Data is not adequate for EFA!"
            self.results['EFA_adequacy'] = {'adequate': adequate, 
                                            'adequacy_stats': adequacy_stats}
        
        # get optimal dimensionality
        if 'c_metric-BIC' not in self.results.keys():
            if verbose: print('Determining Optimal Dimensionality')
            self.get_dimensionality(verbose=verbose)
            
        # create factor tree
        if verbose: print('Creating Factor Tree')
        self.get_loading(c=self.get_c(), rotate=rotate,
                         bootstrap=bootstrap)
        # optional threshold
        if loading_thresh is not None:
            for c, loading in self.results['factor_tree_%s' % rotate].items():
                thresh_loading = self._thresh_loading(loading, loading_thresh)
                self.results['factor_tree_%s' % rotate][c], rejected = thresh_loading
        # get higher level factor solution
        if verbose: print('Determining Higher Order Factors')
        self.compute_higher_order_factors(rotate=rotate)
        # get entropies
        self.get_factor_entropies(rotate=rotate)
    
    def verify_factor_solution(self):
        fa, output = psychFA(self.data, 10)
        scores = output['scores'] # factor scores per subjects derived from psychFA
        scaled_data = scale(self.data)
        redone_scores = scaled_data.dot(output['weights'])
        redone_score_diff = np.mean(scores-redone_scores)
        assert(redone_score_diff < 1e-5)
            
class HCA_Analysis():
    """ Runs Hierarchical Clustering Analysis """
    def __init__(self, dist_metric):
        self.results = {}
        self.dist_metric = dist_metric
        self.metric_name = 'unknown'
        if self.dist_metric == distcorr:
            self.metric_name = 'distcorr'
        else:
            self.metric_name = self.dist_metric
        
    def cluster_data(self, data, dist_metric=None, method='average'):
        if dist_metric is None:
            dist_metric = self.dist_metric
            label_append = ''
        else:
            label_append = '_dist-%s' % dist_metric
        output = hierarchical_cluster(data.T, method=method,
                                      pdist_kws={'metric': dist_metric})
        self.results['data%s' % label_append] = output
        
    def cluster_EFA(self, EFA, c, dist_metric=None, min_cluster_size=3,
                    method='average', rotate='oblimin'):
        if dist_metric is None:
            dist_metric = self.dist_metric
            label_append = ''
        else:
            label_append = '_dist-%s' % dist_metric
        loading = EFA.get_loading(c, rotate=rotate)
        output = hierarchical_cluster(loading, method=method,
                                      min_cluster_size=min_cluster_size,
                                      pdist_kws={'metric': dist_metric})
        self.results['EFA%s_%s%s' % (c, rotate, label_append)] = output
        
    def get_cluster_DVs(self, inp='data'):
        names = self.get_cluster_names(inp=inp)
        cluster = self.results['%s' % inp]
        DVs = cluster['clustered_df'].index
        reorder_vec = cluster['reorder_vec']
        cluster_labels = cluster['labels'][reorder_vec]
        cluster_DVs= [[DVs[i] for i,index in enumerate(cluster_labels) \
                           if index == j] for j in np.unique(cluster_labels)]
        cluster_DVs_dict = odict()
        for name, dv in zip(names, cluster_DVs):
            cluster_DVs_dict[name] = dv
        return cluster_DVs_dict
    
    def build_graphs(self, inp, graph_data):
        """ Build graphs from clusters from HCA analysis
        Args:
            inp: the input label used for the hierarchical analysis
            graph_data: the data to subset based on the clusters found using
                inp. This data will be passed to a graph analysis. E.G, 
                graph_data can be the original data matrix or a EF embedding
        """
        cluster_labels = self.get_cluster_DVs(inp)
        graphs = []
        for cluster in cluster_labels:
            if len(cluster)>1:
                subset = graph_data.loc[:,cluster]
                cor = get_adj(subset, 'abs_pearson')
                GA = Graph_Analysis()
                GA.setup(adj = cor)
                GA.calculate_centrality()
                graphs.append(GA)
            else:
                graphs.append(np.nan)
        return graphs
    
    def get_cluster_loading(self, EFA, c=None, rotate='oblimin'):
        if c is None:
            c = EFA.get_c()
        inp = 'EFA%s_%s' % (c, rotate)
        cluster_labels = self.get_cluster_DVs(inp)
        cluster_loadings = odict({})
        for name, cluster in cluster_labels.items():
            subset = abs(EFA.get_loading(c, rotate=rotate).loc[cluster,:])
            cluster_vec = subset.mean(0)
            cluster_loadings[name] = cluster_vec
        return cluster_loadings
    
    def get_cluster_names(self, inp='data'):
        cluster = self.results['%s' % inp]
        num_clusters = np.max(cluster['labels'])
        if 'cluster_names' in cluster.keys():
            return cluster['cluster_names']
        else:
            return [str(i+1) for i in range(num_clusters)]
        
    def get_graph_vars(self, graphs):
        """ returns variables for each cluster sorted by centrality """
        graph_vars = []
        for GA in graphs:
            g_vars = [(i['name'], i['eigen_centrality']) for i in list(GA.G.vs)]
            sorted_vars = sorted(g_vars, key = lambda x: x[1])
            graph_vars.append(sorted_vars)
        return graph_vars
    
    def name_clusters(self, names, inp):
        cluster_labels = self.results[inp]['labels']
        num_clusters = np.max(cluster_labels)
        assert len(names) == num_clusters
        self.results[inp]['cluster_names'] = names
        
    def run(self, data, EFA, cluster_EFA=False, rotate='oblimin',
            run_graphs=False, verbose=False):
        if verbose: print("Clustering data")
        self.cluster_data(data)
        if cluster_EFA:
            if verbose: print("Clustering EFA")
            self.cluster_EFA(EFA, EFA.get_c(),
                             rotate=rotate)
        if run_graphs == True:
            # run graph analysis on raw data
            graphs = self.build_graphs('data', data)
            self.results['data']['graphs'] = graphs

class Demographic_Analysis(EFA_Analysis):
    """ Runs Hierarchical Clustering Analysis """
    def __init__(self, data, residualize=True, residualize_vars=['Age', 'Sex'],
                 boot_iter=1000):
        self.raw_data = data
        self.residualize_vars = residualize_vars
        if residualize:
            data = residualize_baseline(data, self.residualize_vars)
        if 'BMI' in data.columns:
            data.drop(['WeightPounds', 'HeightInches'], axis=1, inplace=True)
        
        super().__init__(data, boot_iter=boot_iter)
    
    def get_change(self, retest_dataset):
        demographics = self.data
        
        retest = get_demographics(retest_dataset)
        retest = residualize_baseline(retest, self.residualize_vars)
        if 'BMI' in retest.columns:
            retest.drop(['WeightPounds', 'HeightInches'], axis=1, inplace=True)
        # get common variables
        common_index = sorted(list(set(demographics.index) & set(retest.index)))
        common_columns = sorted(list(set(demographics.columns) & set(retest.columns)))
        demographics = demographics.loc[common_index, common_columns] 
        retest = retest.loc[common_index, common_columns]
        raw_change = retest-demographics
        # convert to scores
        c = self.get_c()
        demographic_factor_weights = get_attr(self.results['factor_tree_Rout_oblimin'][c],'weights')
        demographic_scores = scale(demographics).dot(demographic_factor_weights)
        retest_scores = scale(retest).dot(demographic_factor_weights)
        
        
        factor_change = pd.DataFrame(retest_scores-demographic_scores,
                              index=common_index,
                              columns = self.get_scores().columns)
        factor_change = self.reorder_factors(factor_change)
        factor_change.columns = [i + ' Change' for i in factor_change.columns]
        return factor_change, raw_change
    
class Results(EFA_Analysis, HCA_Analysis):
    """ Class to hold olutput of EFA, HCA and graph analyses """
    def __init__(self, 
                 datafile=None, 
                 loading_thresh=None,
                 dist_metric=distcorr,
                 boot_iter=1000,
                 name='',
                 filter_regex='.',
                 ID=None,
                 results_dir=None,
                 residualize_vars=['Age', 'Sex'],
                 saved_obj_file=None
                 ):
        """
        Args:
            datafile: name of a directory in "Data"
            loading_thresh: threshold to use for factor analytic result
            dist_metric: distance metric for hierarchical clustering that is 
            passed to pdist
            name: string to append to ID, default to empty string
            filter_regex: regex string passed to data.filter
            ID: specify if a specific ID is desired
            results_dir: where to save results
        """
        assert datafile is not None or saved_obj_file is not None
        # initialize with the saved object if available
        if saved_obj_file:
            self._load_init(saved_obj_file)
        else:
            # set vars
            self.dataset = datafile
            self.loading_thresh = None
            self.dist_metric = dist_metric
            self.boot_iter = boot_iter
            self.residualize_vars = residualize_vars
            if ID is None:
                self.ID =  '%s_%s' % (name, str(random.getrandbits(16)))
            else:
                self.ID = '%s_%s' % (name, str(ID))
            # set up output files
            self.results_dir = results_dir
            # load data
            self.data = get_behav_data(dataset=datafile, 
                                      file='meaningful_variables_imputed.csv',
                                      filter_regex=filter_regex,
                                      verbose=True)
            self.data_no_impute = get_behav_data(dataset=datafile,
                                                 file='meaningful_variables_clean.csv',
                                                 filter_regex=filter_regex,
                                                 verbose=True)
            self.demographics = get_demographics()
            
        
        # initialize analysis classes
        self.DA = Demographic_Analysis(self.demographics, 
                                       residualize_vars=self.residualize_vars,
                                       boot_iter=self.boot_iter)
        self.EFA = EFA_Analysis(self.data, 
                                self.data_no_impute, 
                                boot_iter=self.boot_iter)
        self.HCA = HCA_Analysis(dist_metric=self.dist_metric)
        
        # load the results from the saved object
        if saved_obj_file:
            self._load_results(saved_obj_file)
    
    def get_output_dir(self):
        if self.results_dir is None:
            results_dir = get_info('results_directory')
        else:
            results_dir = self.results_dir
        output_dir = path.join(results_dir, 'dimensional_structure', 
                               self.dataset, 'Output', self.ID)
        makedirs(output_dir, exist_ok = True)
        return output_dir
        
    def get_plot_dir(self):
        if self.results_dir is None:
            results_dir = get_info('results_directory')
        else:
            results_dir = self.results_dir
        plot_dir = path.join(results_dir, 'dimensional_structure', 
                             self.dataset, 'Plots', self.ID)
        makedirs(plot_dir, exist_ok = True)
        return plot_dir
        
    def run_demographic_analysis(self, bootstrap=False, verbose=False):
        if verbose:
            print('*'*79)
            print('Running demographics')
            print('*'*79)
        self.DA.run(bootstrap=bootstrap, verbose=verbose)
        
    def run_EFA_analysis(self, rotate='oblimin', bootstrap=False, verbose=False):
        if verbose:
            print('*'*79)
            print('Running EFA, rotate: %s' % rotate)
            print('*'*79)
        self.EFA.run(self.loading_thresh, rotate=rotate, 
                     bootstrap=bootstrap, verbose=verbose)

    def run_clustering_analysis(self, cluster_EFA=True, run_graphs=True,
                                rotate='oblimin',  dist_metric=None, verbose=False):
        """ Run HCA Analysis
        
        Args:
            dist_metric: if provided, create a new HCA instances with the
                provided dist_metric and return it. If None (default) run
                the results' internal HCA with the dist_metric provided
                at creation
        """
        if verbose:
            print('*'*79)
            print('Running HCA')
            print('*'*79)
        if dist_metric is None: 
            self.HCA.run(self.data, self.EFA, cluster_EFA=cluster_EFA,
                         rotate=rotate, run_graphs=run_graphs, verbose=verbose)
        else:
            HCA = HCA_Analysis(dist_metric=dist_metric)
            HCA.run(self.data, self.EFA, cluster_EFA=cluster_EFA,
                    rotate=rotate, run_graphs=run_graphs, verbose=verbose)
            return {'HCA': HCA}
    
    def run_prediction(self, shuffle=False, classifier='lasso',
                       include_raw_demographics=False, rotate='oblimin',
                       verbose=False):
        if verbose:
            print('*'*79)
            print('Running Prediction, shuffle: %s, classifier: %s' % (shuffle, classifier))
            print('*'*79)
        factor_scores = self.EFA.get_scores(rotate=rotate)
        demographic_factors = self.DA.reorder_factors(self.DA.get_scores())
        c = factor_scores.shape[1]
        # get raw data reorganized by clustering
        clustering=self.HCA.results['EFA%s_%s' % (c, rotate)]
        labels = clustering['clustered_df'].columns
        raw_data = self.data[labels]
        
        targets = [('demo_factors', demographic_factors)]
        if include_raw_demographics:
            targets.append(('demo_raw', self.demographics))
        for target_name, target in targets:
            for predictors in [('EFA%s_%s' % (c, rotate), factor_scores), ('raw', raw_data)]:
                # predicting using best EFA
                if verbose: print('**Predicting using %s**' % predictors[0])
                run_prediction(predictors[1], 
                               target, 
                               self.get_output_dir(),
                               outfile='%s_%s_prediction' % (predictors[0], target_name), 
                               shuffle=shuffle,
                               classifier=classifier, 
                               verbose=verbose)

    def run_change_prediction(self, shuffle=False, classifier='lasso',
                   include_raw_demographics=False, rotate='oblimin',
                   verbose=False):
        if verbose:
            print('*'*79)
            print('Running Change Prediction, shuffle: %s, classifier: %s' % (shuffle, classifier))
            print('*'*79)
        factor_scores = self.EFA.get_scores(rotate=rotate)
        c = factor_scores.shape[1]
        # get raw data reorganized by clustering
        clustering=self.HCA.results['EFA%s_%s' % (c, rotate)]
        labels = clustering['clustered_df'].columns
        raw_data = self.data[labels]
        
        # get change scores
        factor_change, raw_change = self.DA.get_change(self.dataset.replace('Complete', 'Retest'))
        
        # predict
        targets = [('demo_factors_change', factor_change)]
        if include_raw_demographics:
            targets.append(('demo_raw_change', raw_change))
        for target_name, target in targets:
            for predictors in [('EFA%s_%s' % (c, rotate), factor_scores), ('raw', raw_data)]:
                # predicting using best EFA
                if verbose: print('**Predicting using %s**' % predictors[0])
                run_prediction(predictors[1], 
                               target, 
                               self.get_output_dir(),
                               outfile='%s_%s_prediction' % (predictors[0], target_name), 
                               shuffle=shuffle,
                               classifier=classifier, 
                               verbose=verbose)

        
    def get_prediction_files(self, EFA=True, shuffle=True, change=False):
        prefix = 'EFA' if EFA else 'raw'
        output_dir = self.get_output_dir()
        prediction_files = glob.glob(path.join(output_dir,
                                               'prediction_outputs',
                                               '%s*' % prefix))
        if shuffle:
            prediction_files = [f for f in prediction_files if 'shuffle' in f]
        else:
            prediction_files = [f for f in prediction_files if 'shuffle' not in f]
            
        if change:
            prediction_files = [f for f in prediction_files if 'change' in f]
        else:
            prediction_files = [f for f in prediction_files if 'change' not in f]
        return prediction_files
    
    def load_prediction_object(self, ID=None, shuffle=False, EFA=True, 
                               change=False,  classifier='lasso', 
                               rotate='oblimin'):
        prediction_files = self.get_prediction_files(EFA, shuffle, change)
        prediction_files = [f for f in prediction_files if classifier in f]
        if EFA==True:
            prediction_files = [f for f in prediction_files if rotate in f]
        # sort by time
        if ID is not None:
            filey = [i for i in prediction_files if ID in i][0]
        else:
            prediction_files.sort(key=path.getmtime)
        if len(prediction_files)>0:
            filey = prediction_files[-1]
            behavpredict = pickle.load(open(filey,'rb'))
        else:
            behavpredict = None
        return behavpredict

    def set_EFA(self, EFA):
        """ replace current EFA object with another """
        self.EFA = EFA
        
    def set_HCA(self, HCA):
        """ replace current EFA object with another """
        self.HCA = HCA
        
    # save and load functions
                 

    def save_results(self, save_dir=None):
        save_obj = {}
        # init vars
        info = {
                'dist_metric': self.dist_metric,
                'loading_thresh': self.loading_thresh,
                'boot_iter': self.boot_iter,
                'ID': self.ID,
                'results_dir': self.results_dir,
                'residualize_vars': self.residualize_vars,
                'dataset': self.dataset
                }
        # data
        data = {'data': self.data,
                'data_no_impute': self.data_no_impute,
                'demographics': self.demographics}
        # results
        results = {}
        results['DA'] = self.DA.results
        results['EFA'] = self.EFA.results
        results['HCA'] = self.HCA.results
        
        save_obj['info'] = info
        save_obj['data'] = data
        save_obj['results'] = results
        if save_dir is None:
            save_dir = self.get_output_dir()
        filey = path.join(save_dir, '%s_results.pkl' % self.ID)
        pickle.dump(save_obj, open(filey,'wb'))
        return filey
    
    def _load_init(self, filey):
        save_obj = pickle.load(open(filey, 'rb'))
        info = save_obj['info']
        data = save_obj['data']
        self.__dict__.update(info)
        self.__dict__.update(data)
        
    def _load_results(self, filey):
        save_obj = pickle.load(open(filey, 'rb'))
        results = save_obj['results']
        self.DA.results = results['DA']
        self.EFA.results = results['EFA']
        self.HCA.results = results['HCA']
