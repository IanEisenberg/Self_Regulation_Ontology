#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 19:48:18 2018

@author: ian
"""
import numpy as np
from os import makedirs, path
import pandas as pd
import pickle
from sklearn.covariance import GraphLassoCV
from sklearn.preprocessing import scale

from dimensional_structure.graph_utils import Graph_Analysis
from selfregulation.utils.utils import get_behav_data,  get_recent_dataset
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.r_to_py_utils import qgraph_cor
dataset = get_recent_dataset()
data = get_behav_data(dataset=dataset, file='meaningful_variables_imputed.csv')
all_results = load_results(dataset)
def get_EFA_HCA(results, EFA):
    if EFA == False:
        return results.HCA.results['data']
    else:
        c = results.EFA.results['num_factors']
        return results.HCA.results['EFA%s_oblimin' % c]

EFA=True
survey_HCA = get_EFA_HCA(all_results['survey'], EFA)
survey_order = survey_HCA['reorder_vec']
task_HCA = get_EFA_HCA(all_results['task'], EFA)
task_order = task_HCA['reorder_vec']


all_data = pd.concat([all_results['task'].data.iloc[:, task_order], 
                      all_results['survey'].data.iloc[:, survey_order]], 
                    axis=1)
out, tuning = qgraph_cor(all_data, glasso=True, gamma=.5)

# recreate with sklearn just to check
data = scale(all_data)
clf = GraphLassoCV()
clf.fit(data)

sklearn_covariance = clf.covariance_[np.tril_indices_from(clf.covariance_)]
qgraph_covariance = out.values[np.tril_indices_from(out)]
method_correlation = np.corrcoef(sklearn_covariance, qgraph_covariance)[0,1]
assert method_correlation > .99

def add_attributes(g):
    g.vs['measurement'] = ['task']*len(task_order) + ['survey']*len(survey_order)
    task_clusters = task_HCA['labels'][task_order]
    survey_clusters = survey_HCA['labels'][survey_order] + max(task_clusters)
    g.vs['cluster'] = np.append(task_clusters, survey_clusters)
    
save_loc = path.join(path.dirname(all_results['task'].get_output_dir()), 
                     'graph_results')
makedirs(save_loc, exist_ok=True)
# unweighted
g = Graph_Analysis()
g.setup(abs(out), weighted=False)
add_attributes(g.G)
g.save_graph(path.join(save_loc, 'graph.graphml'), 'graphml')
pickle.dump(g, open(path.join(save_loc, 'graph.pkl'), 'wb'))
# weighted
g = Graph_Analysis()
g.setup(abs(out), weighted=True)
add_attributes(g.G)
g.save_graph(path.join(save_loc, 'weighted_graph.graphml'), 'graphml')
pickle.dump(g, open(path.join(save_loc, 'weighted_graph.pkl'), 'wb'))

"""
gephi settings using weighted graph

Yifan Hu Layout
-Optimal Distance 200
-Relative Strength .1
-Default everything else 
-Edge Weights filter at .01

Force Atlas Layout
...just play with it until it looks right. I started with
Yifan Hu to get it in a relatively nice layout, then used force atlas
-Edge weights filter at .05
"""



