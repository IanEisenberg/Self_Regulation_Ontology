#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 14:45:54 2019

@author: ian
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_recent_dataset

results = load_results(get_recent_dataset())

key = 'task'
EFA = results[key].EFA
EFA.get_loading()

HCA = results[key].HCA
HCA_results = HCA.results['EFA5_oblimin']
cluster_DVs = HCA.get_cluster_DVs(inp='EFA5_oblimin')
cluster_loadings = HCA.get_cluster_loading(EFA)

# get colors for plot
colors = sns.hls_palette(len(cluster_DVs.keys()))
mds_colors = {}
for i, (k,v) in enumerate(cluster_DVs.items()):
    for dv in v:
        mds_colors[dv] = colors[i]


# plot loading manfold
loading_distances = HCA_results['clustered_df']
mds_colors = [mds_colors[v] for v in loading_distances.index]
np.random.seed(2000)
trf = TSNE(metric='precomputed')
# do it twice, cause I like the look of the second more!
for _ in range(2):
    trf_out = trf.fit_transform(loading_distances)
size = 16
plt.figure(figsize=(size,size*.75))
plt.scatter(trf_out[:,0], trf_out[:,1], 
            s=size*25,
            facecolor=mds_colors,
            edgecolors='black',
            linewidth=1)
plt.tick_params(axis='both', length=0)
plt.axis('off'); 