#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 14:45:54 2019

@author: ian
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_recent_dataset

results = load_results(get_recent_dataset())

key = 'task'
data = results[key].data
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
label_colors = sns.hls_palette(3, .46, .5,s=.7)

trf = TSNE(metric='precomputed')
# do it twice, cause I like the look of the second more!
for _ in range(2):
    trf_out = trf.fit_transform(loading_distances)
trf_out = pd.DataFrame(trf_out, index=loading_distances.index)



# plot tsne
tsne_colors = [.6,.6,.6] # mds_colors
size = 10
f = plt.figure(figsize=(size,size*.75))
plt.scatter(trf_out[0], trf_out[1], 
            s=size*25,
            facecolor=tsne_colors,
            edgecolors='k',
            alpha=.5,
            linewidth=1)
plt.tick_params(axis='both', length=0)
plt.axis('off');

# additions
span = trf_out.filter(regex='digit_span.forward_span|raven|^stop_signal.*SSRT_low',axis=0)
plt.scatter(span[0], span[1],
            s=size*50,
            facecolor=label_colors,
            edgecolors='black',
            linewidth=1)


def add_axis(x_var, y_var, rect, color=None):
    ax = f.add_axes(rect, frameon=True, alpha=0)
    ax.patch.set_visible(False)
    index = loading_distances.index.get_loc(y_var)
    if color is None:
        color = tsne_colors[index]
    sns.regplot(x_var, y_var, data=data, ax=ax,
                scatter_kws={'s': size, 'alpha':.2,
                             'color': color},
                line_kws={'color': color})
    sns.despine(ax=ax)
    ax.spines['left'].set_linewidth(size/7)
    ax.spines['bottom'].set_linewidth(size/7)
    ax.set_xlabel(' '.join(x_var.split('.')[0].split('_')), 
                  color=label_colors[2], fontsize=size*2)
    ax.set_ylabel(' '.join(y_var.split('.')[0].split('_')), 
                  color=color, fontsize=size*2)
    ax.tick_params(labelleft=False, labelbottom=False,
                   left=False, bottom=False)
    # set aspect
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))
        
x_var = 'digit_span.forward_span'
y_var =  'ravens.score'
rect = [.38,.85,.22, .22]
add_axis(x_var, y_var, rect, color=label_colors[1])


x_var = 'digit_span.forward_span'
y_var =  'stop_signal.SSRT_low'
rect = [.55,.03,.22, .22]
add_axis(x_var, y_var, rect, color=label_colors[0])

f.savefig('/home/ian/tmp/mapping_plot.png', transparent=True,
          bbox_inches='tight')
