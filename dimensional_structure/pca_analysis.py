#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 22:54:25 2019

@author: ian
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, factor_analysis
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_behav_data, get_recent_dataset

dataset = get_recent_dataset()
data = get_behav_data(file='meaningful_variables_imputed.csv')
results = load_results(dataset)

data_pca = PCA()
data_pc = data_pca.fit_transform(data)
data_pca.explained_variance_ratio_.round(3)[0:5]


def run_PCA(data):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    data_pca = PCA()
    data_pcs = data_pca.fit_transform(scaled)
    return data_pca, data_pcs


# pca for psychology measures
pcs = {}
pcs['task'] = run_PCA(results['task'].data)
pcs['survey'] = run_PCA(results['survey'].data)
pcs['demo'] = run_PCA(results['survey'].DA.data)

pc_df = pd.DataFrame([pcs['task'][1][:,0],
                      pcs['survey'][1][:,0],
                      pcs['demo'][1][:,0]],
                    columns=results['task'].data.index,
                    index=['task_PC1', 'survey_PC1', 'demo_PC1']).T
                      
# IQ
ravens = data.loc[:,'ravens.score']
ravens_pc_corr = pd.concat([pc_df, ravens], axis=1).corr().loc['ravens.score']

# compare factor analysis to pcs
merged = {}
for label in ['task', 'survey']:
    scores = results[label].EFA.get_scores()
    N=3 #scores.shape[1]
    tmp = pd.DataFrame(pcs[label][1][:,:N],
                       columns=['PC-%s' % x for x in range(1,N+1)],
                       index=scores.index)
    merged[label] = scores.join(tmp)
# outcomes
scores = results['task'].DA.get_scores()
N=3 #scores.shape[1]
tmp = pd.DataFrame(pcs['demo'][1][:,:N],
                   columns=['PC-%s' % x for x in range(1,N+1)],
                   index=scores.index)
merged['outcome'] = scores.join(tmp)

N = 3
f, axes = plt.subplots(3,1, figsize=(6,15))
plt.subplots_adjust(hspace=.2)
for i, (ax, label) in enumerate(zip(axes, ['task','survey','outcome'])):
    corr = abs(merged[label].corr()).T.iloc[:-3, -3:]
    corr.sort_values(by='PC-1', ascending=False, inplace=True)
    cbar = True if i==0 else False
    sns.heatmap(corr, annot=True,
                ax=ax, cbar=cbar, vmin=0, vmax=1,
                cmap=sns.light_palette((15, 75, 50), input='husl', n_colors=100, as_cmap=True),
                annot_kws={'fontsize':10},
                cbar_kws={'ticks': [0,.25,.5,.75,1]})
    ax.tick_params(labelsize=15, rotation=0)
    ax.set_ylabel(label.title(), fontsize=20, fontweight='bold')
f.align_ylabels()
f.savefig('/home/ian/tmp/PC_EFA.pdf', dpi=300, bbox_inches='tight')
