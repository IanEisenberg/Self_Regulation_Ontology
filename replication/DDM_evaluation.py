import math
import matplotlib.pyplot as plt
import numpy as np
from os import path
import pandas as pd
import seaborn as sns
from selfregulation.utils.plot_utils import beautify_legend, format_num, format_variable_names
from selfregulation.utils.utils import get_behav_data, get_demographics, get_info

base_dir = get_info('base_directory')
ext= 'png'
data = get_behav_data(file='variables_exhaustive.csv')

# get DDM tasks
tasks = np.unique([t.split('.')[0] for t in data.filter(regex='_hddm').columns])

# reduce data to just DDM tasks
data = data.loc[:,[c.split('.')[0] in tasks for c in data.columns]]

def plot_vars(tasks, contrasts, axes=None,  xlabel='Value', standardize=False):
    colors = sns.hls_palette(4)
    desat_colors = [sns.desaturate(c, .5) for c in colors]        
    for i, task in enumerate(tasks):
        subset = contrasts.filter(regex='^'+task)
        if subset.shape[1] != 0:
            if standardize:
                subset = subset/subset.std()
            subset.columns = [c.split('.')[1] for c in subset.columns]
            subset.columns = format_variable_names(subset.columns)
            # add mean value to columns
            means = subset.mean()
            subset.columns = [subset.columns[i]+': %s' % format_num(means.iloc[i]) for i in range(len(means))]
            subset = subset.melt(var_name = 'Variable', value_name='Value')
            
            sns.stripplot(x='Value', y='Variable', hue='Variable', ax=axes[i],
                        data=subset, palette=desat_colors, jitter=True, alpha=.75)
            # plot central tendency
            N = len(means)
            axes[i].scatter(means, range(N), s=200, 
                            c=colors[:N], 
                            edgecolors='white',
                            linewidths=2,
                            zorder=3)
            
            # add legend
            leg = axes[i].get_legend()
            leg.set_title('')
            beautify_legend(leg, colors=colors, fontsize=14)
            # change axes
            max_val = subset.Value.abs().max()
            axes[i].set_xlim(-max_val, max_val)
            axes[i].set_xlabel(xlabel, fontsize=16)
            axes[i].set_ylabel('')
            axes[i].set_yticklabels('')
        axes[i].set_title(format_variable_names([task])[0].title(), fontsize=20)
    plt.subplots_adjust(hspace=.3)
    
# get DDM contrasts
hddm_contrasts = data.filter(regex='_hddm.*drift')

# get rt contrasts
rt_columns = [c.replace('hddm_drift','rt') for c in hddm_contrasts]
rt_contrasts = pd.DataFrame()
for c in rt_columns:
    rt_contrasts = pd.concat([rt_contrasts, data.filter(regex=c)], axis=1)

# get acc contrasts
acc_columns = [c.replace('hddm_drift','acc') for c in hddm_contrasts]
acc_contrasts = pd.DataFrame()
for c in acc_columns:
    acc_contrasts = pd.concat([acc_contrasts, data.filter(regex=c)], axis=1)

# get EZ contrasts
EZ_columns = [c.replace('hddm','EZ') for c in hddm_contrasts]
EZ_contrasts = pd.DataFrame()
for c in EZ_columns:
    EZ_contrasts = pd.concat([EZ_contrasts, data.filter(regex=c)], axis=1)

# ***********************************************************************
    # Plot DDM differences vs rt and accuracy
# ***********************************************************************

cols = 3
rows = math.ceil(len(tasks)*cols/cols)
f, axes = plt.subplots(rows, cols, figsize=(cols*8, rows*6))
hddm_axes = f.get_axes()[::cols]
rt_axes = f.get_axes()[1::cols]
acc_axes = f.get_axes()[2::cols]

plot_vars(tasks, hddm_contrasts, hddm_axes, xlabel='HDDM Drift Rate')
plot_vars(tasks, rt_contrasts, rt_axes, xlabel='RT (ms)')
plot_vars(tasks, acc_contrasts, acc_axes, xlabel='Accuracy')

# save
save_dir = path.join(base_dir, 'Results', 'replication', 'Plots', 'DDM_rt_effects.%s' % ext)
f.savefig(save_dir, dpi=300, bbox_inches='tight')
plt.close()

# ***********************************************************************
    # Plot drift rate vs difference scores
# ***********************************************************************
cols = 3
rows = math.ceil(len(hddm_contrasts.columns)/cols)
f, axes = plt.subplots(rows, cols, figsize=(cols*8, rows*6))
axes = f.get_axes()
index = 0
for column, contrast in hddm_contrasts.iteritems():
    base_drift_column = column.split('.')[0]+'.hddm_drift'
    base_drift = data.loc[:,base_drift_column]
    sns.regplot(base_drift, contrast, ax=axes[index])
    axes[index].set_xlabel(base_drift_column, fontsize=20)
    axes[index].set_ylabel(column, fontsize=16)
    index+=1
plt.subplots_adjust(hspace=.6)
# save
save_dir = path.join(base_dir, 'Results', 'replication', 'Plots', 'HDDM_difference_vs_drift.%s' % ext)
f.savefig(save_dir, dpi=300, bbox_inches='tight')
plt.close()  
# ***********************************************************************
    # Plot rt vs difference scores
# ***********************************************************************
cols = 3
rows = math.ceil(len(hddm_contrasts.columns)/cols)
f, axes = plt.subplots(rows, cols, figsize=(cols*8, rows*6))
axes = f.get_axes()
index = 0
for column, contrast in rt_contrasts.iteritems():
    base_drift_column = column.split('.')[0]+'.avg_rt'
    base_drift = data.loc[:,base_drift_column]
    sns.regplot(base_drift, contrast, ax=axes[index])
    axes[index].set_xlabel(base_drift_column, fontsize=20)
    axes[index].set_ylabel(column, fontsize=16)
    index+=1
plt.subplots_adjust(hspace=.6)
# save
save_dir = path.join(base_dir, 'Results', 'replication', 'Plots', 'RT_difference_vs_drift.%s' % ext)
f.savefig(save_dir, dpi=300, bbox_inches='tight')
plt.close()   
# ***********************************************************************
    # Plot EZ vs difference scores
# ***********************************************************************
cols = 3
rows = math.ceil(len(hddm_contrasts.columns)/cols)
f, axes = plt.subplots(rows, cols, figsize=(cols*8, rows*6))
axes = f.get_axes()
index = 0
for column, contrast in EZ_contrasts.iteritems():
    base_drift_column = column.split('.')[0]+'.EZ_drift'
    base_drift = data.loc[:,base_drift_column]
    sns.regplot(base_drift, contrast, ax=axes[index])
    axes[index].set_xlabel(base_drift_column, fontsize=20)
    axes[index].set_ylabel(column, fontsize=16)
    index+=1
plt.subplots_adjust(hspace=.6)
# save
save_dir = path.join(base_dir, 'Results', 'replication', 'Plots', 'EZ_difference_vs_drift.%s' % ext)
f.savefig(save_dir, dpi=300, bbox_inches='tight')
plt.close()