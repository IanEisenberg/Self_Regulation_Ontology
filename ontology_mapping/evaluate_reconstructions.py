
# coding: utf-8

# In[ ]:


from glob import glob
import numpy as np
from os import path
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import sklearn
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import statsmodels.formula.api as smf

from dimensional_structure.EFA_plots import get_communality
from dimensional_structure.utils import abs_pdist
from ontology_mapping.reconstruction_plots import (plot_factor_reconstructions,
                                                    plot_reconstruction_hist,
                                                  plot_distance_recon,
                                                  plot_reconstruction_2D)
from ontology_mapping.reconstruction_utils import (combine_files,
                                                   load_files,
                                                  summarize_k,
                                                  summarize_k_partial)
from selfregulation.utils.plot_utils import beautify_legend, format_num, save_figure
from selfregulation.utils.utils import get_info, get_recent_dataset, get_retest_data
from selfregulation.utils.result_utils import load_results


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


dataset = get_recent_dataset()
results_dir = get_info('results_directory')
ontology_results_dir = path.join(results_dir, 'ontology_reconstruction', dataset, '*', 'oblimin')
retest_data = get_retest_data(dataset.replace('Complete', 'Retest'))
plot_dir = glob(path.join(ontology_results_dir, 'Plots'))[0]
save=True


# In[ ]:


results = load_results(dataset)['task']
c = results.EFA.get_c()


# # Load Reconstructions

# In[ ]:


KNNR_files = glob(path.join(ontology_results_dir, 'KNNR_*'))
KNNR_loaded = load_files(KNNR_files)
KNNR_var_summary, KNNR_best_params, KNNR_reconstructions = summarize_k(KNNR_loaded)


# In[ ]:


KNNRind_files = glob(path.join(ontology_results_dir, 'KNNRind_*'))
KNNRind_loaded = load_files(KNNRind_files)
KNNRind_var_summary, KNNRind_best_params, KNNRind_reconstructions = summarize_k(KNNRind_loaded)


# In[ ]:


KNNRpartial_files = glob(path.join(ontology_results_dir, 'KNNRpartial_*'))
KNNRpartial_loaded = load_files(KNNRpartial_files)
KNNRpartial_var_summary = summarize_k_partial(KNNRpartial_loaded)


# In[ ]:


ridge_files = glob(path.join(ontology_results_dir, '*RidgeCV*'))
ridge_loaded = load_files(ridge_files)
linear_files = glob(path.join(ontology_results_dir, '*Linear*'))
linear_loaded = load_files(linear_files)
linear_reconstructions = {'Linear': combine_files(linear_loaded),
                         'RidgeCV': combine_files(ridge_loaded)}


# # Summarize reconstructions

# In[ ]:


KNNR_reconstructions.query('label=="partial_reconstruct"')     .groupby('pop_size')['corr_score'].agg(['mean','std'])


# In[ ]:


KNNRind_reconstructions.query('label=="partial_reconstruct"')     .groupby('pop_size')['corr_score'].agg(['mean','std'])


# In[ ]:


summary = pd.DataFrame()
for clf, df in linear_reconstructions.items():
    tmp = df.query('label=="partial_reconstruct"')         .groupby('pop_size').corr_score.agg([np.mean, np.std])
    tmp.loc[:,'clf'] = clf
    summary = pd.concat([summary, tmp], sort=False)
print(summary)


# # More focuses analyses

# In[ ]:


reconstructions = {'KNNR': KNNR_reconstructions,
                   'RidgeCV': linear_reconstructions['RidgeCV']}
reconstructed_vars = sorted(KNNR_reconstructions['var'].unique())
assert set(reconstructed_vars) == set(reconstructions['RidgeCV']['var'].unique())


# ## How well are we reconstructing distances?

# In[ ]:


orig_loadings = results.EFA.get_loading(c=c).loc[reconstructed_vars]
orig_distances = pd.DataFrame(squareform(abs_pdist(orig_loadings)), index=orig_loadings.index, columns=orig_loadings.index)

reconstructed_distances = {}
for name, reconstruction in reconstructions.items():
    pop_sizes = sorted(reconstruction.pop_size.dropna().unique())
    for pop_size in pop_sizes:
        reconstructed_distances[name+'_%03d' % pop_size] = []
        for rep in range(1, int(reconstruction.rep.max()+1)):
            reconstructed_loadings = reconstruction.query('pop_size == %s and rep==%s' % (pop_size, rep)).sort_values(by='var')
            distances = abs_pdist(reconstructed_loadings.iloc[:,:c])
            reconstructed_distances[name+'_%03d' % pop_size].append(distances)
            
mean_reconstructed_distances = {}
std_reconstructed_distances = {}

for key, distances in reconstructed_distances.items():
    mean_reconstructed_distances[key] =             pd.DataFrame(squareform(np.mean(distances, 0)),
                                    index=orig_loadings.index, 
                                    columns=orig_loadings.index)
    std_reconstructed_distances[key] =             pd.DataFrame(squareform(np.std(distances, 0)),
                                    index=orig_loadings.index, 
                                    columns=orig_loadings.index)


# ## Variable characteristics that influence reconstruction quality

# In[ ]:


# variable characteristics
retest_index = [i.replace('.logTr','').replace('.ReflogTr','') for i in reconstructed_vars]
retest_vals = retest_data.loc[retest_index,'icc3.k']
retest_vals.index = reconstructed_vars
communality = get_communality(results.EFA).loc[retest_index]
communality.index = reconstructed_vars
avg_corr  = abs(results.data.corr()).replace(1,0).mean()
avg_corr.name = "avg_correlation"


# In[ ]:


# create summaries
additional = pd.concat([retest_vals, communality, avg_corr], axis=1, sort=True)
reconstruction_summaries = {}
for name, reconstruction in reconstructions.items():
    s = reconstruction.query('label == "partial_reconstruct"')         .groupby(['var', 'pop_size']).corr_score.agg(['mean', 'std'])
    s = s.reset_index().join(additional, on='var')
    reconstruction_summaries[name] = s
all_reconstructions = pd.concat(reconstruction_summaries).reset_index()
all_reconstructions = all_reconstructions.rename({'level_0': 'approach'}, axis=1).drop('level_1', axis=1)


# Does reconstruction success at one population size predict the next?

# In[ ]:


tmp = []
for i,group in all_reconstructions.groupby(['approach', 'pop_size']):
    group = group.loc[:,['var','mean']].set_index('var')
    group.columns = [i]
    tmp.append(group)
approach_compare = pd.concat(tmp, axis=1)
approach_compare.columns = [i +': '+str(int(j)) for i,j in approach_compare.columns]
# correlation of reconstructions
corr= approach_compare.corr(method='spearman')
overall_correlation = np.mean(corr.values[np.tril_indices_from(corr, -1)])
print('DV reconstruction score correlates %s across approaches' % format_num(overall_correlation))


# Model reconstruction success as a function of DV characteristics, approach and subpopulation size

# In[ ]:


all_reconstructions.loc[:, 'z_mean'] = np.arctanh(all_reconstructions['mean'])
md = smf.mixedlm("z_mean ~ (pop_size + Q('icc3.k') + communality)*C(approach, Sum)", all_reconstructions, groups=all_reconstructions["var"])
mdf = md.fit()
mdf.summary()

# other way to do it
# endog, exog = patsy.dmatrices("z_mean ~ (pop_size + icc + avg_correlation)*C(approach, Sum)", all_reconstructions, return_type='dataframe')
# md = sm.MixedLM(endog=endog, exog=exog, groups=all_reconstructions['var'])


# ## Visualization
# 
# Of concern is the average correspondence and variability between the estimated ontological fingerprint of a DV and its "ground-truth" (the original estimate when it was part of the EFA model)
# 
# One way to look at this is just the average reconstruction score (e.g., for example) and variability of reconstruction score as a function of pseudo-pop-size and model parameters

# In[ ]:


pop_sizes = sorted(reconstructions['KNNR'].pop_size.dropna().unique())
colors = sns.color_palette('Set1', n_colors = len(pop_sizes), desat=.8)


# ### Overall Performance

# In[ ]:


f = plt.figure(figsize=(12,8))
sns.boxplot(x='pop_size', y='mean', hue='approach', data=all_reconstructions, palette='Reds')
plt.legend(loc='best')
if save:
    f.savefig(path.join(plot_dir, 'reconstruction_performance.png'), transparent=True)


# Plot relationship of performance for each DV over different approach parameterizations

# In[ ]:


corr = approach_compare.corr(method='spearman')
mean_success = approach_compare.mean()
plot_df = approach_compare.join(retest_vals).join(communality)
size = 2
f=sns.pairplot(plot_df.iloc[:,0:8], height=size,
             plot_kws={'color': [.4,.4,.4],
                       's': plot_df['communality']*250},
             diag_kws={'bins': 20,
                      'edgecolor': 'k',
                      'linewidth': size/4})
axes = f.axes
# fix axes limits
for i in range(len(f.axes)):
    for j in range(len(f.axes)):
        ax = axes[i][j]
        ax.set_ylim([.15,1.1])
        ax.tick_params(left=False, bottom=False,
                      labelleft=False, labelbottom=False)
        if i!=j:
            ax.set_xlim([.15,1.1])
            ax.plot(ax.get_xlim(), ax.get_ylim(), lw=size, ls="--", c=".3", zorder=-1)
        if j<i:
            x = .6; y = .3
            if mean_success[j] > mean_success[i]:
                x = .28; y = 1
            ax.text(x, y, r'$\rho$ = %s' % format_num(corr.iloc[i,j]),
                   fontsize=size*8)
        # change sizing for upper triangle based on icc
        if j>i: 
            ax.set_visible(False)
            #ax.collections[0].set_sizes(plot_df['icc']**2*100)
            
# color diagonal
for i,ax in enumerate(f.diag_axes):
    ax.set_title(axes[i][0].get_ylabel(), color=colors[i%4], fontsize=size*9)
    for patch in ax.patches:
        patch.set_facecolor(colors[i%4])
        
# color labels
for i in range(len(f.axes)):
    left_ax = axes[i][0]
    bottom_ax = axes[-1][i]
    left_ax.set_ylabel(left_ax.get_ylabel(), color=colors[i%4],labelpad=10, fontsize=size*9)
    bottom_ax.set_xlabel('')
    
# set tick spacing
ax = axes[-1][-2]
ax.tick_params(length=1, width=1, labelleft=True, labelbottom=True)
ax.set_xticks([.18, 1])
ax.set_xticklabels(['0.2', '1.0'], fontsize=size*8, fontweight='bold')
ax.set_yticks([1])
ax.set_yticklabels(['1.0'], fontsize=size*8, fontweight='bold')
# common X
f.fig.text(0.5, 0.02, 'Average DV Reconstruction Score', ha='center', fontsize=size*10)
if save:
    save_figure(f, path.join(plot_dir, 'SFig1_cross_approach_correlations.png'), save_kws={'dpi': 300})


# ### K Nearest Visualization (Example)

# #### Average Performance by Model Parameters

# In[ ]:


desaturated_colors = [sns.desaturate(c, .5) for c in colors]
plot_colors = list(zip(colors, desaturated_colors))

plot_df = KNNR_var_summary.reset_index()
sns.set_context('talk')
f, ax = plt.subplots(1, 1, figsize=(12,6))
axes = f.get_axes()
for i, pop_size in enumerate(pop_sizes):
    sns.pointplot(x='k', y='corr_score', hue='weighting', 
                data=plot_df.query('pop_size==%s' % pop_size),
                ax=ax, dodge=.35, alpha=1, join=False, ci=None,
                palette = plot_colors[i], label=pop_size)
ax.legend().set_visible(False)
ax.set_xticklabels([int(i) for i in plot_df.k.unique()])
ax.set_ylim(.25,1.1)
ax.set_ylabel('Reconstruction Score')
plt.subplots_adjust(hspace=.4)


# #### Performance for each DV
# 
# Only taking the best parameters from the k-nearest neighbor algorithm

# In[ ]:


var = "simon.hddm_drift"
ax = reconstructions['KNNR'].query('var == "%s" and pop_size==100' % var).corr_score.hist(bins=30,
                                                                          edgecolor='white',
                                                                           figsize=[10,6])
ax.set_xlabel('Reconstruction Score', fontsize=40, labelpad=30)
ax.set_yticklabels([])
ax.set_yticks([])
ax.xaxis.set_major_locator(ticker.MultipleLocator(.05))
ax.tick_params(labelsize=30)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(False)


# ##### Histogram of DV reconstruction scores

# In[ ]:


plot_reconstruction_hist(reconstructions['KNNR'], title='KNNR Reconstruction', size=14)
plot_reconstruction_hist(reconstructions['RidgeCV'], title='RidgeCV Reconstruction', size=14)


# In[ ]:


# saving
if save:
    plot_reconstruction_hist(reconstructions['KNNR'], title='KNNR Reconstruction', size=14,
                            filename=path.join(plot_dir, 'Fig3a_KNNR_reconstruction.png'))
    plot_reconstruction_hist(reconstructions['RidgeCV'], title='RidgeCV Reconstruction', size=14,
                            filename=path.join(plot_dir, 'Fig3b_RidgeCV_reconstruction.png'))


# There is clearly a bit of variability in the reconstruction accuracy based on the variable itself. While this variability narrows with larger populations, it's still there, and there are a few variables that cannot be reconstructed at all
# 
# We have access to some characteristics of these DVs (reliability, communality, avg correlation), which we can look at

# ##### Reconstruction score vs. DV characteristics

# In[ ]:


sns.set_context('talk')
sns.set_style('white')
ind_vars = ['icc3.k', 'communality'] # 'avg_correlation' could be included
N = len(ind_vars)*len(reconstruction_summaries.keys())
size=6
f, axes = plt.subplots(2,N,figsize=(size*N, size*2))
for i, (name, reconstruction) in enumerate(reconstruction_summaries.items()):
    for j, var in enumerate(ind_vars):
        col_i = len(ind_vars)*i+j
        for k, pop_size in enumerate(pop_sizes):
            sns.regplot(var, 'mean', 
                        data=reconstruction.query('pop_size==%s' % pop_size), 
                        label=pop_size, ax=axes[0][col_i], color=colors[k])
            sns.regplot(var, 'std', 
                        data=reconstruction.query('pop_size==%s' % pop_size), 
                        label=pop_size, ax=axes[1][col_i], color=colors[k])
        # mean plots
        axes[0][col_i].tick_params(bottom=False, labelbottom=False)
        axes[0][col_i].set_xlabel('')
        axes[0][col_i].set_ylabel('')
        axes[0][col_i].set_ylim(-.2, 1.1)
        # sd plots
        axes[1][col_i].set_xlabel(var.title(), fontweight='bold', fontsize=size*4)
        axes[1][col_i].set_ylabel('')
        axes[1][col_i].set_ylim(-.1, .6)
        if col_i==0:
            axes[0][col_i].set_ylabel(r'$\mu$', fontweight='bold', fontsize=size*5)
            axes[1][col_i].set_ylabel(r'$\sigma$', fontweight='bold', fontsize=size*5)
        else:
            axes[0][col_i].tick_params(left=False, labelleft=False)
            axes[1][col_i].tick_params(left=False, labelleft=False)
    f.text(0.31+.4*i, .93, name.title(), ha='center', fontsize=size*5)

axes[0][-1].legend(title='N')
plt.subplots_adjust(wspace=.1, hspace=.1)


# Simple plot for paper - just looking at mean for communality

# In[ ]:


sns.set_context('talk')
sns.set_style('white')
ind_vars = ['communality'] # 'avg_correlation' could be included
N = len(ind_vars)*len(reconstruction_summaries.keys())
size=6
f, axes = plt.subplots(2,N,figsize=(size*N, size*2))
for i, (name, reconstruction) in enumerate(reconstruction_summaries.items()):
    for j, var in enumerate(ind_vars):
        col_i = len(ind_vars)*i+j
        for k, pop_size in enumerate(pop_sizes):
            sns.regplot(var, 'mean', ci=None,
                        data=reconstruction.query('pop_size==%s' % pop_size), 
                        label=pop_size, ax=axes[0][col_i], color=colors[k])
            sns.regplot(var, 'std', ci=None,
                        data=reconstruction.query('pop_size==%s' % pop_size), 
                        label=pop_size, ax=axes[1][col_i], color=colors[k])
        # mean plots
        axes[0][col_i].tick_params(bottom=False, labelbottom=False, left=True,
                                  length=size/2, width=size/2)
        axes[0][col_i].set_xlabel('')
        axes[0][col_i].set_ylabel('')
        axes[0][col_i].set_ylim(-.2, 1.1)
        # sd plots
        axes[1][col_i].tick_params(length=size/2, width=size/2, left=True, bottom=True)
        axes[1][col_i].set_xlabel(var.title(), fontweight='bold', fontsize=size*4)
        axes[1][col_i].set_ylabel('')
        axes[1][col_i].set_ylim(-.05, .6)
        if col_i==0:
            axes[0][col_i].set_ylabel(r'$\mu$', fontweight='bold', fontsize=size*5)
            axes[1][col_i].set_ylabel(r'$\sigma$', fontweight='bold', fontsize=size*5)
        else:
            axes[0][col_i].tick_params(left=False, labelleft=False)
            axes[1][col_i].tick_params(left=False, labelleft=False)
    f.text(0.31+.4*i, .9, name, ha='center', fontsize=size*5)

axes[0][-1].legend(title='N', fontsize=size*3)
plt.subplots_adjust(wspace=.1, hspace=.1)


if save:
    save_figure(f, path.join(plot_dir, 'Fig5_DV_characteristics.png'), save_kws={'dpi': 300})


# It seems clear that DVs with poor reliability and communality are not reconstructed well. A less "analysis based" way to think about this is reconstruction will be worse if you are far away from the other variables in the set.
# 
# Similarly, correlation with the overall dataset is important for reconstruction. All of this says that ontological mapping will be more successful if you have an a-priori reason to believe your new variable has something to do with the rest of the variables in the ontology. The weaker you believe that bond, the more data you should collect to articulate the connection

# We can dive in and look at one high/mediun/low reliable variable to see the reconstruction performance

# In[ ]:


"""
sorted_retest_vals = retest_vals.sort_values().index
N = len(sorted_retest_vals)
high_var = sorted_retest_vals[N-1]
med_var = sorted_retest_vals[N//2]
low_var = sorted_retest_vals[0]

f, axes = plt.subplots(1,3, figsize=(20,8))
for ax, var in zip(axes, [high_var, med_var, low_var]):
    retest_in = var.replace('.logTr','').replace('.ReflogTr','')
    reliability = format_num(retest_data.loc[retest_in]['icc'])
    plot_df = k_reconstruction.query('var == "%s" and label=="partial_reconstruct"' % var)
    sns.boxplot(x='pop_size', y='corr_score', data=plot_df,  ax=ax)
    ax.set_title('%s\nICC: %s' % (var, reliability))
    ax.set_ylim([-.2,1.1])
plt.subplots_adjust(wspace=.6)
"""


# #### Visualization of reconstructed distances

# In[ ]:


plot_distance_recon(mean_reconstructed_distances, orig_distances, size=12)


# In[ ]:


# save
if save:
    plot_distance_recon(mean_reconstructed_distances, orig_distances, size=15, 
                       filename=path.join(plot_dir, 'Fig8_distance_reconstructions.png'))


# #### Visualization of Variability

# ##### Visualizing each factor's reconstruction separately

# In[ ]:


plot_factor_reconstructions(reconstructions['KNNR'], size=15, plot_diagonal=True, plot_regression=False)
plot_factor_reconstructions(reconstructions['RidgeCV'], size=15, plot_diagonal=True, plot_regression=False)


# In[ ]:


# save
if save:
    plot_factor_reconstructions(reconstructions['KNNR'], size=10, plot_diagonal=True, plot_regression=False,
                                filename=path.join(plot_dir, 'Fig6_KNN_factor_reconstructions.png'))
    plot_factor_reconstructions(reconstructions['RidgeCV'], size=10, plot_diagonal=True, plot_regression=False,
                                filename=path.join(plot_dir, 'Fig7_RidgeCV_factor_reconstructions.png'))


# ##### Using TSNE

# More complicate, we can visualize this by looking at the MDS plotting:
# 1. The original DVs
# 2. The "best" reconstruction using all the data
# 3. The n_reps simulated estimates with a smaller population size

# In[ ]:


plot_reconstruction_2D(reconstructions['KNNR'], n_reps=30, n_colored=6, use_background=True, seed=100)


# ## Visualization of Reduced Reconstruction

# In[ ]:


sns.set_context('notebook')
sns.set_style('white')
size=12
f, axes = plt.subplots(1,2, figsize=(size*2, size*.75))

# random subset
KNNRpartial_var_summary.pop_size = KNNRpartial_var_summary.pop_size.astype(int)
KNNRpartial_var_summary.num_available_measures = KNNRpartial_var_summary.num_available_measures.astype(int)

sns.pointplot(x='num_available_measures', y='corr_score', hue='pop_size', data=KNNRpartial_var_summary, 
             palette=colors, ax=axes[0], ci=None, scale=1.4)
leg = axes[0].legend(loc='best', frameon=False, handlelength=0, handletextpad=0,
                     fontsize=size*1.5)
beautify_legend(leg, colors=colors)
leg.get_title().set_fontsize(size*1.5)

axes[0].set_ylabel('Reconstruction Score', fontsize=size*3)
axes[0].set_xlabel('# of Measures', fontsize=size*2)
axes[0].set_title('KNNR with Random Subset', fontsize=size*3)
axes[0].tick_params(width=2, length=2, labelsize=size*1.8)

# efficiency subset
closest_files = glob(path.join(ontology_results_dir, 'KNNRclosest_correlation_summary.pkl'))
closest_summary = pd.read_pickle(closest_files[0])
sns.pointplot(x='num_available_measures', y='mean', hue='pop_size', data=plot_df, 
             palette=colors, ci=None, ax=axes[1], scale=1.2)
axes[1].get_legend().remove()
axes[1].set_ylabel('', fontsize=size*2)
axes[1].set_xlabel('# of Measures', fontsize=size*2)
axes[1].set_title('KNNR with Efficient Subset', fontsize=size*3)
axes[1].tick_params(width=2, length=2, labelsize=size*1.8)

# set axes to he the same
ylim = (np.min([ax.get_ylim()[0] for ax in axes]), np.max([ax.get_ylim()[1] for ax in axes]))
axes[0].set_ylim(ylim)
axes[1].set_ylim(ylim)
if save:
    save_figure(f, path.join(plot_dir, 'Fig4_partial_reconstructions.png'), save_kws={'dpi': 300})

