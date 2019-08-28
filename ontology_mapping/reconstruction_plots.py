import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA
from dimensional_structure.utils import hierarchical_cluster
from selfregulation.utils.plot_utils import beautify_legend, format_num, save_figure

def plot_factor_reconstructions(reconstructions, title=None, size=12, 
                                plot_regression=True, plot_diagonal=False,
                                filename=None, dpi=300):
    # construct plotting dataframe
    c = reconstructions.columns.get_loc('var') 
    ground_truth = reconstructions.query('label=="true"')
    ground_truth.index = ground_truth['var']
    ground_truth = ground_truth.iloc[:, :c]
    ground_truth.columns = [str(c) + '_GT' for c in ground_truth.columns]
    
    plot_df = reconstructions.join(ground_truth, on='var')
    pop_sizes = sorted(reconstructions.pop_size.dropna().unique())
    # plot
    sns.set_context('talk')
    sns.set_style('white')
    f, axes = plt.subplots(c,len(pop_sizes),figsize=(size,size*1.2))
    colors = sns.color_palette(n_colors = len(pop_sizes))

    for j, pop_size in enumerate(pop_sizes):
        reconstruction = plot_df.query('pop_size == %s' % pop_size)
        for i, factor in enumerate(plot_df.columns[:c]):
            factor = str(factor)
            ax = axes[i][j]
            # plot scatter
            means = reconstruction.groupby('var').mean()
            std = reconstruction.groupby('var').std()
            ax.errorbar(means.loc[:,factor+'_GT'],
                       means.loc[:,factor],
                       yerr=std[factor],
                       color=colors[j],
                       linestyle='',
                       marker='o',
                       markersize=size/2,
                       markeredgecolor='white',
                       markeredgewidth=size/15,
                       linewidth=size/10)
            #ax.set_xlim([-.9,.9])
            #ax.set_ylim([-.9,.9])
            # calculate regression slope
            if plot_regression:
                slope, intercept, r_value, p_value, std_err = \
                     scipy.stats.linregress(x=reconstruction[factor+'_GT'],
                                            y=reconstruction[factor])
                xlims = ax.get_xlim()
                new_x = np.arange(xlims[0], xlims[1],(xlims[1]-xlims[0])/250.)
                y = intercept + slope *  new_x
                ax.plot(new_x, y, color=colors[j], 
                        linestyle='-', lw = size/4, zorder=5)
            # plot diagonal
            if plot_diagonal:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                lims = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
                ax.plot(lims, lims, ls="-", c=".5", linewidth=size/10, zorder=-1)
                ax.set_xlim(lims); ax.set_ylim(lims)
            # labels and ticks
            ax.tick_params(axis='both', labelleft=False, labelbottom=False, bottom=False, left=False)
            if j==(len(pop_sizes)-1) and i==0:
                ax.set_ylabel('Reconstruction', fontsize=size*1.5)
                ax.set_xlabel('Ground Truth', fontsize=size*1.5)
            if j==0:
                ax.set_ylabel(factor, fontsize=size*1.9)
            if i==(c-1):
                ax.set_xlabel(format_num(pop_size), fontsize=size*1.9, color=colors[j])
            # indicate the correlation
            corr = reconstruction.corr().loc[factor, factor+'_GT']
            s = '$\it{r}=%s$' % format_num(corr)
            ax.text(.05,.85, s, transform = ax.transAxes, fontsize=size*1.5)
    f.text(0.5, 0.06, 'Subpopulation Size', ha='center', fontweight='bold', fontsize=size*2)
    f.text(0.04, 0.5, 'Factor', va='center', rotation=90, fontweight='bold', fontsize=size*2)
    if title:
        f.suptitle(title, y=.93, size=size*2, fontweight='bold')
    if filename is not None:
        save_figure(f, filename, {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()

def plot_reconstruction_hist(reconstructions, title=None, size=12, 
                             filename=None, dpi=300):
    sns.set_context('talk')
    sns.set_style('white')
    reconstructions = reconstructions.query('label == "partial_reconstruct"')
    pop_sizes = sorted(reconstructions.pop_size.dropna().unique())
    f, axes = plt.subplots(1,len(pop_sizes)+1,figsize=(size,size/5))
    colors = sns.color_palette(n_colors = len(pop_sizes))
    plt.subplots_adjust(wspace=.3)
    for i, pop_size in enumerate(pop_sizes):
        # plot the mean reconstruction score for each variable
        reconstruction = reconstructions.query('pop_size == %s' % pop_size) \
                                         .groupby('var')['corr_score'].mean()
        reconstruction.hist(bins=20, ax=axes[i], grid=False, color=colors[i])
        # axis labels
        if i == 0:
            axes[i].set_ylabel('# of Variables', fontsize=size*1.5)
        for spine in ['top', 'right']:
            axes[i].spines[spine].set_visible(False)
        axes[i].set_title('N: %s' % int(pop_size), fontsize=size*1.25,
                          color=colors[i])
    # plot graph comparing all
    tmp=reconstructions.groupby(['var','pop_size']).corr_score.mean().reset_index()
    tmp.pop_size = tmp.pop_size.astype('category')
    sns.boxplot(x='corr_score', y='pop_size', data=tmp.reset_index(),
                ax=axes[-1], width=.5, fliersize=size//5, linewidth=size*.1,
                order=pop_sizes[::-1], palette=colors[::-1])
    axes[-1].set_title('Summary', fontsize=size*1.25)
    axes[-1].set_xlabel('')
    axes[-1].set_yticklabels([])
    axes[-1].set_ylabel('')
    for spine in ['top', 'left', 'right']:
        axes[-1].spines[spine].set_visible(False)
    # move boxplot to the right
    pos1 = axes[-1].get_position().bounds
    new_pos = [pos1[0]+.01, pos1[1], pos1[2]*1.5, pos1[3]]
    axes[-1].set_position(new_pos)
    # set xticks
    for ax in axes:
        ax.tick_params(length=1)
        ax.set_xlim(-.2,1)
        ax.set_xticks([0, .5, 1])
        ax.set_xticklabels([0, .5, 1])
    
    f.text(0.5, -0.1, 'Average Reconstruction Score', 
           ha='center',  fontsize=size*1.5)
    if title:
        f.suptitle(title, y=1.15, size=size*1.75)
    if filename is not None:
        save_figure(f, filename, {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
        
def plot_distance_recon(reconstructed_distances, orig_distances, size=10, filename=None, dpi=300):
    # convert distances to flat array
    flattened_distances = {k:scipy.spatial.distance.squareform(v) for k,v in reconstructed_distances.items()}
    flattened_distances = pd.DataFrame(flattened_distances)
    flattened_distances.insert(0, 'original', scipy.spatial.distance.squareform(orig_distances))
    # clustered order
    out = hierarchical_cluster(orig_distances, compute_dist=False)
    orig_clustering = out['clustered_df']
    new_order = orig_clustering.index
    # get dimensions of reconstruction grid
    recon_names = np.unique([i.split('_')[0] for i in reconstructed_distances.keys()])
    ncols = len(recon_names)
    pop_sizes = np.unique([int(i.split('_')[1]) for i in reconstructed_distances.keys()])
    nrows = len(pop_sizes)
    # change column names
    flattened_distances.columns = [' '.join(i.split('_')) for i in flattened_distances.columns]
    flattened_distances.columns = [c.replace(' 0', ' ') for c in flattened_distances.columns]
    # plot
    colors = sns.color_palette(n_colors = len(pop_sizes))
    f = plt.figure(figsize=(size,size))
    # create axes
    gs0 = gridspec.GridSpec(1, 2)
    gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[0])
    orig_ax = plt.subplot(gs00[0,0])
    distdistances_ax = plt.subplot(gs00[1,0])
    # original distance
    sns.heatmap(orig_distances.loc[new_order, new_order],
                ax=orig_ax, vmin=0, vmax=1, square=True,
               xticklabels=False, yticklabels=False, cbar=False)
    orig_ax.set_title('Original Distance Matrix', 
                      size=size*2, y=1.02)
    # plot correlation between reconstructions
    corr = flattened_distances.corr(method='pearson')
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, vmin=0, vmax=1, yticklabels=True, xticklabels=False,
               ax = distdistances_ax, square=True, cbar=False, annot=True,
               mask=mask, annot_kws={'fontsize':size})
    distdistances_ax.set_xlabel('Correlation among reconstructions', 
                               size=size*2, labelpad=size*1.25)
    distdistances_ax.tick_params(length=0, labelsize = size*1.25)

    for i in range(len(corr)):
        distdistances_ax.text((i+.5), (i-.1), corr.columns[i], 
                        ha="center", va="bottom", rotation=90, fontsize=size*1.25)
    gs01 = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=gs0[1])
    # plot the individual reconstructions
    keys = sorted(reconstructed_distances.keys())
    axes = []
    for i, recon_name in enumerate(recon_names):
        for j, pop in enumerate(pop_sizes):
            axes.append(plt.subplot(gs01[j,i]))
            key = recon_name + '_%03d' % pop
            reordered = reconstructed_distances[key].loc[new_order, new_order]
            sns.heatmap(reordered, ax=axes[-1], xticklabels=False, yticklabels=False, 
                        cbar=False, vmin=0, vmax=1, square=True)

            if j==0:
                axes[-1].set_title(recon_name, size=size*2, y=1.05)
            if i==len(recon_names)-1:
                axes[-1].set_ylabel(pop, rotation=-90, labelpad=size*2, size=size*2,
                                   color=colors[j])
                axes[-1].yaxis.set_label_position("right")
    if filename is not None:
        save_figure(f, filename, {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
        
def plot_reconstruction_2D(reconstructions, n_reps=None, 
                           reducer = TSNE(2, metric='precomputed'),
                           n_colored=5, use_background=False,
                           seed=None,title=None, size=12, 
                           filename=None, dpi=300):
    if n_reps is None:
        n_reps = reconstructions.rep.max()
    if seed:
        np.random.seed(seed)
    var_list = reconstructions['var'].unique()
    colored_vars = np.random.choice(var_list, size=n_colored, replace=False)
    print(colored_vars)
    reconstructions = reconstructions[~(reconstructions.rep>n_reps)]
    reconstructions = reconstructions.query('label != "full_reconstruct"')
    if not use_background:
        reconstructions = reconstructions.query('var in %s' % list(colored_vars))

    pop_sizes = sorted(reconstructions.pop_size.dropna().unique())
    c = reconstructions.columns.get_loc('var') 
    # create reduced representation
    reduced = []
    for pop_size in pop_sizes:
        subset = reconstructions.query('label=="true" or pop_size == %s'% pop_size)
        subset = subset.iloc[:, :c]
        distances = squareform(pdist(subset, metric='correlation'))
        reduced.append(reducer.fit_transform(distances))
        
    N_pop = len(pop_sizes)
    # get colors
    colors = sns.color_palette(n_colors = len(pop_sizes))
    tmp_subset = reconstructions.query('label=="true" or pop_size == %s'% pop_sizes[-1]).reset_index(drop=True)
    base_colors = sns.color_palette(palette='hls', n_colors=len(colored_vars))
    color_map = {k:v for k,v in zip(colored_vars, base_colors)}
    colored_indices = tmp_subset[tmp_subset['var'].isin(colored_vars)].index
    color_list = list(tmp_subset.loc[colored_indices,'var'].apply(lambda x: color_map[x]))
    colored_sizes = [300 if x=='true' else 75 for x in tmp_subset.loc[colored_indices,'label']]
    uncolored_indices = list(set(tmp_subset.index) - set(colored_indices))
    # plot scatter
    f,axes = plt.subplots(N_pop,1,figsize=(12,12*N_pop))
    for ax, red, pop_size in zip(axes, reduced, pop_sizes):
        ax.scatter(red[uncolored_indices,0], red[uncolored_indices,1], s=10, c=[.5,.5,.5], alpha=.5)
        ax.scatter(red[colored_indices,0], red[colored_indices,1], s=colored_sizes,
                   c=color_list, edgecolor='black', linewidth=2, alpha=.75)
        ax.set_title('Pseudo-Population Size: %s' % pop_size)

    if title:
        f.suptitle(title, y=1.15, size=size*1.75)
    if filename is not None:
        save_figure(f, filename, {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
    np.random.seed()