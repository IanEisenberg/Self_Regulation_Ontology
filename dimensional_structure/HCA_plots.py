# imports
from itertools import combinations
from math import ceil
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from os import makedirs, path
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler, scale

from dimensional_structure.utils import (abs_pdist, get_constant_height_labels, 
                                         set_seed, silhouette_analysis)
from dimensional_structure.plot_utils import plot_loadings, plot_tree
from selfregulation.utils.plot_utils import (dendroheatmap, format_num,
                                             format_variable_names, 
                                             get_dendrogram_color_fun,
                                             save_figure)

def plot_clusterings(results, plot_dir=None, inp='data', figsize=(50,50),
                     titles=None, show_clusters=True, verbose=False, ext='png'):    
    HCA = results.HCA
    # get all clustering solutions
    clustering = HCA.results[inp]
    name = inp
    
    # plot dendrogram heatmaps
    if titles is None:
        title = name + '_metric-' + HCA.dist_metric
    else:
        title=titles.pop(0)
    filename = None
    if plot_dir is not None:
        filename = path.join(plot_dir, 'dendroheatmap_%s.%s' % (name, ext))
    if show_clusters == True:
        fig = dendroheatmap(clustering['linkage'], clustering['distance_df'], 
                            clustering['labels'],
                            figsize=figsize, title=title,
                            filename = filename)
    else:
        fig = dendroheatmap(clustering['linkage'], clustering['distance_df'], 
                            figsize=figsize, title=title,
                            filename = filename)
    return fig
            
def plot_clustering_similarity(results, plot_dir=None, verbose=False, ext='png'):  
    HCA = results.HCA
    # get all clustering solutions
    clusterings = HCA.results.items()
    # plot cluster agreement across embedding spaces
    names = [k for k,v in clusterings]
    cluster_similarity = np.zeros((len(clusterings), len(clusterings)))
    cluster_similarity = pd.DataFrame(cluster_similarity, 
                                     index=names,
                                     columns=names)
    
    distance_similarity = np.zeros((len(clusterings), len(clusterings)))
    distance_similarity = pd.DataFrame(distance_similarity, 
                                     index=names,
                                     columns=names)
    for clustering1, clustering2 in combinations(clusterings, 2):
        name1 = clustering1[0].split('-')[-1]
        name2 = clustering2[0].split('-')[-1]
        # record similarity of distance_df
        dist_corr = np.corrcoef(squareform(clustering1[1]['distance_df']),
                                squareform(clustering2[1]['distance_df']))[1,0]
        distance_similarity.loc[name1, name2] = dist_corr
        distance_similarity.loc[name2, name1] = dist_corr
        # record similarity of clustering of dendrogram
        clusters1 = clustering1[1]['labels']
        clusters2 = clustering2[1]['labels']
        rand_score = adjusted_rand_score(clusters1, clusters2)
        MI_score = adjusted_mutual_info_score(clusters1, clusters2)
        cluster_similarity.loc[name1, name2] = rand_score
        cluster_similarity.loc[name2, name1] = MI_score
    
    with sns.plotting_context(context='notebook', font_scale=1.4):
        clust_fig = plt.figure(figsize = (12,12))
        sns.heatmap(cluster_similarity, square=True)
        plt.title('Cluster Similarity: TRIL: Adjusted MI, TRIU: Adjusted Rand',
                  y=1.02)
        
        dist_fig = plt.figure(figsize = (12,12))
        sns.heatmap(distance_similarity, square=True)
        plt.title('Distance Similarity, metric: %s' % HCA.dist_metric,
                  y=1.02)
        
    if plot_dir is not None:
        save_figure(clust_fig, path.join(plot_dir, 
                                   'cluster_similarity_across_measures.%s' % ext),
                    {'bbox_inches': 'tight'})
        save_figure(dist_fig, path.join(plot_dir, 
                                   'distance_similarity_across_measures.%s' % ext),
                    {'bbox_inches': 'tight'})
        plt.close(clust_fig)
        plt.close(dist_fig)
    
    if verbose:
        # assess relationship between two measurements
        rand_scores = cluster_similarity.values[np.triu_indices_from(cluster_similarity, k=1)]
        MI_scores = cluster_similarity.T.values[np.triu_indices_from(cluster_similarity, k=1)]
        score_consistency = np.corrcoef(rand_scores, MI_scores)[0,1]
        print('Correlation between measures of cluster consistency: %.2f' \
              % score_consistency)
        
    
def plot_subbranch(target_color, cluster_i, tree, loading, cluster_sizes, title=None,
                   size=2.3, dpi=300, plot_loc=None):
    sns.set_style('white')
    colormap = sns.diverging_palette(220,15,n=100,as_cmap=True)
    # get variables in subbranch based on coloring
    curr_color = tree['color_list'][0]
    start = 0
    for i, color in enumerate(tree['color_list']):
        if color != curr_color:
            end = i
            if curr_color == to_hex(target_color):
                break
            if color != "#808080":
                start = i
            curr_color = color
    
    if (end-start)+1 != cluster_sizes[cluster_i]:
        return
    
    # get subset of loading
    cumsizes = np.cumsum(cluster_sizes)
    if cluster_i==0:
        loading_start = 0
    else:
        loading_start = cumsizes[cluster_i-1]
    subset_loading = loading.T.iloc[:,loading_start:cumsizes[cluster_i]]
    
    # plotting
    N = subset_loading.shape[1]
    length = N*.05
    dendro_size = [0,.746,length,.12]
    heatmap_size = [0,.5,length,.25]
    fig = plt.figure(figsize=(size,size*2))
    dendro_ax = fig.add_axes(dendro_size) 
    heatmap_ax = fig.add_axes(heatmap_size)
    cbar_size = [length+.22, .5, .05, .25]
    factor_avg_size = [length+.01,.5,.2,.25]
    factor_avg_ax = fig.add_axes(factor_avg_size)
    cbar_ax = fig.add_axes(cbar_size)
    #subset_loading.columns = [col.replace(': ',':\n', 1) for col in subset_loading.columns]
    plot_tree(tree, range(start, end), dendro_ax, linewidth=size/2)
    dendro_ax.set_xticklabels('')
    
    max_val = np.max(loading.values)
    # if max_val is high, just make it 1
    if max_val > .9:
        max_val = 1
    sns.heatmap(subset_loading, ax=heatmap_ax, 
                cbar=True,
                cbar_ax=cbar_ax,
                cbar_kws={'ticks': [-max_val, 0, max_val]},
                yticklabels=True,
                vmin=-max_val,
                vmax=max_val,
                cmap=colormap,)
    yn, xn = subset_loading.shape
    tick_label_size = size*30/max(yn, 8)
    heatmap_ax.tick_params(labelsize=tick_label_size, length=size*.5, 
                           width=size/5, pad=size)
    heatmap_ax.set_yticklabels(heatmap_ax.get_yticklabels(), rotation=0)
    heatmap_ax.set_xticks([i+.5 for i in range(0,subset_loading.shape[1])])
    heatmap_ax.set_xticklabels([str(i) for i in range(1,subset_loading.shape[1]+1)], 
                                size=size*2, rotation=0, ha='center')

    avg_factors = abs(subset_loading).mean(1)
    # format cbar axis
    cbar_ax.set_yticklabels([format_num(-max_val), 0, format_num(max_val)])
    cbar_ax.tick_params(axis='y', length=0)
    cbar_ax.tick_params(labelsize=size*3)
    cbar_ax.set_ylabel('Factor Loading', rotation=-90, fontsize=size*3,
                       labelpad=size*2)
    # add axis labels as text above
    text_ax = fig.add_axes([-.22,.44-.02*N,.4,.02*N]) 
    for spine in ['top','right','bottom','left']:
        text_ax.spines[spine].set_visible(False)
    for i, label in enumerate(subset_loading.columns):
        text_ax.text(0, 1-i/N, str(i+1)+'.', fontsize=size*2.8, ha='right')
        text_ax.text(.1, 1-i/N, label, fontsize=size*3)
    text_ax.tick_params(which='both', labelbottom=False, labelleft=False,
                        bottom=False, left=False)
    # average factor bar                
    avg_factors[::-1].plot(kind='barh', ax = factor_avg_ax, width=.7,
                     color= tree['color_list'][start])
    factor_avg_ax.set_xlim(0, max_val)
    #factor_avg_ax.set_xticks([max(avg_factors)])
    #factor_avg_ax.set_xticklabels([format_num(max(avg_factors))])
    factor_avg_ax.set_xticklabels('')
    factor_avg_ax.set_yticklabels('')
    factor_avg_ax.tick_params(length=0)
    factor_avg_ax.spines['top'].set_visible(False)
    factor_avg_ax.spines['bottom'].set_visible(False)
    factor_avg_ax.spines['left'].set_visible(False)
    factor_avg_ax.spines['right'].set_visible(False)
        
    # title and axes styling of dendrogram
    if title:
        dendro_ax.set_title(title, fontsize=size*3, y=1.05, fontweight='bold')
    dendro_ax.get_yaxis().set_visible(False)
    dendro_ax.spines['top'].set_visible(False)
    dendro_ax.spines['right'].set_visible(False)
    dendro_ax.spines['bottom'].set_visible(False)
    dendro_ax.spines['left'].set_visible(False)
    if plot_loc is not None:
        save_figure(fig, plot_loc, {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
    else:
        return fig

def plot_subbranches(results, rotate='oblimin', EFA_clustering=True,
                     cluster_range=None, absolute_loading=False,
                     size=2.3, dpi=300, ext='png', plot_dir=None):
    """ Plots HCA results as dendrogram with loadings underneath
    
    Args:
        results: results object
        c: number of components to use for loadings
        orientation: horizontal or vertical, which determines the direction
            the dendrogram leaves should be spread out on
        plot_dir: if set, where to save the plot
        inp: which clustering solution to use
        titles: list of titles. Should correspond to number of clusters in
                results object if "inp" is not set. Otherwise should be a list of length 1.
    """
    HCA = results.HCA
    EFA = results.EFA
    loading = EFA.reorder_factors(EFA.get_loading(rotate=rotate), rotate=rotate)
    loading.index = format_variable_names(loading.index)
    if EFA_clustering:
        inp = 'EFA%s_%s' % (EFA.get_c(), rotate)
    else:
        inp = 'data'
    clustering = HCA.results[inp]
    name = inp
    
    # extract cluster vars
    link = clustering['linkage']
    labels = clustering['clustered_df'].columns
    labels = format_variable_names(labels)
    ordered_loading = loading.loc[labels]
    if absolute_loading:
        ordered_loading = abs(ordered_loading)
    # get cluster sizes
    cluster_labels, DVs= list(zip(*HCA.get_cluster_DVs(inp=name).items()))
    cluster_sizes = [len(i) for i in DVs]
    link_function, colors = get_dendrogram_color_fun(link, clustering['reorder_vec'],
                                                     clustering['labels'])
    tree = dendrogram(link,  link_color_func=link_function, no_plot=True,
                      no_labels=True)
    
    if plot_dir is not None:
        function_directory = 'subbranches_input-%s' % inp
        makedirs(path.join(plot_dir, function_directory), exist_ok=True)
        
    plot_loc = None
    if cluster_range is None:
        cluster_range = range(len(cluster_labels))
    # titles = 
    figs = []
    for cluster_i in cluster_range:
        if plot_dir:
            filey = 'cluster_%s.%s' % (str(cluster_i).zfill(2), ext)
            plot_loc = path.join(plot_dir, function_directory, filey)
        fig = plot_subbranch(colors[cluster_i], cluster_i, tree, 
                             ordered_loading, cluster_sizes,
                             title=cluster_labels[cluster_i], 
                             size=size, plot_loc=plot_loc)
        if fig:
            figs.append(fig)
    return figs
                
def plot_results_dendrogram(results, rotate='oblimin', hierarchical_EFA=False,
                            EFA_clustering=True, title=None, size=4.6,  
                            ext='png', plot_dir=None,
                            **kwargs):
    subset = results.ID.split('_')[0]
    HCA = results.HCA
    EFA = results.EFA     
    c = EFA.get_c()
    if EFA_clustering:
        inp = 'EFA%s_%s' % (c, rotate)
    else:
        inp = 'data'
    if hierarchical_EFA or not EFA_clustering:
        loading = EFA.reorder_factors(EFA.get_loading(c, rotate=rotate))
    else:
        loading = EFA.get_loading(c, rotate=rotate)
        cluster_names = HCA.get_cluster_names(inp=inp)
        cluster_loadings = HCA.get_cluster_loading(EFA, rotate=rotate)
        loading_order = []
        i = 0
        while len(loading_order) < c and i<len(cluster_names):
            cluster_loading = cluster_loadings[cluster_names[i]]
            top_factor = cluster_loading.idxmax()
            if top_factor not in loading_order:
                loading_order.append(top_factor)
            i+=1
        loading_order += list(set(loading.columns)-set(loading_order))
        loading = loading.loc[:, loading_order]
            
    clustering = HCA.results[inp]
    name = inp
    if title is None:
        title = subset.title() + " Dependent Variable Structure"
    if plot_dir:
        filename =  path.join(plot_dir, 'dendrogram_%s.%s' % (name, ext))
    else:
        filename = None

    return plot_dendrogram(loading, clustering, 
                    title=title, 
                    size=size, 
                    filename=filename,
                    **kwargs)
    
def transform_name(name):
    """ helper function to transform names for plotting dendrogram """
    if name == 'Decisiveness':
        name= 'Decis.'
    elif name == 'Response Inhibition':
        name= 'Resp. Inhib.'
    elif name[:2] == "NA":
        name = ""
    return '\n'.join(name.split())
    
def plot_dendrogram(loading, clustering, title=None, 
                    break_lines=True, drop_list=None, double_drop_list=None,
                    absolute_loading=False,  size=4.6,  dpi=300, 
                    filename=None):
    """ Plots HCA results as dendrogram with loadings underneath
    
    Args:
        loading: pandas df, a results EFA loading matrix
        clustering: pandas df, a results HCA clustering
        title (optional): str, title to plot
        break_lines: whether to separate EFA heatmap based on clusters, default=True
        drop_list (optional): list of cluster indices to drop the cluster label
        drop_list (optional): list of cluster indices to drop the cluster label twice
        absolute_loading: whether to plot the absolute loading value, default False
        plot_dir: if set, where to save the plot
        
    """


    c = loading.shape[1]
    # extract cluster vars
    link = clustering['linkage']
    DVs = clustering['clustered_df'].columns
    ordered_loading = loading.loc[DVs]
    if absolute_loading:
        ordered_loading = abs(ordered_loading)
    # get cluster sizes
    labels=clustering['labels']
    cluster_sizes = [np.sum(labels==(i+1)) for i in range(max(labels))]
    link_function, colors = get_dendrogram_color_fun(link, clustering['reorder_vec'],
                                                     labels)
    
    # set figure properties
    figsize = (size, size*.6)
    # set up axes' size 
    heatmap_height = ordered_loading.shape[1]*.035
    heat_size = [.1, heatmap_height]
    dendro_size=[np.sum(heat_size), .3]
    # set up plot axes
    dendro_size = [.15,dendro_size[0], .78, dendro_size[1]]
    heatmap_size = [.15,heat_size[0],.78,heat_size[1]]
    cbar_size = [.935,heat_size[0],.015,heat_size[1]]
    ordered_loading = ordered_loading.T

    with sns.axes_style('white'):
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_axes(dendro_size) 
        # **********************************
        # plot dendrogram
        # **********************************
        with plt.rc_context({'lines.linewidth': size*.125}):
            dendrogram(link, ax=ax1, link_color_func=link_function,
                       orientation='top')
        # change axis properties
        ax1.tick_params(axis='x', which='major', labelsize=14,
                        labelbottom=False)
        ax1.get_yaxis().set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        # **********************************
        # plot loadings as heatmap below
         # **********************************
        ax2 = fig.add_axes(heatmap_size)
        cbar_ax = fig.add_axes(cbar_size)
        max_val = np.max(abs(loading.values))
        # bring to closest .25
        max_val = ceil(max_val*4)/4
        sns.heatmap(ordered_loading, ax=ax2, 
                    cbar=True, cbar_ax=cbar_ax,
                    yticklabels=True,
                    xticklabels=True,
                    vmax =  max_val, vmin = -max_val,
                    cbar_kws={'orientation': 'vertical',
                              'ticks': [-max_val, 0, max_val]},
                    cmap=sns.diverging_palette(220,15,n=100,as_cmap=True))
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
        ax2.tick_params(axis='y', labelsize=size*heat_size[1]*30/c, pad=size/4, length=0)            
        # format cbar axis
        cbar_ax.set_yticklabels([format_num(-max_val), 0, format_num(max_val)])
        cbar_ax.tick_params(labelsize=size*heat_size[1]*25/c, length=0, pad=size/2)
        cbar_ax.set_ylabel('Factor Loading', rotation=-90, 
                       fontsize=size*heat_size[1]*30/c, labelpad=size*2)
        # add lines to heatmap to distinguish clusters
        if break_lines == True:
            xlim = ax2.get_xlim(); 
            ylim = ax2.get_ylim()
            step = xlim[1]/len(labels)
            cluster_breaks = [i*step for i in np.cumsum(cluster_sizes)]
            ax2.vlines(cluster_breaks[:-1], ylim[0], ylim[1], linestyles='dashed',
                       linewidth=size*.1, colors=[.5,.5,.5], zorder=10)
        # **********************************
        # plot cluster names
        # **********************************
        beginnings = np.hstack([[0],np.cumsum(cluster_sizes)[:-1]])
        centers = beginnings+np.array(cluster_sizes)//2+.5
        offset = .07
        if 'cluster_names' in clustering.keys():
            ax2.tick_params(axis='x', reset=True, top=False, bottom=False, width=size/8, length=0)
            names = [transform_name(i) for i in clustering['cluster_names']]
            ax2.set_xticks(centers)
            ax2.set_xticklabels(names, rotation=0, ha='center', 
                                fontsize=heatmap_size[2]*size*1)
            ticks = ax2.xaxis.get_ticklines()[::2]
            for i, label in enumerate(ax2.get_xticklabels()):
                if label.get_text() != '':
                    ax2.hlines(c+offset,beginnings[i]+.5,beginnings[i]+cluster_sizes[i]-.5, 
                               clip_on=False, color=colors[i], linewidth=size/5)
                    label.set_color(colors[i])
                    ticks[i].set_color(colors[i])
                    y_drop = .005
                    line_drop = .3
                    if drop_list and i in drop_list:
                        y_drop = .05
                        line_drop = 1.6
                    if double_drop_list and i in double_drop_list:
                        y_drop = .1
                        line_drop = 2.9
                    label.set_y(-(y_drop/heatmap_height+heatmap_height/c*offset))
                    ax2.vlines(beginnings[i]+cluster_sizes[i]/2, 
                               c+offset, c+offset+line_drop,
                               clip_on=False, color=colors[i], 
                               linewidth=size/7.5)

        # add title
        if title:
            ax1.set_title(title, fontsize=size*2, y=1.05)
            
    if filename is not None:
        save_figure(fig, filename,
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
    else:
        return fig
    
def plot_graphs(HCA_graphs, plot_dir=None, ext='png'):
    if plot_dir is not None:
        makedirs(path.join(plot_dir, 'graphs'))
    plot_options = {'inline': False,  'target': None}
    for i, GA in enumerate(HCA_graphs):
        if plot_dir is not None:
            plot_options['target'] = path.join(plot_dir, 
                                                'graphs', 
                                                'graph%s.%s' % (i, ext))
        GA.set_visual_style()
        GA.display(plot_options)
    
    
    
@set_seed(seed=15)
def MDS_visualization(results, c, rotate='oblimin', plot_dir=None, 
                      dist_metric='abs_correlation', ext='png', **plot_kws):
    """ visualize EFA loadings and compares to raw space """
    def scale_plot(input_data, data_colors=None, cluster_colors=None,
                   cluster_sizes=None, dissimilarity='euclidean', filey=None):
        """ Plot MDS of data and clusters """
        if data_colors is None:
            data_colors = 'r'
        if cluster_colors is None:
            cluster_colors='b'
        if cluster_sizes is None:
            cluster_sizes = 2200
            
        # scale
        mds = MDS(dissimilarity=dissimilarity)
        mds_out = mds.fit_transform(input_data)
        
        with sns.axes_style('white'):
            f=plt.figure(figsize=(14,14))
            plt.scatter(mds_out[n_clusters:,0], mds_out[n_clusters:,1], 
                        s=75, color=data_colors)
            plt.scatter(mds_out[:n_clusters,0], mds_out[:n_clusters,1], 
                        marker='*', s=cluster_sizes, color=cluster_colors,
                        edgecolor='black', linewidth=2)
            # plot cluster number
            offset = .011
            font_dict = {'fontsize': 17, 'color':'white'}
            for i,(x,y) in enumerate(mds_out[:n_clusters]):
                if i<9:
                    plt.text(x-offset,y-offset,i+1, font_dict)
                else:
                    plt.text(x-offset*2,y-offset,i+1, font_dict)
        if filey is not None:
            plt.title(path.basename(filey)[:-4], fontsize=20)
            save_figure(f, filey)
            plt.close()
            
    # set up variables
    data = results.data
    HCA = results.HCA
    EFA = results.EFA
    
    names, cluster_loadings = zip(*HCA.get_cluster_loading(EFA, rotate=rotate).items())
    cluster_DVs = HCA.get_cluster_DVs(inp='EFA%s_%s' % (EFA.get_c(), rotate))
    cluster_DVs = [cluster_DVs[n] for n in names]
    cluster_loadings_mat = np.vstack(cluster_loadings)
    EFA_loading = abs(EFA.get_loading(c, rotate=rotate))
    EFA_loading_mat = EFA_loading.values
    EFA_space = np.vstack([cluster_loadings_mat, EFA_loading_mat])
    
    # set up colors
    n_clusters = cluster_loadings_mat.shape[0]
    color_palette = sns.color_palette(palette='hls', n_colors=n_clusters)
    colors = []
    for var in EFA_loading.index:
        # find which cluster this variable is in
        index = [i for i,cluster in enumerate(cluster_DVs) \
                 if var in cluster][0]
        colors.append(color_palette[index])
    # set up cluster sizes proportional to number of members
    n_members = np.reshape([len(i) for i in cluster_DVs], [-1,1])
    scaler = MinMaxScaler()
    relative_members = scaler.fit_transform(n_members).flatten()
    sizes = 1500+2000*relative_members
    
    if dist_metric == 'abs_correlation':
        EFA_space_distances = squareform(abs_pdist(EFA_space))
    else: 
        EFA_space_distances = squareform(pdist(EFA_space, dist_metric))
    
    # repeat the same thing as above but with raw distances
    scaled_data = pd.DataFrame(scale(data).T,
                               index=data.columns,
                               columns=data.index)
    clusters_raw = []
    for labels, name in zip(cluster_DVs, names):
        subset = scaled_data.loc[labels,:]
        cluster_vec = subset.mean(0)
        clusters_raw.append(cluster_vec)
    raw_space = np.vstack([clusters_raw, scaled_data])
    # turn raw space into distances
    if dist_metric == 'abs_correlation':
        raw_space_distances = squareform(abs_pdist(raw_space))
    else:
        raw_space_distances = squareform(pdist(raw_space, dist_metric))
    
    # plot distances
    distances = {'EFA%s' % c: EFA_space_distances,
                 'subj': raw_space_distances}
    filey=None
    for label, space in distances.items():
        if plot_dir is not None:
            filey = path.join(plot_dir, 
                              'MDS_%s_metric-%s.%s' % (label, dist_metric, ext))
        scale_plot(space, data_colors=colors,
                   cluster_colors=color_palette,
                   cluster_sizes=sizes,
                   dissimilarity='precomputed',
                   filey=filey)

def visualize_importance(importance, ax, xticklabels=True, 
                           yticklabels=True, pad=0, ymax=None, legend=True):
    """Plot task loadings on one axis"""
    importance_vars = importance[0]
    importance_vals = [abs(i)+pad for i in importance[1].T]
    plot_loadings(ax, importance_vals, kind='line', offset=.5,
                  plot_kws={'alpha': 1, 'linewidth': 3})
    # set up x ticks
    xtick_locs = np.arange(0.0, 2*np.pi, 2*np.pi/len(importance_vars))
    ax.set_xticks(xtick_locs)
    ax.set_xticks(xtick_locs+np.pi/len(importance_vars), minor=True)
    if xticklabels:
        if type(importance_vars[0]) == str:
            ax.set_xticklabels(importance_vars, 
                               y=.08, minor=True)
        else:
            ax.set_xticklabels(['Fac %s' % str(i+1) for i in importance_vars], 
                               y=.08, minor=True)
    # set up yticks
    if ymax:
        ax.set_ylim(top=ymax)
    ytick_locs = ax.yaxis.get_ticklocs()
    new_yticks = np.linspace(0, ytick_locs[-1], 7)
    ax.set_yticks(new_yticks)
    if yticklabels:
        labels = np.round(new_yticks,2)
        replace_dict = {i:'' for i in labels[::2]}
        labels = [replace_dict.get(i, i) for i in labels]
        ax.set_yticklabels(labels)
    if legend:
        ax.legend(loc='upper center', bbox_to_anchor=(.5,-.15))
        
def plot_cluster_factors(results, c, rotate='oblimin',  ext='png', plot_dir=None):
    """
    Args:
        EFA: EFA_Analysis object
        c: number of components for EFA
        task_sublists: a dictionary whose values are sets of tasks, and 
                        whose keywords are labels for those lists
    """
    # set up variables
    HCA = results.HCA
    EFA = results.EFA
    
    names, cluster_loadings = zip(*HCA.get_cluster_loading(EFA, rotate=rotate).items())
    cluster_DVs = HCA.get_cluster_DVs(inp='EFA%s_%s' % (EFA.get_c(), rotate))
    cluster_loadings = list(zip([cluster_DVs[n] for n in names], cluster_loadings))
    max_loading = max([max(abs(i[1])) for i in cluster_loadings])
    # plot
    colors = sns.hls_palette(len(cluster_loadings))
    ncols = min(5, len(cluster_loadings))
    nrows = ceil(len(cluster_loadings)/ncols)
    f, axes = plt.subplots(nrows, ncols, 
                               figsize=(ncols*10,nrows*(8+nrows)),
                               subplot_kw={'projection': 'polar'})
    axes = f.get_axes()
    for i, (measures, loading) in enumerate(cluster_loadings):
        plot_loadings(axes[i], loading, kind='line', offset=.5,
              plot_kws={'alpha': .8, 'c': colors[i]})
        axes[i].set_title('Cluster %s' % i, y=1.14, fontsize=25)
        # set tick labels
        xtick_locs = np.arange(0.0, 2*np.pi, 2*np.pi/len(loading))
        axes[i].set_xticks(xtick_locs)
        axes[i].set_xticks(xtick_locs+np.pi/len(loading), minor=True)
        if i%(ncols*2)==0 or i%(ncols*2)==(ncols-1):
            axes[i].set_xticklabels(loading.index,  y=.08, minor=True)
            # set ylim
            axes[i].set_ylim(top=max_loading)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.subplots_adjust(hspace=.5, wspace=.5)
    
    filename = 'polar_factors_EFA%s_%s.%s' % (c, rotate, ext)
    if plot_dir is not None:
        save_figure(f, path.join(plot_dir, filename),
                    {'bbox_inches': 'tight'})
        plt.close()

def plot_silhouette(results, inp='data', labels=None, axes=None,
                    size=4.6,  dpi=300,  ext='png', plot_dir=None):
    HCA = results.HCA
    clustering = HCA.results[inp]
    name = inp
    sample_scores, avg_score = silhouette_analysis(clustering, labels)
    # raw clustering for comparison
    raw_clustering = HCA.results['data']
    _, raw_avg_score = silhouette_analysis(raw_clustering, labels)
    
    if labels is None:
        labels = clustering['labels']
    n_clusters = len(np.unique(labels))
    colors = sns.hls_palette(n_clusters)
    if axes is None:
        fig, (ax, ax2) =  plt.subplots(1, 2, figsize=(size, size*.375))
    else:
        ax, ax2 = axes
    y_lower = 5
    ax.grid(False)
    ax2.grid(linewidth=size/10)
    cluster_names = HCA.get_cluster_names(inp=inp)
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_scores[labels == i+1]
        # skip "clusters" with one value
        if len(ith_cluster_silhouette_values) == 1:
            continue
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        # update y range and plot
        y_upper = y_lower + size_cluster_i
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          alpha=0.7, color=colors[i],
                          linewidth=size/10)
        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.02, y_lower + 0.25 * size_cluster_i, cluster_names[i], fontsize=size/1.7, ha='right')
        # Compute the new y_lower for next plot
        y_lower = y_upper + 5  # 10 for the 0 samples
    ax.axvline(x=avg_score, color="red", linestyle="--", linewidth=size*.1)
    ax.set_xlabel('Silhouette score', fontsize=size, labelpad=5)
    ax.set_ylabel('Cluster Separated DVs', fontsize=size)
    ax.tick_params(pad=size/4, length=size/4, labelsize=size*.8, width=size/10,
                   left=False, labelleft=False, bottom=True)
    ax.set_title('Dynamic tree cut', fontsize=size*1.2, y=1.02)
    ax.set_xlim(-1, 1)
    # plot silhouettes for constant thresholds
    _, scores, _ = get_constant_height_labels(clustering)
    ax2.plot(*zip(*scores), 'o', color='b', 
             markeredgecolor='white', markeredgewidth=size*.1, markersize=size*.5, 
             label='Fixed Height Cut')
    # plot the dynamic tree cut point
    ax2.plot(n_clusters, avg_score, 'o', color ='r', 
             markeredgecolor='white', markeredgewidth=size*.1, markersize=size*.75, 
             label='EFA Dynamic Cut')
    ax2.plot(n_clusters, raw_avg_score, 'o', color ='k', 
             markeredgecolor='white', markeredgewidth=size*.1, markersize=size*.75, 
             label='Raw Dynamic Cut')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_xlabel('Number of clusters', fontsize=size)
    ax2.set_ylabel('Average Silhouette Score', fontsize=size)
    ax2.set_title('Single cut height', fontsize=size*1.2, y=1.02)
    ax2.tick_params(labelsize=size*.8, pad=size/4, length=size/4, width=size/10, bottom=True)
    ax2.legend(loc='center right', fontsize=size*.8)
    plt.subplots_adjust(wspace=.3)
    if plot_dir is not None:
        save_figure(fig, path.join(plot_dir, 
                                         'silhouette_analysis_%s.%s' % (name, ext)),
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
    
# Plotly dependent Sankey plots
# check if plotly exists
import importlib
plotly_spec = importlib.util.find_spec("plotly")
plotly_exists = plotly_spec is not None
if plotly_exists:
    import plotly.plotly as py   
    import plotly.offline as offline

def get_relationship(source_cluster, target_clusters):
    links = {}
    for DV in source_cluster:
        target_index = [i for i,c in enumerate(target_clusters) if DV in c][0]
        links[target_index] = links.get(target_index, 0) + 1
    return links
    
    
def plot_cluster_sankey(results):
    if plotly_exists:
        HCA = results.HCA
        inputs = [i.split('-')[-1] for i in HCA.results.keys() if 'EFA' in i][::-1]
        HCA.get_cluster_DVs(inputs[0])
        sources, targets, values = [], [], []
        source_clusters = HCA.get_cluster_DVs(inputs[0])
        target_clusters = HCA.get_cluster_DVs(inputs[1])
        max_index = len(source_clusters)
        for i, cluster in enumerate(source_clusters):
            links = get_relationship(cluster, target_clusters)
            t, v = zip(*links.items())
            # adjust target index based on last max index
            t = [i+max_index for i in t]
            sources += [i] * len(t)
            targets += t
            values += v
        sankey_df = pd.DataFrame({'Source': sources,
                                 'Target': targets,
                                 'Value': values})
            
            
    
        cs = sns.color_palette('hls', len(source_clusters)).as_hex()
        colors = [cs[s] for s in sources]
        sankey_df = sankey_df.assign(Color=colors)
        
        
        HCA.get_cluster_DVs('EFA5')
        data_trace = dict(
            type='sankey',
            domain = dict(
              x =  [0,1],
              y =  [0,1]
            ),
            orientation = "h",
            valueformat = ".0f",
            node = dict(
              pad = 10,
              thickness = 30,
              line = dict(
                color = "black",
                width = 0.5
              ),
              label = sankey_df['Source'],
              color = sankey_df['Color']
            ),
            link = dict(
              source = sankey_df['Source'].dropna(axis=0, how='any'),
              target = sankey_df['Target'].dropna(axis=0, how='any'),
              value = sankey_df['Value'].dropna(axis=0, how='any'),
              color = sankey_df['Color']
          )
        )
        
        layout =  dict(
            title = "Test",
            height = 772,
            width = 950,
            font = dict(
              size = 10
            ),    
        )
        fig = dict(data=[data_trace], layout=layout)
        py.iplot(fig, validate=True)
    else:
        print("Plotly wasn't found, can't plot!")
    

def plot_HCA(results, plot_dir=None, rotate='oblimin', 
             size=10, dpi=300, verbose=False, ext='png', **dendrogram_kws):
    if plot_dir:
        plot_rotate_dir = path.join(plot_dir, rotate)
    else:
        plot_rotate_dir = None
    c = results.EFA.get_c()
    # plots, woo
#    if verbose: print("Plotting dendrogram heatmaps")
#    plot_clusterings(results, inp='data', plot_dir=plot_dir, verbose=verbose, ext=ext)
#    plot_clusterings(results, inp='EFA%s' % c, plot_dir=plot_dir, verbose=verbose, ext=ext)
    if verbose: print("Plotting dendrograms")
    plot_results_dendrogram(results, rotate=rotate, EFA_clustering=False,
                        title=False,  plot_dir=plot_dir, 
                        size=size, ext=ext, dpi=dpi)
    plot_results_dendrogram(results, rotate=rotate, EFA_clustering=True,
                        title=False, plot_dir=plot_rotate_dir, 
                        size=size, ext=ext, dpi=dpi,
                        **dendrogram_kws)
    if verbose: print("Plotting dendrogram subbranches")
    plot_subbranches(results, rotate=rotate,  EFA_clustering=False,
                     size=size/2, plot_dir=plot_dir, ext=ext, dpi=dpi)
    plot_subbranches(results, rotate=rotate,  EFA_clustering=True,
                     size=size/2, plot_dir=plot_rotate_dir, ext=ext, dpi=dpi)
    if verbose: print("Plotting silhouette analysis")
    plot_silhouette(results, inp='EFA%s_%s' % (c, rotate), size=size,
                    plot_dir=plot_rotate_dir, ext=ext, dpi=dpi)
#    if verbose: print("Plotting clustering similarity")
#    plot_clustering_similarity(results, plot_dir=plot_dir, verbose=verbose, ext=ext)
#    if verbose: print("Plotting cluster polar plots")
#    plot_cluster_factors(results, c, inp='data', plot_dir=plot_dir, ext=ext)
#    plot_cluster_factors(results, c, inp='EFA%s' % c, plot_dir=plot_dir, ext=ext)
    if verbose: print("Plotting MDS space")
    for metric in ['abs_correlation']:
        MDS_visualization(results, c, rotate=rotate, plot_dir=plot_rotate_dir,
                          dist_metric=metric, ext=ext)








