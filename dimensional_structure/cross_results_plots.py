#!/usr/bin/env python

# Script to generate all_results or plots across all_results objects
from itertools import combinations, product
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from os import path, remove
import pandas as pd
import pickle
import seaborn as sns
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import cross_val_score
import svgutils.transform as sg
import subprocess
from dimensional_structure.HCA_plots import plot_silhouette
from selfregulation.utils.plot_utils import (beautify_legend, 
                                             format_variable_names, 
                                             place_letter, save_figure)
from selfregulation.utils.r_to_py_utils import get_attr
from selfregulation.utils.utils import get_retest_data


def extract_tril(mat, k=0):
    return mat[np.tril_indices_from(mat, k=k)]


def plot_corr_hist(all_results, reps=100, size=4.6, 
                   dpi=300, ext='png', plot_dir=None):
    colors = sns.color_palette('Blues_d',3)[0:2] + sns.color_palette('Reds_d',2)[:1]
    survey_corr = abs(all_results['survey'].data.corr())
    task_corr = abs(all_results['task'].data.corr())
    all_data = pd.concat([all_results['task'].data, all_results['survey'].data], axis=1)
    datasets = [('survey', all_results['survey'].data), 
                ('task', all_results['task'].data), 
                ('all', all_data)]
    # get cross corr
    cross_corr = abs(all_data.corr()).loc[survey_corr.columns,
                                                    task_corr.columns]
    
    plot_elements = [(extract_tril(survey_corr.values,-1), 'Within Surveys'),
                     (extract_tril(task_corr.values,-1), 'Within Tasks'),
                     (cross_corr.values.flatten(), 'Surveys x Tasks')]
    
    # get shuffled 95% correlation
    shuffled_95 = []
    for label, df in datasets:
        shuffled_corr = np.array([])
        for _ in range(reps):
            # create shuffled
            shuffled = df.copy()
            for i in shuffled:
                shuffle_vec = shuffled[i].sample(len(shuffled)).tolist()
                shuffled.loc[:,i] = shuffle_vec
            if label == 'all':
                shuffled_corr = abs(shuffled.corr()).loc[survey_corr.columns,
                                                    task_corr.columns]
            else:
                shuffled_corr = abs(shuffled.corr())
            np.append(shuffled_corr, extract_tril(shuffled_corr.values,-1))
        shuffled_95.append(np.percentile(shuffled_corr,95))
    
    # get cross_validated r2
    average_r2 = {}
    for (slabel, source), (tlabel, target) in product(datasets[:-1], repeat=2):
        scores = []
        for var, values in target.iteritems():
            if var in source.columns:
                predictors = source.drop(var, axis=1)
            else:
                predictors = source
            lr = RidgeCV()  
            cv_score = np.mean(cross_val_score(lr, predictors, values, cv=10))
            scores.append(cv_score)
        average_r2[(slabel, tlabel)] = np.mean(scores)

                
    # bring everything together
    plot_elements = [(extract_tril(survey_corr.values,-1), 'Within Surveys', 
                      average_r2[('survey','survey')]),
                     (extract_tril(task_corr.values,-1), 'Within Tasks',
                      average_r2[('task','task')]),
                     (cross_corr.values.flatten(), 'Surveys x Tasks',
                      average_r2[('survey', 'task')])]
    
    with sns.axes_style('white'):
        f, axes = plt.subplots(1,3, figsize=(10,4))
        plt.subplots_adjust(wspace=.3)
        for i, (corr, label, r2) in enumerate(plot_elements):
            #h = axes[i].hist(corr, normed=True, color=colors[i], 
            #         bins=12, label=label, rwidth=1, alpha=.4)
            sns.kdeplot(corr, ax=axes[i], color=colors[i], shade=True,
                        label=label, linewidth=3)
            axes[i].text(.4, axes[i].get_ylim()[1]*.5, 'CV-R2: {0:.2f}'.format(r2))
        for i, ax in enumerate(axes):
            ax.vlines(shuffled_95[i], *ax.get_ylim(), color=[.2,.2,.2], 
                      linewidth=2, linestyle='dashed', zorder=10)
            ax.set_xlim(0,1)
            ax.set_ylim(0, ax.get_ylim()[1])
            ax.set_xticks([0,.5,1])
            ax.set_xticklabels([0,.5,1], fontsize=16)
            ax.set_yticks([])
            ax.spines['right'].set_visible(False)
            #ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            leg=ax.legend(fontsize=14, loc='upper center')
            beautify_legend(leg, [colors[i]])
        axes[1].set_xlabel('Pearson Correlation', fontsize=20, labelpad=10)
        axes[0].set_ylabel('Normalized Density', fontsize=20, labelpad=10)
    
    # save
    if plot_dir is not None:
        # make histogram plot
        save_figure(f, path.join(plot_dir, 'within-across_correlations.%s' % ext),
                                {'bbox_inches': 'tight', 'dpi': dpi})
    
        
    
def plot_corr_heatmap(all_results, EFA=False, size=4.6, 
                   dpi=300, ext='png', plot_dir=None):
    def get_EFA_HCA(results, EFA):
        if EFA == False:
            return results.HCA.results['data']
        else:
            c = results.EFA.results['num_factors']
            return results.HCA.results['EFA%s_oblimin' % c]
    

    survey_order = get_EFA_HCA(all_results['survey'], EFA)['reorder_vec']
    task_order = get_EFA_HCA(all_results['task'], EFA)['reorder_vec']
    
    if EFA == False:
        all_data = pd.concat([all_results['task'].data.iloc[:, task_order], 
                              all_results['survey'].data.iloc[:, survey_order]], 
                            axis=1)
    else:
        all_data = pd.concat([all_results['task'].EFA.get_loading().T.iloc[:, task_order], 
                              all_results['survey'].EFA.get_loading().T.iloc[:, survey_order]], 
                            axis=1)

    f = plt.figure(figsize=(size,size))
    ax = f.add_axes([.05,.05,.8,.8])
    cbar_ax = f.add_axes([.86,.1,.04,.7])
    corr = abs(all_data.corr())
    sns.heatmap(corr, square=True, ax=ax, cbar_ax=cbar_ax,
                xticklabels=False, yticklabels=False,
                vmax=1, vmin=0,
                cbar_kws={'ticks': [0, 1]},
                cmap=ListedColormap(sns.color_palette('Reds', 100)))
    # add separating lines
    if ax.get_ylim()[0] > ax.get_ylim()[1]:
        ax.hlines(len(task_order), 0, all_data.shape[1], lw=size/4, 
                   color='k', linestyle='--')
    else:
        ax.hlines(len(survey_order), 0, all_data.shape[1], lw=size/4, 
                   color='k', linestyle='--')
    ax.vlines(len(task_order), 0, all_data.shape[1], lw=size/4, 
               color='k', linestyle='--')
    # format cbar
    cbar_ax.tick_params(axis='y', length=0)
    cbar_ax.set_yticklabels([0, 1])
    cbar_ax.tick_params(labelsize=size*2, pad=size/2)
    cbar_ax.set_ylabel('Pearson Correlation', rotation=-90, labelpad=size*2, fontsize=size*2)
    # add bars to indicate category
    left_ax = f.add_axes([.01,.05,.04,.8])
    bottom_ax = f.add_axes([.05,0.01,.8,.04])
    left_ax.axis('off'); bottom_ax.axis('off')
    perc_task = len(task_order)/all_data.shape[1]
    # add labels
    left_ax.text(0, (1-perc_task/2), 'Task DVs', rotation=90, va='center', fontsize=size*3)
    left_ax.text(0, ((1-perc_task)/2), 'Survey DVs', rotation=90, va='center', fontsize=size*3)
    bottom_ax.text(perc_task/2, 0, 'Task DVs', ha='center', fontsize=size*3)
    bottom_ax.text((1-(1-perc_task)/2), 0, 'Survey DVs', ha='center', fontsize=size*3)
    if plot_dir is not None:
        # make histogram plot
        save_figure(f, path.join(plot_dir, 'data_correlations.%s' % ext),
                                {'dpi': dpi,
                                 'transparent': True})   
        plt.close()
    else:
        return f

def plot_glasso_edge_strength(all_results, graph_loc,  size=4.6, 
                             dpi=300, ext='png', plot_dir=None):
    task_length = all_results['task'].data.shape[1]
    g = pickle.load(open(graph_loc, 'rb'))
    # subset graph
    task_within = squareform(g.graph_to_dataframe().iloc[:task_length, :task_length])
    survey_within = squareform(g.graph_to_dataframe().iloc[task_length:, task_length:])
    across = g.graph_to_dataframe().iloc[:task_length, task_length:].values.flatten()
    

    titles = ['Within Tasks', 'Within Surveys', 'Between Tasks And Surveys']
    colors = [sns.color_palette('Blues_d',3)[0],
              sns.color_palette('Reds_d',3)[0],
              [0,0,0]]
    
    with sns.axes_style('whitegrid'):
        f, axes = plt.subplots(3,1, figsize=(size,size*1.5))

    for i, corr in enumerate([task_within, survey_within, across]):
        sns.stripplot(corr, jitter=.2, alpha=.5, orient='h', ax=axes[i],
                      color=colors[i], s=size/2)
        
    max_x = max([ax.get_xlim()[1] for ax in axes])*1.1
    for i, ax in enumerate(axes):
        [i.set_linewidth(size*.3) for i in ax.spines.values()]
        ax.grid(linewidth=size*.15)
        ax.set_xlim([0, max_x])
        ax.text(max_x*.02, -.35, titles[i], color=colors[i], ha='left',
                fontsize=size*3.5)
        ax.set_xticks(np.arange(0, round(max_x*10)/10,.1))
        if i!=(len(axes)-1):
            ax.set_xticklabels([])
        else:
            ax.tick_params(labelsize=size*2.5, pad=size, length=0)
    axes[-1].set_xlabel('Edge Weight', fontsize=size*5)
    plt.subplots_adjust(hspace=0)
    if plot_dir is not None:
        # make histogram plot
        save_figure(f, path.join(plot_dir, 'glasso_edge_strength.%s' % ext),
                                {'dpi': dpi,
                                 'transparent': True})   
        plt.close()
    else:
        return f


def plot_cross_within_prediction(prediction_loc, size=4.6, 
                                 dpi=300, ext='png', plot_dir=None):
    predictions = pickle.load(open(prediction_loc, 'rb'))

    titles = ['Within Tasks', 'Within Surveys', 'Survey-By-Tasks', 'Task-By-Surveys']
    colors = [sns.color_palette('Blues_d',3)[0],
              sns.color_palette('Reds_d',3)[0],
              [.4,.4,.4],
              [.4,.4,.4]]
    
    with sns.axes_style('whitegrid'):
        f, axes = plt.subplots(4,1, figsize=(size,size*1.5))

    for i, vals in enumerate([predictions['within']['task'],
                              predictions['within']['survey'],
                              predictions['across']['task_to_survey'],
                              predictions['across']['survey_to_task']]):
        sns.violinplot(list(vals.values()), orient='h', color=colors[i],
                    ax=axes[i], width=.5, linewidth=size*.3)
        
    min_x = min([ax.get_xlim()[0] for ax in axes])
    for i, ax in enumerate(axes):
        [i.set_linewidth(size*.3) for i in ax.spines.values()]
        ax.grid(linewidth=size*.15, which='both')
        ax.set_xlim([min_x, 1])
        ax.text(min_x+(1-min_x)*.02, -.34, titles[i], color=colors[i], ha='left',
                fontsize=size*3.5)
        xticks = np.arange(math.floor(min_x*10)/10,1,.2)
        ax.set_xticks(xticks)
        if i!=(len(axes)-1):
            ax.set_xticklabels([])
        else:
            ax.tick_params(labelsize=size*2.5, pad=size, length=0)
    axes[-1].set_xlabel(r'$R^2$', fontsize=size*5)
    plt.subplots_adjust(hspace=0)
    if plot_dir is not None:
        # make histogram plot
        save_figure(f, path.join(plot_dir, 'cross_prediction.%s' % ext),
                                {'dpi': dpi,
                                 'transparent': True})   
        plt.close()
    else:
        return f
 
def plot_cross_relationship(all_results, graph_loc, prediction_loc, size=4.6,
                            dpi=300, ext='pdf', plot_dir=None):
    assert ext in ['pdf', 'svg'], 'Must use svg or pdf'
    tmp_dir = '/tmp/'
    plot_corr_heatmap(all_results, size=size/3, plot_dir=tmp_dir, ext='svg')
    plot_glasso_edge_strength(all_results, graph_loc, size/4, plot_dir=tmp_dir, ext='svg')
    plot_cross_within_prediction(prediction_loc, size/4, plot_dir=tmp_dir, ext='svg')

    fig1 = sg.fromfile('/tmp/data_correlations.svg')
    fig2 = sg.fromfile('/tmp/cross_prediction.svg')
    fig3 = sg.fromfile('/tmp/glasso_edge_strength.svg')
    width = float(fig1.get_size()[0][:-2]) 
    height = float(fig2.get_size()[1][:-2]) 
    fig = sg.SVGFigure(width*2.5, height)
    fig.root.set("viewbox", "0 0 %s %s" % (width*2, height))
    plot1 = fig1.getroot()
    plot2 = fig2.getroot()
    plot3 = fig3.getroot()
    # move plots
    plot2.moveto(width, 0)
    plot3.moveto(width*1.8, 0)
    fig.append([plot1, plot2, plot3])
    # add text
    txt1 = sg.TextElement(0, height*.1, "A", size=size*1.5, weight="bold")
    txt2 = sg.TextElement(width*1.05, height*.1, "B", size=size*1.5, weight="bold")
    txt3 = sg.TextElement(width*1.85, height*.1, "C", size=size*1.5, weight="bold")
    fig.append([txt1, txt2, txt3])
    # save
    svg_file = path.join(plot_dir, 'cross_relationship.svg')
    fig.save(svg_file)
    if ext=='pdf':
        pdf_file = path.join(plot_dir, 'cross_relationship.pdf')
        a=subprocess.Popen('cairosvg %s -o %s' % (svg_file, pdf_file),
                            shell=True, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
    # remove temp files
    remove('/tmp/data_correlations.svg')
    remove('/tmp/glasso_edge_strength.svg')
    remove('/tmp/cross_prediction.svg')
    
def plot_EFA_relationships(all_results):
    EFA_all_results = {k:v.EFA for k,v in all_results.items()}
    scores = {k:v.get_scores() for k,v in EFA_all_results.items()}
    # quantify relationships using linear regression
    for name1, name2 in combinations(scores.keys(), 2):
        scores1 = scores[name1]
        scores2 = scores[name2]
        lr = LinearRegression()  
        cv_score = np.mean(cross_val_score(lr, scores1, scores2, cv=10))
        print(name1, name2, cv_score)
    # plot
    # plot task factors in task PCA space
    pca = PCA(2)
    task_pca = pca.fit_transform(scores['task'])
    palettes = ['Reds', 'Blues', 'Greens']
    all_colors = []
    # plot scores in task PCA space
    f, ax = plt.subplots(figsize=[12,8])
    ax.set_facecolor('white')

    for k,v in scores.items():
        palette = sns.color_palette(palettes.pop(), n_colors = len(v.columns))
        all_colors += palette
        lr = LinearRegression()
        lr.fit(task_pca, v)
        for i, coef in enumerate(lr.coef_):
            plt.plot([0,coef[0]], [0, coef[1]], linewidth=3, 
                     c=palette[i], label=k+'_'+str(v.columns[i]))
    leg = plt.legend(bbox_to_anchor=(.8, .5))
    frame = leg.get_frame()
    frame.set_color('black')
    beautify_legend(leg, all_colors)

def plot_BIC(all_results, size=4.6, dpi=300, ext='png', plot_dir=None):
    """ Plots BIC and SABIC curves
    
    Args:
        all_results: a dimensional structure all_results object
        dpi: the final dpi for the image
        ext: the extension for the saved figure
        plot_dir: the directory to save the figure. If none, do not save
    """
    all_colors = [sns.color_palette('Blues_d',3)[0:3],
              sns.color_palette('Reds_d',3)[0:3],
              sns.color_palette('Greens_d',3)[0:3],
              sns.color_palette('Oranges_d',3)[0:3]]
    height= size*.75/len(all_results)
    with sns.axes_style('white'):
        fig, axes = plt.subplots(1, len(all_results), figsize=(size, height))
    for i, results in enumerate([all_results[key] for key in ['task','survey']]):
        ax1 = axes[i]
        name = results.ID.split('_')[0].title()
        EFA = results.EFA
        # Plot BIC and SABIC curves
        colors = all_colors[i]
        with sns.axes_style('white'):
            x = list(EFA.results['cscores_metric-BIC'].keys())
            # score keys
            keys = [k for k in EFA.results.keys() if 'cscores' in k]
            for key in keys:
                metric = key.split('-')[-1]
                BIC_scores = [EFA.results[key][i] for i in x]
                BIC_c = EFA.results['c_metric-%s' % metric]
                ax1.plot(x, BIC_scores,  'o-', c=colors[0], lw=size/6, label=metric,
                         markersize=height*2)
                ax1.plot(BIC_c, BIC_scores[BIC_c-1], '.', color='white',
                         markeredgecolor=colors[0], markeredgewidth=height/2, 
                         markersize=height*4)
            if i==0:
                if len(keys)>1:
                    ax1.set_ylabel('Score', fontsize=height*3)
                    leg = ax1.legend(loc='center right',
                                     fontsize=height*3, markerscale=0)
                    beautify_legend(leg, colors=colors)
                else:
                    ax1.set_ylabel(metric, fontsize=height*4)
            ax1.set_xlabel('# Factors', fontsize=height*4)
            ax1.set_xticks(x)
            ax1.set_xticklabels(x)
            ax1.tick_params(labelsize=height*2, pad=size/4, length=0)
            ax1.set_title(name, fontsize=height*4, y=1.01)
            ax1.grid(linewidth=size/8)
            [i.set_linewidth(size*.1) for i in ax1.spines.values()]
    if plot_dir is not None:
        save_figure(fig, path.join(plot_dir, 'BIC_curves.%s' % ext),
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
            
def plot_cross_silhouette(all_results, rotate, size=4.6,  dpi=300, 
                    ext='png', plot_dir=None):
    with sns.axes_style('white'):
        fig, axes =  plt.subplots(len(all_results), 2, 
                                  figsize=(size, size*.375*len(all_results)))
    axes = fig.get_axes()
    letters = [chr(i).upper() for i in range(ord('a'),ord('z')+1)]
    
    for i, (name, results) in enumerate(all_results.items()):
        ax = axes[i*2]
        ax2 = axes[i*2+1]
        inp = 'EFA%s_%s' % (results.EFA.get_c(), rotate)
        plot_silhouette(results, inp=inp, axes=(ax,ax2), size=size)
        ax.set_ylabel('%s cluster separated DVs' % name.title(), fontsize=size*1.2)
        ax2.set_ylabel('%s average silhouette score' % name.title(), fontsize=size*1.2)
        if i == 0:
            ax.set_xlabel('')
            ax2.set_xlabel('')
        else:
            ax.set_xlabel('Silhouette score', fontsize=size*1.2)
            ax2.set_xlabel('Number of clusters', fontsize=size*1.2)
        if i != 0:
            ax.set_title('')
            ax2.set_title('')
        [i.set_linewidth(size*.1) for i in ax.spines.values()]
        [i.set_linewidth(size*.1) for i in ax2.spines.values()]
    plt.subplots_adjust(hspace=.2)
    max_x = max([ax.get_xlim()[1] for ax in axes[::2]])
    min_x = min([ax.get_xlim()[0] for ax in axes[::2]])
    for i in range(len(all_results)):
        ax = axes[i*2]
        ax2 = axes[i*2+1]
        ax.set_xlim([min_x, max_x])
        place_letter(ax, letters.pop(0), fontsize=size*9/4.6)
        place_letter(ax2, letters.pop(0), fontsize=size*9/4.6)
        
    if plot_dir is not None:
        save_figure(fig, path.join(plot_dir, rotate,
                                         'silhouette_analysis.%s' % ext),
                    {'dpi': dpi})
        plt.close()

def plot_cross_communality(all_results, rotate='oblimin', retest_threshold=.2,
                           size=4.6, dpi=300, ext='png', plot_dir=None):
    
    retest_data = None
    num_cols = 2
    num_rows = math.ceil(len(all_results.keys())/2)
    with sns.axes_style('white'):
        f, axes = plt.subplots(num_rows, num_cols, figsize=(size, size/2*num_rows))
    max_y = 0
    for i, (name, results) in enumerate(all_results.items()):
        if retest_data is None:
            # load retest data
            retest_data = get_retest_data(dataset=results.dataset.replace('Complete','Retest'))
            if retest_data is None:
                print('No retest data found for datafile: %s' % results.dataset)
        c = results.EFA.get_c()
        EFA = results.EFA
        loading = EFA.get_loading(c, rotate=rotate)
        # get communality from psych out
        fa = EFA.results['factor_tree_Rout_%s' % rotate][c]
        communality = get_attr(fa, 'communalities')
        communality = pd.Series(communality, index=loading.index)
        # alternative calculation
        #communality = (loading**2).sum(1).sort_values()
        communality.index = [i.replace('.logTr','') for i in communality.index]
        
        # reorder data in line with communality
        retest_subset= retest_data.loc[communality.index]
        # reformat variable names
        communality.index = format_variable_names(communality.index)
        retest_subset.index = format_variable_names(retest_subset.index)
        if len(retest_subset) > 0:
            # noise ceiling
            noise_ceiling = retest_subset.pearson
            # remove very low reliabilities
            if retest_threshold:
                noise_ceiling[noise_ceiling<retest_threshold]= np.nan
            # adjust
            adjusted_communality = communality/noise_ceiling
            
        # plot communality histogram
        if len(retest_subset) > 0:
            ax = axes[i]
            ax.set_title(name.title(), fontweight='bold', fontsize=size*2)
            colors = sns.color_palette(n_colors=2, desat=.75)
            sns.kdeplot(communality, linewidth=size/4, ax=ax, vertical=True,
                        shade=True, label='Communality', color=colors[0])
            sns.kdeplot(adjusted_communality, linewidth=size/4, ax=ax, vertical=True,
                        shade=True, label='Adjusted Communality', color=colors[1])
            xlim = ax.get_xlim()
            ax.hlines(np.mean(communality), xlim[0], xlim[1],
                      color=colors[0], linewidth=size/4, linestyle='--')
            ax.hlines(np.mean(adjusted_communality), xlim[0], xlim[1],
                      color=colors[1], linewidth=size/4, linestyle='--')
            ax.set_xticks([])
            ax.tick_params(labelsize=size*1.2)
            ax.set_ylim(0, ax.get_ylim()[1])
            ax.set_xlim(0, ax.get_xlim()[1])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if (i+1) == len(all_results):
                ax.set_xlabel('Normalized Density', fontsize=size*2)
                leg=ax.legend(fontsize=size*1.5, loc='upper right',
                              bbox_to_anchor=(1.2, 1.0), 
                              handlelength=0, handletextpad=0)
                beautify_legend(leg, colors)
            elif i>=len(all_results)-2:
                ax.set_xlabel('Normalized Density', fontsize=size*2)
                ax.legend().set_visible(False)
            else:
                ax.legend().set_visible(False)
            if i%2==0:
                ax.set_ylabel('Communality', fontsize=size*2)
                ax.tick_params(labelleft=True, left=True, 
                               length=size/4, width=size/8)
            else:
                ax.tick_params(labelleft=False, left=True, 
                               length=0, width=size/8)
            # update max_x
            if ax.get_ylim()[1] > max_y:
                max_y = ax.get_ylim()[1]
            ax.grid(False)
            [i.set_linewidth(size*.1) for i in ax.spines.values()]
        for ax in axes:
            ax.set_ylim((0, max_y))
        plt.subplots_adjust(wspace=0)
                    
        if plot_dir:
            filename = 'communality_adjustment.%s' % ext
            save_figure(f, path.join(plot_dir, rotate, filename), 
                        {'bbox_inches': 'tight', 'dpi': dpi})
            plt.close()
        
    

