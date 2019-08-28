import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from dimensional_structure.plot_utils import get_short_names, plot_loadings
from selfregulation.utils.plot_utils import beautify_legend, CurvedText, format_num, save_figure

colors = sns.color_palette('Blues_d',3) + sns.color_palette('Reds_d',2)[:1]
shortened_factors = get_short_names()

def visualize_importance(importance, ax, xticklabels=True, yticklabels=True, 
                         axes_linewidth=None, label_size=10,
                         label_scale=0, title=None, 
                         ymax=None, color=colors[1], outline_color=None,
                         show_sign=True):
    importance_vars = importance[0]
    importance_vars = [shortened_factors.get(v,v) for v in importance_vars]
    if importance[1] is not None:
        importance_vals = [abs(i) for i in importance[1]]
        if outline_color is not None:
            plot_loadings(ax, importance_vals, kind='line', offset=.5, 
                          colors=outline_color, plot_kws={'alpha': 1, 
                                                     'linewidth': label_size/2})
    
        plot_loadings(ax, importance_vals, kind='line', offset=.5, 
                      colors=[color], plot_kws={'alpha': 1, 
                                                 'linewidth': label_size/4})
    else:
        ax.set_yticks([])
    ax.grid(linewidth=label_size/8)
    if axes_linewidth:
        plt.setp(list(ax.spines.values()), linewidth=axes_linewidth)

    # set up x ticks
    xtick_locs = np.arange(0.0, 2*np.pi, 2*np.pi/len(importance_vars))
    ax.set_xticks(xtick_locs)
    ax.set_xticks(xtick_locs+np.pi/len(importance_vars), minor=True)
    if xticklabels:
        if type(importance_vars[0]) != str:
            importance_vars = ['Fac %s' % str(i+1) for i in importance_vars]
        ticks = importance_vars
        scale = 1+label_scale
        size = ax.get_position().expanded(scale, scale)
        ax2=ax.get_figure().add_axes(size,zorder=2)
        for i, text in enumerate(ticks):
            fontcolor='k'
            if importance[1][i] < 0 and show_sign:
                fontcolor = 'r'
            arc_start = (i+.25)*2*np.pi/len(importance_vars)
            arc_end = (i+.85)*2*np.pi/len(importance_vars)
            curve = [
                .85*np.cos(np.linspace(arc_start,arc_end,100)),
                .85*np.sin(np.linspace(arc_start,arc_end,100))
            ]  
            plt.plot(*curve, alpha=0)
            curvetext = CurvedText(
                x = curve[0][::-1],
                y = curve[1][::-1],
                text=text, #'this this is a very, very long text',
                axes = ax2,
                va = 'bottom',
                fontsize=label_size,
                color=fontcolor
            )
            ax2.set_xlim([-1,1]); ax2.set_ylim([-1,1])
            ax2.axis('off')
        
    if title:
        ax.set_title(title, fontsize=label_size*1.5, y=1.1)
    # set up yticks
    if len(importance[1]) != 0:
        ax.set_ylim(bottom=0)
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
    """
    # optional to shade to show sign of beta value
    if show_sign:
        data_coords = ax.lines[0].get_data()
        ylim = ax.get_ylim()
        gap = data_coords[0][1]-data_coords[0][0]
        centers = []
        for i, val in enumerate([j for j in importance[1]]):
            if val<0:
                centers.append(data_coords[0][i])
        for center in centers:
            ax.axvspan(xmin=center-gap/2, xmax=center+gap/2,
                       ymin=ylim[0], ymax=ylim[1]*50,
                       facecolor='r', alpha=.1)
    """

def importance_bar_plots(predictions, target_order=None, show_sign=True, 
                colorbar=True, size=5, dpi=300, filename=None):
    #palette = sns.cubehelix_palette(100)
    # plot
    if target_order is None:
        target_order = predictions.keys()
    n_predictors = len(predictions[list(target_order)[0]]['importances'][0])
    #set up color styling
    palette = sns.color_palette('Blues_d',n_predictors)
    # get max r2
    max_r2 = 0
    vals = [predictions[i] for i in target_order]
    max_r2 = max(max_r2, max([i['scores_cv'][0]['R2'] for i in vals]))
    importances = [(i['predvars'], 
                        i['importances'][0]) for i in vals]
    prediction_df = pd.DataFrame([i[1] for i in importances], columns=importances[0][0], index=target_order)    
    prediction_df.sort_values(axis=1, by=prediction_df.index[0], inplace=True, ascending=False)
    
    # plot
    sns.set_style('white')
    ax = prediction_df.plot(kind='bar', edgecolor=None, linewidth=0,
                             figsize=(size,size*.67), color=palette)
    fig = ax.get_figure()
    ax.tick_params(labelsize=size)
    #ax.tick_params(axis='x', rotation=0)
    ax.set_ylabel(r'Standardized $\beta$', fontsize=size*1.5)
    # set up legend and other aesthetic
    ax.grid(axis='y', linewidth=size/10)
    leg = ax.legend(frameon=False, fontsize=size*1.5, bbox_to_anchor=(1.25,.8), 
                     handlelength=0, handletextpad=0, framealpha=1)
    beautify_legend(leg, colors=palette)          
    for name, spine in ax.spines.items():
        spine.set_visible(False)
    if filename is not None:
        save_figure(fig, filename, {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
    else:
        return fig
            
def importance_polar_plots(predictions, target_order=None, show_sign=True, 
                colorbar=True, size=5, dpi=300, filename=None):
    # set up color styling
    palette = sns.color_palette('Blues_d',100)
    #palette = sns.cubehelix_palette(100)
    # plot
    if target_order is None:
        target_order = list(predictions.values())[0].keys()
    N = len(target_order)
    f = plt.figure(figsize=(size, size))
    background_ax = f.add_axes([0,0,1,1])
    polar_axes = []
    subplot_size = 1/N
    # get max r2
    max_r2 = 0
    for prediction in predictions.values():
        vals = [prediction[i] for i in target_order]
        max_r2 = max(max_r2, max([i['scores_cv'][0]['R2'] for i in vals]))
    for row_i, (name, prediction) in enumerate(predictions.items()):
        # get importances
        vals = [prediction[i] for i in target_order]
        importances = [(i['predvars'], 
                        i['importances'][0]) for i in vals]
        r2s = [i['scores_cv'][0]['R2'] for i in vals]
        for i, target in enumerate(target_order):
            xticklabels = True
            polar_axes.append(f.add_axes([subplot_size*i*1.3, 
                                    row_i*1.4*subplot_size, 
                                    subplot_size, subplot_size], 
                projection='polar'))
            importance = importances[i]
            visualize_importance(importance, polar_axes[-1],
                         yticklabels=False, 
                         xticklabels=xticklabels,
                         label_size=size*1.5,
                         color=palette[max(int(r2s[i]/max_r2*len(palette))-1,0)],
                         outline_color='k',
                         axes_linewidth=size/20,
                         label_scale=.25,
                         show_sign=show_sign)
            polar_axes[-1].text(.5, -.2, 'R2: ' + format_num(r2s[i]), 
                      zorder=5, fontsize=size*1.5,
                      fontweight='bold',
                      ha='center',
                      transform=polar_axes[-1].transAxes)
            # change axis color
            polar_axes[-1].grid(color=[.6,.6,.6])
            polar_axes[-1].set_facecolor((0.91, 0.91, 0.94, 1.0))
    # add column labels
    for i, label in enumerate(target_order):
        pos = polar_axes[i-3].get_position().bounds
        x_pos = pos[0]+pos[2]*.5
        y_pos = pos[1]+pos[3]
        background_ax.text(x_pos, y_pos+.05, 
                           '\n'.join(label.split()), 
                           fontsize=size*2,
                           fontweight='bold',
                           ha='center')
    # add row labels
    for i, key in enumerate(predictions.keys()):
        pos = polar_axes[i*N].get_position().bounds
        x_pos = pos[0]
        y_pos = pos[1]+pos[3]*.5
        background_ax.text(x_pos-.1, y_pos, 
                           ' '.join(key.title().split('_')), 
                           fontsize=size*2,
                           fontweight='bold',
                           va='center',
                           rotation=90)
    # make background ax invisible
    background_ax.tick_params(bottom=False, left=False,
                              labelbottom=False, labelleft=False)
    # add colorbar
    if colorbar == True:
        # get x position of center plots
        if N%2==1:
            pos = polar_axes[N//2].get_position().bounds
            x_pos = pos[0]+pos[2]*.5
        else:
            pos1 = polar_axes[N//2-1].get_position().bounds
            pos2 = polar_axes[N//2].get_position().bounds
            x_pos = (pos2[0]-(pos1[0]+pos[2]))*2+pos[0]+pos[2]
    
        color_ax = f.add_axes([x_pos-.3,-.2, .6, .025])
        cbar = mpl.colorbar.ColorbarBase(ax=color_ax, cmap=ListedColormap(palette),
                                  orientation='horizontal')
        cbar.set_ticks([0,1])
        cbar.set_ticklabels([0, format_num(max_r2)])
        color_ax.tick_params(labelsize=size)
        cbar.set_label('R2', fontsize=size*1.5)
    for key, spine in background_ax.spines.items():
        spine.set_visible(False)
    if filename is not None:
        save_figure(f, filename, {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
    else:
        return f

        
def plot_prediction(predictions, shuffled_predictions, 
                    target_order=None, 
                    metric='R2', size=4.6,  
                    dpi=300, filename=None):
    """ Plots predictions resulting from "run_prediction" function
    
    Args:
        predictions: dictionary of run_prediction results
        shuffled_predictions: dictionary of run_prediction shuffled results
        target_order: (optional) a list of targets to order the plot
        metric: which metric from the output of run_prediction to use
        size: figure size
        dpi: dpi to use for saving
        ext: extension to use for saving (e.g., pdf)
        filename: if provided, save to this location
    """
    colors = sns.color_palette('Blues_d',5)
    basefont = max(size, 5)
    sns.set_style('white')
    if target_order is None:
        target_order = predictions.keys()
    prediction_keys = predictions.keys()
    # get prediction success
    # plot
    shuffled_grey = [.3,.3,.3,.3]
    # plot variables
    figsize = (size, size*.75)
    fig = plt.figure(figsize=figsize)
    # plot bars
    width=1/(len(prediction_keys)+1)
    ax1 = fig.add_axes([0,0,1,.5]) 
    for predictor_i, key in enumerate(prediction_keys):
        prediction = predictions[key]
        shuffled_prediction = shuffled_predictions[key]
        r2s = [[k,prediction[k]['scores_cv'][0][metric]] for k in target_order]
        # get shuffled values
        shuffled_r2s = []
        for i, k in enumerate(target_order):
            # normalize r2s to significance
            R2s = [i[metric] for i in shuffled_prediction[k]['scores_cv']]
            R2_95 = np.percentile(R2s, 95)
            shuffled_r2s.append((k,R2_95))
        # convert nans to 0
        r2s = [(i, k) if k==k else (i,0) for i, k in r2s]
        shuffled_r2s = [(i, k) if k==k else (i,0) for i, k in shuffled_r2s]
        
        ind = np.arange(len(r2s))-(width*(len(prediction_keys)/2-1))
        ax1.bar(ind+width*predictor_i, [i[1] for i in r2s], width, 
                label='%s Prediction' % ' '.join(key.title().split('_')),
                linewidth=0, color=colors[predictor_i])
        # plot shuffled values above
        if predictor_i == len(prediction_keys)-1:
            shuffled_label = '95% shuffled prediction'
        else:
            shuffled_label = None
        ax1.bar(ind+width*predictor_i, [i[1] for i in shuffled_r2s], width, 
                 color=shuffled_grey, linewidth=0, 
                 label=shuffled_label)
        
    ax1.set_xticks(np.arange(0,len(r2s))+width/2)
    ax1.set_xticklabels(['\n'.join(i[0].split()) for i in r2s], 
                        rotation=90, fontsize=basefont*.75, ha='center')
    ax1.tick_params(axis='y', labelsize=size*1.2)
    ax1.tick_params(length=size/2, width=size/10, pad=size/2, bottom=True, left=True)
    xlow, xhigh = ax1.get_xlim()
    if metric == 'R2':
        ax1.set_ylabel(r'$R^2$', fontsize=basefont*1.5, labelpad=size*1.5)
    else:
        ax1.set_ylabel(metric, fontsize=basefont*1.5, labelpad=size*1.5)
    # add a legend
    leg = ax1.legend(fontsize=basefont*1.4, loc='upper right', 
                     bbox_to_anchor=(1.3, 1.1), frameon=True, 
                     handlelength=0, handletextpad=0, framealpha=1)
    beautify_legend(leg, colors[:len(predictions)]+[shuffled_grey])
    # draw grid
    ax1.set_axisbelow(True)
    plt.grid(axis='y', linestyle='dotted', linewidth=size/6)
    plt.setp(list(ax1.spines.values()), linewidth=size/10)
    if filename is not None:
        save_figure(fig, filename, {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
    else:
        return fig

def plot_prediction_scatter(predictions, predictors, targets, 
                            target_order=None, metric='R2', size=4.6,  
                            dpi=300, filename=None):
    # subset predictors
    predictors = predictors.loc[targets.index]
    if target_order is None:
        target_order = predictions.keys()
        
    sns.set_style('white')
    n_cols = 4
    n_rows = math.ceil(len(target_order)/n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(size, size/n_cols*n_rows))
    axes = fig.get_axes()
    for i,v in enumerate(target_order):
        MAE = format_num(predictions[v]['scores_cv'][0]['MAE'])
        R2 = format_num(predictions[v]['scores_cv'][0]['R2'])
        axes[i].set_title('%s\nR2: %s, MAE: %s' % ('\n'.join(v.split()), R2, MAE), 
            fontweight='bold', fontsize=size*1)
        clf=predictions[v]['clf']
        axes[i].scatter(targets[v], clf.predict(predictors), s=size*2.5,
                        edgecolor='white', linewidth=size/30)  
        axes[i].tick_params(length=0, labelsize=0)
        # add diagonal
        xlim = axes[i].get_xlim()
        ylim = axes[i].get_ylim()
        axes[i].plot(xlim, ylim, ls="-", c=".5", zorder=-1)
        axes[i].set_xlim(xlim); axes[i].set_ylim(ylim)
        for spine in ['top', 'right']:
            axes[i].spines[spine].set_visible(False)
        if i%n_cols==0:
            axes[i].set_ylabel('Predicted Score', fontsize=size*1.2)
    for ax in axes[-(len(target_order)+1):]:
        ax.set_xlabel('Target Score', fontsize=size*1.2)
    
    empty_plots = n_cols*n_rows - len(targets.columns)
    if empty_plots > 0:
        for ax in axes[-empty_plots:]:
            ax.set_visible(False)
    plt.subplots_adjust(hspace=.6, wspace=.3)
    if filename is not None:
        save_figure(fig, filename, {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()

def plot_RSA(corr, cluster=False, size=8, dpi=300, filename=None):
    """ plots similarity of ontological fingerprints between outcomes """
    figsize = (size,size)
    if cluster == False:
        f = plt.figure(figsize=figsize)
        ax=sns.heatmap(corr, square=True,
                     cmap=sns.diverging_palette(220,15,n=100,as_cmap=True))
    else:
        f = sns.clustermap(corr,
                         cmap=sns.diverging_palette(220,15,n=100,as_cmap=True),
                         figsize=figsize)
        ax = f.ax_heatmap
        corr = f.data2d
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    if filename is not None:
        save_figure(f, filename, {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
    return corr

def plot_outcome_ontological_similarity(predictions, size=8, cluster=False,
                                        reorder=None,
                                        dpi=300, filename=None):
    """ plots similarity of ontological fingerprints between outcomes """


    targets = list(predictions.keys())
    predictors = predictions[targets[0]]['predvars']
    importances = np.vstack([predictions[k]['importances'] for k in targets])
    # convert to dataframe
    df = pd.DataFrame(importances, index=targets, columns=predictors)
    if reorder is not None:
        df = df.loc[reorder, :]
    corr = plot_RSA(df.T.corr(), cluster, size, dpi, filename)
    return corr

def plot_predictors_comparison(R2_df, size=2, dpi=300, filename=None):
    CV_df = R2_df.filter(regex='CV', axis=0)
    CV_corr = CV_df.corr(method='spearman')
    
    max_R2 = round(CV_df.max(numeric_only=True).max(),1)
    size=2
    grid = sns.pairplot(CV_df, hue='Target_Cat', height=size)
    for i, row in enumerate(grid.axes):
        for j, ax in enumerate(row):
            ax.set_xlim([0,max_R2])
            ax.set_ylim([0,max_R2])
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.plot(xlim, ylim, ls=":", c=".5", zorder=-1)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            if j<i:
                ax.text(.5, 1, r'$\rho$ = %s' % format_num(CV_corr.iloc[i,j]),
                        ha='center',
                        fontsize=size*7,
                        fontweight='bold',
                        transform=ax.transAxes)
            if j>i:
                ax.set_visible(False)
    if filename is not None:
        save_figure(grid.fig, filename, {'bbox_inches': 'tight', 'dpi': dpi})
    else:
        return grid