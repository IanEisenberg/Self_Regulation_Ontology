import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from os import path
import pandas as pd
import pickle
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from dimensional_structure.plot_utils import get_short_names, plot_loadings
from dimensional_structure.utils import hierarchical_cluster
from selfregulation.utils.plot_utils import beautify_legend, CurvedText, format_num, save_figure

ref_colors = {'survey': [sns.color_palette('Reds_d', 3)[i] for i in [0,2]], 
              'task': [sns.color_palette('Blues_d', 3)[i] for i in [0,2]]}

shortened_factors = get_short_names()

def visualize_importance(importance, ax, xticklabels=True, yticklabels=True, 
                         axes_linewidth=None, label_size=10,
                         label_scale=0, title=None, 
                         ymax=None, color='k'):
    importance_vars = importance[0]
    importance_vars = [shortened_factors.get(v,v) for v in importance_vars]
    if importance[1] is not None:
        importance_vals = [abs(i) for i in importance[1]]
        plot_loadings(ax, importance_vals, kind='line', offset=.5, 
                      colors=[color], plot_kws={'alpha': 1, 'linewidth': label_size/4})
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
            importance_vars = [str(i+1) for i in importance_vars]
        scale = 1+label_scale
        size = ax.get_position().expanded(scale, scale)
        ax2=ax.get_figure().add_axes(size,zorder=2)
        for i, var in enumerate(importance_vars):
            arc_start = (i+.15)*2*np.pi/len(importance_vars)
            arc_end = (i+.85)*2*np.pi/len(importance_vars)
            curve = [
                .85*np.cos(np.linspace(arc_start,arc_end,100)),
                .85*np.sin(np.linspace(arc_start,arc_end,100))
            ]  
            plt.plot(*curve, alpha=0)
            curvetext = CurvedText(
                x = curve[0][::-1],
                y = curve[1][::-1],
                text=var, #'this this is a very, very long text',
                va = 'bottom',
                axes = ax2,
                fontsize=label_size
            )
            ax2.set_xlim([-1,1]); ax2.set_ylim([-1,1])
            ax2.axis('off')
        
    if title:
        ax.set_title(title, fontsize=label_size*1.5, y=1.1)
    # set up yticks
    if len(importance[1]) != 0:
        ax.set_ylim(bottom=0)
        if ymax is None:
            ymax = np.max(importance_vals)*1.1
        ax.set_ylim(top=ymax)
        new_yticks = np.linspace(0, ymax, 7)
        ax.set_yticks(new_yticks)
        if yticklabels:
            labels = np.round(new_yticks,2)
            replace_dict = {i:'' for i in labels[::2]}
            labels = [replace_dict.get(i, i) for i in labels]
            ax.set_yticklabels(labels)


def plot_results_prediction(results, EFA=True, classifier='ridge',
                            rotate='oblimin', change=False,
                            size=4.6, ext='png', dpi=300, plot_dir=None,
                            **kwargs):
    predictions = results.load_prediction_object(EFA=EFA, 
                                                 change=change,
                                                 classifier=classifier,
                                                 rotate=rotate)
    if predictions is None:
        print('No prediction object found!')
        return
    else:
        predictions = predictions['data']
    shuffled_predictions = results.load_prediction_object(EFA=EFA, 
                                                          classifier=classifier, 
                                                          change=change,
                                                          rotate=rotate,
                                                          shuffle=True)['data']
    colors = ref_colors[results.ID.split('_')[0]]
    if plot_dir is not None:
        changestr = '_change' if change else ''
        if EFA:
            filename = 'EFA%s_%s_prediction_bar.%s' % (changestr, classifier, ext)
        else:
            filename = 'IDM%s_%s_prediction_bar.%s' % (changestr, classifier, ext)
        filename = path.join(plot_dir, filename)
    else:
        filename = None
    if EFA:
        EFA = results.EFA
    else:
        EFA = None
        
    plot_prediction(predictions, shuffled_predictions, colors, EFA=EFA,
                    size=size, filename=filename, dpi=dpi,
                    **kwargs)
    
    
def plot_prediction(predictions, comparison_predictions, 
                    colors=None, EFA=None, comparison_label=None,
                    target_order=None,  metric='R2', size=4.6,  
                    dpi=300, filename=None):
    if colors is None:
        colors = [sns.color_palette('Purples_d', 4)[i] for i in [1,3]]
    if comparison_label is None:
        comparison_label = '95% shuffled prediction'
    basefont = max(size, 5)
    sns.set_style('white')
    if target_order is None:
        target_order = predictions.keys()
    # get prediction success
    r2s = [[k,predictions[k]['scores_cv'][0][metric]] for k in target_order]
    insample_r2s = [[k, predictions[k]['scores_insample'][0][metric]] for k in target_order]
    # get shuffled values
    shuffled_r2s = []
    insample_shuffled_r2s = []
    for i, k in enumerate(target_order):
        # normalize r2s to significance
        R2s = [i[metric] for i in comparison_predictions[k]['scores_cv']]
        R2_95 = np.percentile(R2s, 95)
        shuffled_r2s.append((k,R2_95))
        # and insample
        R2s = [i[metric] for i in comparison_predictions[k]['scores_insample']]
        R2_95 = np.percentile(R2s, 95)
        insample_shuffled_r2s.append((k,R2_95))
        
    # convert nans to 0
    r2s = [(i, k) if k==k else (i,0) for i, k in r2s]
    insample_r2s = [(i, k) if k==k else (i,0) for i, k in insample_r2s]
    shuffled_r2s = [(i, k) if k==k else (i,0) for i, k in shuffled_r2s]
    
    # plot
    shuffled_grey = [.3,.3,.3]
    # plot variables
    figsize = (size, size*.75)
    fig = plt.figure(figsize=figsize)
    # plot bars
    ind = np.arange(len(r2s))
    width=.25
    ax1 = fig.add_axes([0,0,1,.5]) 
    ax1.bar(ind, [i[1] for i in r2s], width, 
            label='Cross-validated prediction', color=colors[0])
    ax1.bar(ind+width, [i[1] for i in insample_r2s], width, 
            label='Insample prediction', color=colors[1])
    # plot shuffled values above
    ax1.bar(ind, [i[1] for i in shuffled_r2s], width, 
             color='none', edgecolor=shuffled_grey, 
            linewidth=size/10, linestyle='--', label=comparison_label)
    ax1.bar(ind+width, [i[1] for i in insample_shuffled_r2s], width, 
            color='none', edgecolor=shuffled_grey, 
            linewidth=size/10, linestyle='--')
    
    ax1.set_xticks(np.arange(0,len(r2s))+width/2)
    ax1.set_xticklabels([i[0] for i in r2s], rotation=15, fontsize=basefont*1.4)
    ax1.tick_params(axis='y', labelsize=size*1.2)
    ax1.tick_params(length=size/4, width=size/10, pad=size/2, left=True, bottom=True)
    xlow, xhigh = ax1.get_xlim()
    if metric == 'R2':
        ax1.set_ylabel(r'$R^2$', fontsize=basefont*1.5, labelpad=size*1.5)
    else:
        ax1.set_ylabel(metric, fontsize=basefont*1.5, labelpad=size*1.5)
    # add a legend
    leg = ax1.legend(fontsize=basefont*1.4, loc='upper left', framealpha=1,
                     frameon=True, handlelength=0, handletextpad=0)
    leg.get_frame().set_linewidth(size/10)
    beautify_legend(leg, colors[:2]+[shuffled_grey])
    # change y extents
    ylim = ax1.get_ylim()
    r2_max = max(max(r2s, key=lambda x: x[1])[1],
                 max(insample_r2s, key=lambda x: x[1])[1])
    ymax = r2_max*1.5
    ax1.set_ylim(ylim[0], ymax)
    # change yticks
    if ymax<.15:
        ax1.set_ylim(ylim[0], .15)
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(.025))
    else:
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(.05))
        ax1.set_yticks(np.append([0, .025, .05, .075, .1, .125], np.arange(.15, .45, .05)))
    # draw grid
    ax1.set_axisbelow(True)
    plt.grid(axis='y', linestyle='dotted', linewidth=size/6)
    plt.setp(list(ax1.spines.values()), linewidth=size/10)
    # Plot Polar Plots for importances
    if EFA is not None:
        reorder_vec = EFA.get_factor_reorder(EFA.results['num_factors'])
        reorder_fun = lambda x: [x[i] for i in reorder_vec]
        # get importances
        vals = [predictions[i] for i in target_order]
        importances = [(reorder_fun(i['predvars']), 
                        reorder_fun(i['importances'][0])) for i in vals]
        # plot
        axes=[]
        N = len(importances)
        best_predictors = sorted(enumerate(r2s), key = lambda x: x[1][1])
        #if plot_heights is None:
        ylim = ax1.get_ylim(); yrange = np.sum(np.abs(ylim))
        zero_place = abs(ylim[0])/yrange
        plot_heights = [int(r2s[i][1]>0)
                        *(max(r2s[i][1],
                              insample_r2s[i][1],
                              shuffled_r2s[i][1],
                              insample_shuffled_r2s[i][1])/yrange)
                        for i, k in enumerate(target_order)]
        plot_heights = [(h+zero_place+.02)*.5 for h in plot_heights]
        # mask heights
        plot_heights = [plot_heights[i] if r2s[i][1]>max(shuffled_r2s[i][1],0) else np.nan
                        for i in range(len(plot_heights))]
        plot_x = (ax1.get_xticks()-xlow)/(xhigh-xlow)-(1/N/2)
        for i, importance in enumerate(importances):
            if pd.isnull(plot_heights[i]):
                continue
            axes.append(fig.add_axes([plot_x[i], plot_heights[i], 1/N,1/N], projection='polar'))
            color = colors[0]
            visualize_importance(importance, axes[-1],
                                 yticklabels=False, xticklabels=False,
                                 label_size=figsize[1]*1,
                                 color=color,
                                 axes_linewidth=size/10)
        # plot top 2 predictions, labeled  
        if best_predictors[-1][0] < best_predictors[-2][0]:
            locs = [.32, .68]
        else:
            locs = [.68, .32]
        label_importance = importances[best_predictors[-1][0]]
        # write abbreviation key
        pad = 0
        text = [(l, shortened_factors.get(l, None)) for l in label_importance[0]] # for abbeviations text
        if len([True for t in text if t[1] is not None]) > 0:
            pad = .05
            text_ax = fig.add_axes([.8,.56,.1,.34]) 
            text_ax.tick_params(labelleft=False, left=False, 
                                labelbottom=False, bottom=False)
            for spine in ['top','right','bottom','left']:
                text_ax.spines[spine].set_visible(False)
            for i, (val, abr) in enumerate(text):
                text_ax.text(0, i/len(text), abr+':', fontsize=size*1.2)
                text_ax.text(.5, i/len(text), val, fontsize=size*1.2)
                
        ratio = figsize[1]/figsize[0]
        axes.append(fig.add_axes([locs[0]-.2*ratio-pad,.56,.3*ratio,.3], projection='polar'))
        visualize_importance(label_importance, axes[-1], yticklabels=False,
                             xticklabels=True,
                             label_size=max(figsize[1]*1.5, 5),
                             label_scale=.22,
                             title=best_predictors[-1][1][0],
                             color=colors[0],
                             axes_linewidth=size/10)
        # 2nd top
        label_importance = importances[best_predictors[-2][0]]
        ratio = figsize[1]/figsize[0]
        axes.append(fig.add_axes([locs[1]-.2*ratio-pad,.56,.3*ratio,.3], projection='polar'))
        visualize_importance(label_importance, axes[-1], yticklabels=False,
                             xticklabels=True,
                             label_size=max(figsize[1]*1.5, 5),
                             label_scale=.22,
                             title=best_predictors[-2][1][0],
                             color=colors[0],
                             axes_linewidth=size/10)
    if filename is not None:
        save_figure(fig, filename, 
            {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()


def plot_prediction_scatter(results, target_order=None, EFA=True, change=False,
                            classifier='ridge', rotate='oblimin', 
                            normalize=False, metric='R2', size=4.6,  
                            dpi=300, ext='png', plot_dir=None):
    predictions = results.load_prediction_object(EFA=EFA, 
                                                 change=change,
                                                 classifier=classifier,
                                                 rotate=rotate)
    if predictions is None:
        print('No prediction object found!')
        return
    else:
        predictions = predictions['data']
    if EFA:
        predictors = results.EFA.get_scores()
    else:
        predictors = results.data
    if change:
        target_factors, _ = results.DA.get_change(results.dataset.replace('Complete', 'Retest'))
        predictors = predictors.loc[target_factors.index]
    else:
        target_factors = results.DA.get_scores()
    
    sns.set_style('whitegrid')
    n_cols = 2
    n_rows = math.ceil(len(target_factors.columns)/n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(size, size/n_cols*n_rows))
    axes = fig.get_axes()
    for i,v in enumerate(target_factors.columns):
        MAE = format_num(predictions[v]['scores_cv'][0]['MAE'])
        R2 = format_num(predictions[v]['scores_cv'][0]['R2'])
        axes[i].set_title('%s: R2: %s, MAE: %s' % (v, R2, MAE), 
            fontweight='bold', fontsize=size*1.5)
        clf=predictions[v]['clf']
        axes[i].scatter(target_factors[v], clf.predict(predictors), s=size*3)  
        axes[i].tick_params(length=0, labelsize=0)
        if i%2==0:
            axes[i].set_ylabel('Predicted Factor Score', fontsize=size*1.5)
    axes[i].set_xlabel('Target Factor Score', fontsize=size*1.5)
    axes[i-1].set_xlabel('Target Factor Score', fontsize=size*1.5)
    
    empty_plots = n_cols*n_rows - len(target_factors.columns)
    for ax in axes[-empty_plots:]:
        ax.set_visible(False)
    plt.subplots_adjust(hspace=.4, wspace=.3)
    
    if plot_dir is not None:
        changestr = '_change' if change else ''
        if EFA:
            filename = 'EFA%s_%s_prediction_scatter.%s' % (changestr, classifier, ext)
        else:
            filename = 'IDM%s_%s_prediction_scatter.%s' % (changestr, classifier, ext)
        save_figure(fig, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
        
def plot_prediction_comparison(results, size=4.6, change=False,
                               dpi=300, ext='png', plot_dir=None):
    colors = ref_colors[results.ID.split('_')[0]]
    R2s = {}
    for EFA in [False, True]:
        predictions = results.get_prediction_files(EFA=EFA, change=change, 
                                                   shuffle=False)
        predictions = sorted(predictions, key = path.getmtime)
        classifiers = np.unique([i.split('_')[-2] for i in predictions])
        # get last prediction file of each type
        for classifier in classifiers:
            filey = [i for i in predictions if classifier in i][-1]
            prediction_object = pickle.load(open(filey, 'rb'))['data']
            R2 = [i['scores_cv'][0]['R2'] for i in prediction_object.values()]
            R2 = np.nan_to_num(R2)
            feature = 'EFA' if EFA else 'IDM'
            R2s[feature+'_'+classifier] = R2
    if len(R2s) == 0:
        print('No prediction objects found')
        return
    R2s = pd.DataFrame(R2s).melt(var_name='Classifier', value_name='R2')
    R2s['Feature'], R2s['Classifier'] = R2s.Classifier.str.split('_', 1).str
    f = plt.figure(figsize=(size, size*.62))
    sns.barplot(x='Classifier', y='R2', data=R2s, hue='Feature',
                palette=colors[:2], errwidth=size/5)
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=size*1.8)
    ax.tick_params(axis='x', labelsize=size*1.8)
    leg = ax.legend(fontsize=size*2, loc='upper right')
    beautify_legend(leg, colors[:2])
    plt.xlabel('Classifier', fontsize=size*2.2, labelpad=size/2)
    plt.ylabel('R2', fontsize=size*2.2, labelpad=size/2)
    plt.title('Comparison of Prediction Methods', fontsize=size*2.5, y=1.05)
    
    if plot_dir is not None:
        filename = 'prediction_comparison.%s' % ext
        save_figure(f, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
    
def plot_prediction_relevance(results, EFA=True, classifier='ridge', 
                              rotate='oblimin', change=False, size=4.6, 
                              dpi=300, ext='png', plot_dir=None):
    """ Plots the relevant relevance of each factor for predicting all outcomes """
    predictions = results.load_prediction_object(EFA=EFA, 
                                                 change=change,
                                                 classifier=classifier,
                                                 rotate=rotate)['data']

    targets = list(predictions.keys())
    predictors = predictions[targets[0]]['predvars']
    importances = abs(np.vstack([predictions[k]['importances'] for k in targets]))
    # scale to 0-1 
    scaler = MinMaxScaler()
    scaled_importances = scaler.fit_transform(importances.T).T
    # make proportion
    scaled_importances = scaled_importances/np.expand_dims(scaled_importances.sum(1),1)
    # convert to dataframe
    scaled_df = pd.DataFrame(scaled_importances, index=targets, columns=predictors)
    melted = scaled_df.melt(var_name='Factor', value_name='Importance')
    plt.figure(figsize=(8,12))
    f=sns.boxplot(y='Factor', x='Importance',  data=melted,
                  width=.5)
    if plot_dir is not None:
        filename = 'prediction_relevance'
        save_figure(f, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()

def plot_outcome_ontological_similarity(results, EFA=True, classifier='ridge', 
                                        rotate='oblimin', change=False, size=4.6, 
                                        dpi=300, ext='png',  plot_dir=None):
    """ plots similarity of ontological fingerprints between outcomes """
    predictions = results.load_prediction_object(EFA=EFA, 
                                                 change=change,
                                                 classifier=classifier,
                                                 rotate=rotate)['data']

    targets = list(predictions.keys())
    predictors = predictions[targets[0]]['predvars']
    importances = np.vstack([predictions[k]['importances'] for k in targets])
    # convert to dataframe
    df = pd.DataFrame(importances, index=targets, columns=predictors)
    clustered = hierarchical_cluster(df, pdist_kws = {'metric': 'abscorrelation'})
    corr = 1-clustered['clustered_df']
    mask = np.zeros_like(corr)
    mask[np.tril_indices_from(mask, -1)] = True
    n = len(corr)
    # plot
    f = plt.figure(figsize=(size*5/4, size))
    ax1 = f.add_axes([0,0,.9,.9])
    cbar_ax = f.add_axes([.91, .05, .03, .8])
    sns.heatmap(corr, ax=ax1, square=True, vmax=1, vmin=0,
                cbar_ax=cbar_ax, linewidth=2,
                cmap=sns.light_palette((15, 75, 50), input='husl', n_colors=100, as_cmap=True))
    sns.heatmap(corr, ax=ax1, square=True, vmax=1, vmin=0,
                cbar_ax=cbar_ax, annot=True, annot_kws={"size": size/n*15},
                cmap=sns.light_palette((15, 75, 50), input='husl', n_colors=100, as_cmap=True),
                mask=mask, linewidth=2)
    yticklabels = ax1.get_yticklabels()
    ax1.set_yticklabels(yticklabels, rotation=0, ha="right")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    
    ax1.tick_params(labelsize=size*2)
    # format cbar
    cbar_ax.tick_params(axis='y', length=0)
    cbar_ax.tick_params(labelsize=size*2)
    cbar_ax.set_ylabel('Pearson Correlation', rotation=-90, labelpad=size*4, fontsize=size*3)
    if plot_dir is not None:
        filename = 'ontological_similarity.%s' % ext
        save_figure(f, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()

        
def plot_factor_fingerprint(results, classifier='ridge', rotate='oblimin', 
                            change=False, size=4.6,  
                            dpi=300, ext='png', plot_dir=None):
    colors = ref_colors[results.ID.split('_')[0]]
    reorder_vec = results.DA.get_factor_reorder(results.DA.results['num_factors'])
    targets = results.DA.get_loading().columns
    targets = [targets[i] for i in reorder_vec]
    if change:
        targets = [t+' Change' for t in targets]
        
    predictions = results.load_prediction_object(EFA=True, 
                                                 change=change,
                                                 classifier=classifier,
                                                 rotate=rotate)
    if predictions is None:
        print('No prediction object found!')
        return
    else:
        predictions = predictions['data']
    factors = predictions[targets[0]]['predvars']
    importances = np.vstack([predictions[k]['importances'] for k in targets])

    ncols = 3
    nrows = math.ceil(len(factors)/ncols)
    figsize = (size, size*nrows/ncols)
    f, axes = plt.subplots(nrows, ncols, figsize=figsize, 
                           subplot_kw={'projection':'polar'})
    plt.subplots_adjust(wspace=.5, hspace=.5)
    axes = f.get_axes()
    for i, factor in enumerate(factors):
        label_importance = [targets, importances[:,i]]
        visualize_importance(label_importance, axes[i], yticklabels=False,
                             xticklabels=True,
                             title=factor,
                             label_size=size*1.2,
                             label_scale=.2,
                             color=colors[0],
                             ymax=math.ceil(np.max(importances)*10)/10*1.1)
    
    if plot_dir is not None:
        changestr = '_change' if change else ''
        filename = 'EFA%s_%s_factor_fingerprint.%s' % (changestr, classifier, ext)

        save_figure(f, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
    
    
    