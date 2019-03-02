import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import array as scipyarray
import seaborn as sns

from dimensional_structure.utils import reorder_data
from selfregulation.utils.plot_utils import CurvedText, beautify_legend



def get_short_names():
    shortened_factors = {
        'Sensation Seeking': 'SS',
        'Mindfulness': 'MND',
        'Emotional Control': 'EMC',
        'Impulsivity': 'IMP',
        'Goal-Directedness': 'GD',
        'Reward Sensitivity': 'RS',
        'Risk Perception': 'RP',
        'Eating Control': 'EC',
        'Ethical Risk-Taking': 'ERT',
        'Social Risk-Taking': 'SRT',
        'Financial Risk-Taking': 'FRT',
        'Agreeableness': 'AGR',
        'Unsafe Drinking': 'UD',
        'Binge Drinking': 'BD',
        'Problem Drinking': 'PD',
        'Income / Life Milestones': 'ILM',
        'Drug Use': 'DU',
        'Daily Smoking': 'DS',
        'Lifetime Smoking': 'LS',
        'Mental Health': 'MH',
        'Obesity': 'OBS'
        }
    return shortened_factors

def get_var_group(label):
    groups = ['Drift Rate', 'Threshold', 'Non-Decision', 'SSRT',
              'Discounting', 'Memory', 'Learning']
    group = {'drift': 0,
            'thresh': 1, 
            'non_decision': 2,
            'SSRT': 3,
            'discount': 4,
            'kirby.percent_patient': 4,
            'mean_load': 5,
            'span': 5,
            'keep_track': 5,
            'two_stage_decision': 6,
            'shift_task': 6,
            'hierarchical_rule': 6
            }
    keys = group.keys()
    key = [k for k in keys if k in label]
    if len(key)==0:
        return (7, None)
    else:
        assert len(key)==1
        group_index = group[key[0]]
        return group_index, groups[group_index]

def get_var_color(label):
    colors = [[.8, 0.62, 0.0], 
              [0.0, 0.5, 0.0], 
              [0.0, 0.7, 0.7], 
              [0.75, 0.0, 0.75],
              [0, .25, .7],
              [.8, .25, 0],
              [.4, 0, 0]]
    group, group_name = get_var_group(label)
    if group_name is None:
        return [.5, .5, .5]
    else:
        return colors[group]
    
    

# ****************************************************************************
# helper functions for hierarchical plotting
# ****************************************************************************

def plot_tree(tree, pos=None, ax=None, linewidth=3):
    """ Plots a subset of a dendrogram 
    
    Args:
        tree: output of dendrogram function
        pos: which positions from the dendrogram to plot. Doesn't correspond to
        leaves
    """
    
    icoord = scipyarray( tree['icoord'] )
    dcoord = scipyarray( tree['dcoord'] )
    color_list = scipyarray( tree['color_list'] )
    if pos:
        icoord = icoord[pos]
        dcoord = dcoord[pos]
        color_list = color_list[pos]
    if ax is None:
        f, ax = plt.subplots(1,1)
    for xs, ys, color in zip(icoord, dcoord, color_list):
        ax.plot(xs, ys,  color, linewidth=linewidth)
    
    
# ****************************************************************************
# Helper functions for visualization of component loadings
# ****************************************************************************

def plot_loadings(ax, component_loadings, groups=None, colors=None, 
                  width_scale=1, offset=0, kind='bar', plot_kws=None):
    """Plot component loadings
    
    Args:
        ax: axis to plot on. If a polar axist, a polar bar plot will be created.
            Otherwise, a histogram will be plotted
        component_loadings (array or pandas Series): loadings to plot
        groups (list, optional): ordered list of tuples of the form: 
            [(group_name, list of group members), ...]. If not supplied, all
            elements will be treated as one group
        colors (list, optional): if supplied, specifies the colors for the groups
        width_scale (float): scale of bars. Default is 1, which fills the entire
            plot
        offset (float): offset as a proportion of width. Used to plot multiple
            columns side by side under one factor
        bar_kws (dict): keywords to pass to ax.bar
    """
    if plot_kws is None:
        plot_kws = {}
    N = len(component_loadings)
    if groups is None:
        groups = [('all', [0]*N)]
    if colors is not None:
        assert(len(colors) == len(groups))
    else:
        colors = sns.hls_palette(len(groups), l=.5, s=.8)
    ax.set_xticklabels([''])
    ax.set_yticklabels([''])
    
    width = np.pi/(N/2)*width_scale*np.ones(N)
    theta = np.array([2*np.pi/N*i for i in range(N)]) + width[0]*offset
    radii = component_loadings
    if kind == 'bar':
        bars = ax.bar(theta, radii, width=width, bottom=0.0, **plot_kws)
        for i,r,bar in zip(range(N),radii, bars):
            color_index = sum((np.cumsum([len(g[1]) for g in groups])<i+1))
            bar.set_facecolor(colors[color_index])
    elif kind == 'line':
        if 'linewidth' not in plot_kws.keys():
            plot_kws['linewidth'] = 5
        theta = np.append(theta, theta[0])
        radii = np.append(radii, radii[0])
        lines = ax.plot(theta, radii, color=colors[0], **plot_kws)
    return colors
        
def create_categorical_legend(labels,colors, ax):
    """Take a list of labels and colors and creates a legend"""
    import matplotlib
    def create_proxy(color):
        line = matplotlib.lines.Line2D([0], [0], linestyle='none',
                    mec='none', marker='o', color=color)
        return line
    proxies = [create_proxy(item) for item in colors]
    ncol = max(len(proxies)//6, 1)
    ax.legend(proxies, labels, numpoints=1, markerscale=2.5, ncol=ncol,
              bbox_to_anchor=(1, .95), prop={'size':20})

def visualize_factors(loading_df, groups=None, n_rows=2, 
                      legend=True, input_axes=None, colors=None):
    """
    Takes in a dataset to run EFA on, and a list of groups in the form
    of a list of tuples. Each element of this list should be of the form
    ("group name", [list of variables]). Each element of the list should
    be mututally exclusive. These groups are used for coloring the plots
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    loading_df = loading_df.select_dtypes(include=numerics)
    if groups:
        loading_df = reorder_data(loading_df, groups, axis=0)
            
    n_components = loading_df.shape[1]
    n_cols = int(np.ceil(n_components/n_rows))
    sns.set_style("white")
    if input_axes is None:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*5,n_rows*5),
                           subplot_kw={'projection': 'polar'})
        axes = fig.get_axes()
        fig.tight_layout()
    else:
        axes = input_axes
    for i in range(n_components):
        component_loadings = loading_df.iloc[:,i]
        colors = plot_loadings(axes[i], abs(component_loadings), groups,
                               colors=colors)
    for j in range(n_components, len(axes)):
        axes[j].set_visible(False)
    if legend and groups is not None:
        create_categorical_legend([g[0] for g in groups], 
                                  colors, axes[n_components-1])
    if input_axes is None:
        return fig

def visualize_task_factors(task_loadings, ax, xticklabels=True, label_size=12,
                           yticklabels=False, pad=0, ymax=None, legend=True):
    """Plot task loadings on one axis"""
    n_measures = len(task_loadings)
    colors = sns.hls_palette(len(task_loadings), l=.4, s=.8)
    for i, (name, DV) in enumerate(task_loadings.iterrows()):
        plot_loadings(ax, abs(DV)+pad, width_scale=1/(n_measures), 
                      colors = [colors[i]], offset=i+.5,
                      kind='line',
                      plot_kws={'label': name, 'alpha': .8})
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
    # set up x ticks
    xtick_locs = np.arange(0.0, 2*np.pi, 2*np.pi/len(DV))
    ax.set_xticks(xtick_locs)
    ax.set_xticks(xtick_locs+np.pi/len(DV), minor=True)
    if xticklabels:
        labels = task_loadings.columns
        if type(labels[0]) != str:
            labels = ['Fac %s' % str(i) for i in labels]
        scale = 1.2
        size = ax.get_position().expanded(scale, scale)
        ax2=ax.get_figure().add_axes(size,zorder=2)
        max_var_length = max([len(v) for v in labels])
        for i, var in enumerate(labels):
            offset=.3*25/len(labels)**2
            start = (i-offset)*2*np.pi/len(labels)
            end = (i+(1-offset))*2*np.pi/len(labels)
            curve = [
                np.cos(np.linspace(start,end,100)),
                np.sin(np.linspace(start,end,100))
            ]  
            plt.plot(*curve, alpha=0)
            # pad strings to longest length
            num_spaces = (max_var_length-len(var))
            var = ' '*(num_spaces//2) + var + ' '*(num_spaces-num_spaces//2)
            curvetext = CurvedText(
                x = curve[0][::-1],
                y = curve[1][::-1],
                text=var, #'this this is a very, very long text',
                va = 'top',
                axes = ax2,
                fontsize=label_size##calls ax.add_artist in __init__
            )
            ax2.axis('off')
    if legend:
        leg = ax.legend(loc='upper center', bbox_to_anchor=(.5,-.15), frameon=False)
        beautify_legend(leg, colors[:len(task_loadings)])
        
def plot_factor_tree(factor_tree, groups=None, filename=None):
    """
    Runs "visualize_factors" at multiple dimensionalities and saves them
    to a pdf
    data: dataframe to run EFA on at multiple dimensionalities
    groups: group list to be passed to visualize factors
    filename: filename to save pdf
    component_range: limits of EFA dimensionalities. e.g. (1,5) will run
                     EFA with 1 component, 2 components... 5 components.
    reorder_list: optional. List of index values in an order that will be used
                  to rearrange data
    """
    max_c = np.max(list(factor_tree.keys()))
    min_c = np.min(list(factor_tree.keys()))
    n_cols = max_c
    n_rows = max_c-min_c+1
    f,axes = plt.subplots(n_rows, n_cols, subplot_kw=dict(projection='polar'),
               figsize=(n_cols*5,n_rows*5))
    f.tight_layout()
    # move axes:
    for i,row in enumerate(axes):
        for j,ax in enumerate(row):
            pos1 = ax.get_position() # get the original position
            pos2 = [pos1.x0 + (n_rows-i-1)*pos1.width/1.76, pos1.y0,  pos1.width, pos1.height] 
            ax.set_position(pos2) # set a new position

    # plot
    for rowi, c in enumerate(range(min_c,max_c+1)):
        tmp_loading_df = factor_tree[c]
        if rowi == 0:
            visualize_factors(tmp_loading_df, groups, 
                              n_rows=1, input_axes=axes[rowi,0:c], legend=True)
        else:
            visualize_factors(tmp_loading_df, groups, 
                              n_rows=1, input_axes=axes[rowi,0:c], legend=False)
        for ax in axes[rowi,c:]:
            ax.set_axis_off()
    if filename:
        save_figure(f, filename, {'bbox_inches': 'tight'})
    else:
        return f