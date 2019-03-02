import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
import numpy as np
from os import path
import pandas as pd
from scipy.spatial.distance import  squareform
from sklearn.manifold import MDS
from sklearn.preprocessing import scale
import seaborn as sns
from dimensional_structure.plot_utils import get_var_color
from dimensional_structure.HCA_plots import abs_pdist
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.plot_utils import format_num
from selfregulation.utils.utils import get_info, get_recent_dataset

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', default=None)
parser.add_argument('-dpi', type=int, default=300)
parser.add_argument('-size', type=float, default=15)
parser.add_argument('-ext', default='pdf')
parser.add_argument('-plot_file', default=None)
parser.add_argument('-cluster_color', action='store_true')
args = parser.parse_args()

dataset = args.dataset

# get dataset of interest
basedir=get_info('base_directory')
if dataset == None:
    dataset = get_recent_dataset()
dataset = path.join(basedir,'Data',dataset)
datafile = dataset.split(path.sep)[-1]

# load data
results = load_results(datafile)
data = results['task'].data
out = results['task'].EFA.get_loading()
nfactors = out.shape[1]
task_subset = pd.concat([
    out.filter(regex='choice_reaction_time', axis=0),
    out.filter(regex='^stop_signal\.(hddm|SSRT)', axis=0)[1:5]])
task_subset_data = data.loc[:, task_subset.index]
task_variables = list(task_subset.index)

# plotting args
size = args.size
dpi = args.dpi
ext = args.ext
if args.plot_file is None:
    plot_file = path.dirname(results['task'].get_plot_dir())
else:
    plot_file = args.plot_file


# plot
f = plt.figure(figsize=(size, size))
basefont = size*1.3
basemarker = size**2*1.2
basewidth = size*.12
# how to color MDS
cluster_color = args.cluster_color

with sns.axes_style("white"):
    participant_ax1 = f.add_axes([.25,.555,.28,.16]) 
    participant_ax2 = f.add_axes([.25,.75,.28,.2]) 
    
    loading_ax1 = f.add_axes([.6,.545,.25,.1625]) 
    loading_ax2 = f.add_axes([.6,.743,.25,.193]) 
    
    participant_distance = f.add_axes([.3,.29,.16,.16]) 
    loading_distance = f.add_axes([.675,.29,.16,.16]) 
    #participant_distance.axis('off'); loading_distance.axis('off')
    participant_mds = f.add_axes([.25,-.02,.25,.25]) 
    loading_mds = f.add_axes([.625,-.02,.25,.25]) 
    # color bars for heatmaps
    cbar_ax = f.add_axes([.88,.595,.03,.3]) 
    cbar_ax2 = f.add_axes([.86,.31,.02,.12]) 
    # set background
    back = f.add_axes([0,0,1,1])
    back.axis('off')
    back.patch.set_alpha(0)
    back.set_xlim([0,1]); back.set_ylim([0,1])

tasks = sorted(np.unique([i.split('.')[0] for i in task_subset.index]))
participant_axes = [participant_ax1, participant_ax2]
loading_axes = [loading_ax1, loading_ax2]
for task_i in range(len(tasks)):
    tick_names = []; tick_colors = []
    # *************************************************************************
    # ***** plot participants on two tasks ***** 
    # *************************************************************************
    ax = participant_axes[task_i]
    plot_data = task_subset_data.filter(regex=tasks[task_i], axis=1)
    for i, (label, vals) in enumerate(plot_data.iteritems()):
        color = get_var_color(label)
        if 'drift' in label:
            name = 'drift rate'
        elif 'thresh' in label:
            name = 'threshold'
        elif 'non_decision' in label:
            name = 'non-decision'
        else:
            name = 'SSRT'
        tick_names.append(name)
        tick_colors.append(color)
        plot_vals = scale(vals[20:40].values)*.25+i*1.5
        # add mean line
        ax.hlines(i*1.5, 0, len(plot_vals)*.8, alpha=.6,
                  linestyle='--', color=color,
                  linewidth=basewidth)
        # plot values
        scatter_colors = [list(color)+[alpha] for alpha in np.linspace(1,0, len(plot_vals))]
        ax.scatter(range(len(plot_vals)), plot_vals, color=scatter_colors,
                   s=basemarker*.23, edgecolors='none')
        ax.grid(False)
    # make x ticks invisible
    ax.set_xticklabels('')
    ax.tick_params(axis='both', length=0)
    # remove splines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # add tick labels
    ax.set_yticks([x*1.5 for x in range(len(tick_names))])
    ax.set_yticklabels(tick_names, fontsize=basefont)
    # change tick color
    tick_colors = tick_colors
    [t.set_color(i) for (i,t) in
         zip(tick_colors,ax.yaxis.get_ticklabels())]
    
    # *************************************************************************
    # ***** plot loading ***** 
    # *************************************************************************
    max_val = round(abs(task_subset).max().max(),1)
    loading_data = task_subset.filter(regex=tasks[task_i], axis=0)
    # for visualization purposes remove "reflections" from loading matrix
    # by multiplying by -1
    reflects = [-1 if 'ReflogTr' in i else 1 for i in loading_data.index]
    loading_data = loading_data.multiply(reflects, axis=0)
    # plot loadings
    sns.heatmap(loading_data.iloc[::-1,:], ax=loading_axes[task_i], 
                yticklabels=False, xticklabels=False,
                linecolor='white', linewidth=basewidth,
                cbar_ax=cbar_ax, vmax =  max_val, vmin = -max_val,
                cbar_kws={'ticks': [-max_val, 0, max_val]},
                cmap=sns.diverging_palette(220,16,n=100, as_cmap=True))
    # format cbar
    cbar_ax.set_yticklabels([format_num(-max_val, 1), 0, format_num(max_val, 1)])
    cbar_ax.tick_params(axis='y', length=0)
    cbar_ax.tick_params(labelsize=basefont)
    for i in range(1,loading_data.shape[0]+1):
        #loading_axes[task_i].hlines(i, -.2, 6.1, color='white', linewidth=basewidth*3)
        loading_axes[task_i].add_patch(Rectangle([-.1,i-.2], 
                    width=loading_data.shape[1]+.2, height=.2, zorder=100,
                    facecolor='white', edgecolor='white', 
                    linewidth=basewidth, clip_on=False))
    # add boxes
    for i in range(len(tick_names)):
        box_color = tick_colors[len(tick_names)-(i+1)]
        box_pos = [-.15, i+.2]
        loading_axes[task_i].add_patch(Rectangle(box_pos, 
                    width=.15, height=.4, zorder=100,
                    facecolor=box_color, edgecolor=box_color, 
                    linewidth=basewidth, clip_on=False))
        loading_axes[task_i].hlines(i+.4, -2, -.5, color=box_color, 
                    clip_on=False, linewidth=basewidth, linestyle=':')
        

        
# ****************************************************************************
# Distance Matrices
# ****************************************************************************
participant_distances = squareform(abs_pdist(data.T))
participant_distances = results['task'].HCA.results['data']['clustered_df']
loading_distances = results['task'].HCA.results['EFA5_oblimin']['clustered_df']
sns.heatmap(participant_distances, ax=participant_distance,
            cmap=ListedColormap(sns.color_palette('gray', n_colors=100)),
            xticklabels=False, yticklabels=False, square=True, cbar=False, linewidth=0)
sns.heatmap(loading_distances, ax=loading_distance,
            xticklabels=False, yticklabels=False, square=True, 
            cmap=ListedColormap(sns.color_palette('gray', n_colors=100)),
            cbar_kws={'ticks': [0, .99]}, cbar_ax=cbar_ax2, linewidth=0)
participant_distance.set_ylabel('DV', fontsize=basefont)
loading_distance.set_ylabel('DV', fontsize=basefont)
participant_distance.set_title(r'$\vec{DV}\in \mathrm{\mathbb{R}}^{522}$', fontsize=basefont)
loading_distance.set_title(r'$\vec{DV}\in \mathrm{\mathbb{R}}^{5}$', fontsize=basefont)

# plot location of top variables
# update limits
lim = list(participant_distance.get_xlim())
for label in task_variables:
    if 'drift' in label:
        name = 'drift rate'
    elif 'thresh' in label:
        name = 'threshold'
    elif 'non_decision' in label:
        name = 'non-decision'
    else:
        name = 'SSRT'
    line_color = get_var_color(label)
    var_index = np.where(participant_distances.index==label)[0][0]
    # plot pretty colors
    x_index = var_index
    y_index = var_index
    if participant_distance.get_ylim()[0]==0:
        y_index = len(participant_distances)-y_index
    participant_distance.plot([x_index, x_index], 
                              lim,
                              color=line_color,
                              linewidth=basewidth)
    participant_distance.plot(lim,
                              [y_index, y_index],
                              color=line_color,
                              linewidth=basewidth)
    
    var_index = np.where(loading_distances.index==label)[0][0]
    x_index = var_index
    y_index = var_index
    if participant_distance.get_ylim()[0]==0:
        y_index = len(participant_distances)-y_index
    loading_distance.plot([x_index, x_index], 
                          lim,
                          color=line_color,
                          linewidth=basewidth)
    loading_distance.plot(lim, 
                          [y_index, y_index],
                          color=line_color,
                          linewidth=basewidth)


# format cbar
cbar_ax2.set_yticklabels([0, 1])
cbar_ax2.tick_params(axis='y', length=0)
cbar_ax2.tick_params(labelsize=basefont*.75)
# ****************************************************************************
# MDS Plots
# ****************************************************************************
if not cluster_color: # color based on drift/thresh/non-decision
    mds_colors = np.array([[.5, .5, .5, .3]]*loading_distances.shape[0])
    interest_index = []
    misc_index = []
    for i, label in enumerate(loading_distances.index):
        if '.hddm_drift' in label:
            name = 'drift rate'
        elif '.hddm_thresh' in label:
            name = 'threshold'
        elif '.hddm_non_decision' in label:
            name = 'non-decision'
        elif 'SSRT' in label:
            name = 'SSRT'
        else:
            misc_index.append(i)
            continue
        interest_index.append(i)
        mds_colors[i] = get_var_color(label) + [1]
    mds_index = misc_index + interest_index
else:
    reorder = results['task'].HCA.results['EFA5_oblimin']['reorder_vec']
    labels = results['task'].HCA.results['EFA5_oblimin']['labels'][reorder]
    palette= sns.hls_palette(n_colors = max(labels))
    mds_colors = np.array([palette[i-1] for i in labels])
    mds_index = range(len(mds_colors))
    
# plot raw MDS
np.random.seed(2000)
mds = MDS(dissimilarity='precomputed')
mds_out = mds.fit_transform(participant_distances)
participant_mds.scatter(mds_out[mds_index,0], mds_out[mds_index,1], 
            s=basemarker,
            marker='h',
            facecolors=mds_colors[mds_index],
            edgecolors='white',
            linewidths=basewidth/2)
participant_mds.set_xticklabels(''); participant_mds.set_yticklabels('')
participant_mds.tick_params(axis='both', length=0)
participant_mds.axis('off')
# plot loading MDS
mds = MDS(dissimilarity='precomputed')
mds_out = mds.fit_transform(loading_distances)
loading_mds.scatter(mds_out[mds_index,0], mds_out[mds_index,1], 
            s=basemarker,
            marker='h',
            facecolors=mds_colors[mds_index],
            edgecolors='white',
            linewidths=basewidth/2)
loading_mds.set_xticklabels(''); loading_mds.set_yticklabels('')
loading_mds.tick_params(axis='both', length=0)
loading_mds.axis('off'); 


"""
# get example points
for ax, distances in [(loading_mds, loading_distances), 
                      (participant_mds, participant_distances)]:
    var_locs = []
    subplot_colors=[]
    for label in task_subset.index:
        var_color = get_var_color(label)
        index = np.where(distances.index==label)[0][0]
        var_loc = mds_out[index]
        var_locs.append((label, var_loc))
        subplot_colors.append(var_color)
    
    width = sum(np.abs(list(ax.get_xlim())))
    height = sum(np.abs(list(ax.get_ylim())))
    ax.scatter([v[1][0] for v in var_locs],
                        [v[1][1] for v in var_locs],
                        edgecolors='white',
                        facecolors=subplot_colors,
                        marker='h',
                        s=basemarker)

    ax.scatter([v[1][0] for v in var_locs],
                        [v[1][1] for v in var_locs],
                        edgecolors='white',
                        facecolors='white',
                        marker='.',
                        s=basemarker*.4)
"""
# ****************************************************************************
# Text and additional pretty lines
# ****************************************************************************
# label 
back.text(-.03, .96, 'Measure', horizontalalignment='center', fontsize=basefont,
          fontweight='bold')
back.text(.21, .94, 'DV', horizontalalignment='center', fontsize=basefont,
          fontweight='bold')
# task labels
#back.text(.05, .75, 'Measure', horizontalalignment='center', 
#          verticalalignment='center',
#          fontsize=basefont*1.56250,
 #         fontweight='bold', rotation=90)
back.text(-.03,.62, 'Choice RT', fontsize=basefont, rotation=0, 
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.87, 'Stop Signal', fontsize=basefont, rotation=0, 
              horizontalalignment='center', verticalalignment='center')
# other tasks
alpha=.3
back.text(-.03,.6, 'Bickel Titrator', fontsize=basefont, rotation=0, alpha=alpha,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.58, 'ART', fontsize=basefont, rotation=0, alpha=alpha,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.56, 'ANT', fontsize=basefont, rotation=0, alpha=alpha,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.54, 'Adaptive N-Back', fontsize=basefont, rotation=0, alpha=alpha,
              horizontalalignment='center', verticalalignment='center')

back.text(-.03,.64, 'CCT-Cold', fontsize=basefont, rotation=0, alpha=alpha,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.66, 'CCT-Hot', fontsize=basefont, rotation=0, alpha=alpha,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.68, 'Dietary Decision', fontsize=basefont, rotation=0, alpha=alpha,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.7, 'Digit Span', fontsize=basefont, rotation=0, alpha=alpha,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.72, 'Directed Forgetting', fontsize=basefont, rotation=0, alpha=alpha,
              horizontalalignment='center', verticalalignment='center')

back.text(-.03,.758, '...', fontsize=basefont*3, rotation=0, alpha=alpha,
              horizontalalignment='center', verticalalignment='center')

back.text(-.03,.89, 'Stroop', fontsize=basefont, rotation=0, alpha=alpha,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.91, 'Three-By-Two', fontsize=basefont, rotation=0, alpha=alpha,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.93, 'Tower of London', fontsize=basefont, rotation=0, alpha=alpha,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.85, 'Stim-Selective SS', fontsize=basefont, rotation=0, alpha=alpha,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.83, 'Spatial Span', fontsize=basefont, rotation=0, alpha=alpha,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.81, 'Simple RT', fontsize=basefont, rotation=0, alpha=alpha,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.79, 'Simon', fontsize=basefont, rotation=0, alpha=alpha,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.77, 'Shift Task', fontsize=basefont, rotation=0, alpha=alpha,
              horizontalalignment='center', verticalalignment='center')
# add labels 
cbar_ax.tick_params('y', which='major', pad=basefont*.5)
cbar_ax.set_ylabel('Factor Loading', rotation=-90, fontsize=basefont, labelpad=basefont)
cbar_ax2.tick_params('y', which='major', pad=basefont*.5)
cbar_ax2.set_ylabel('Distance', rotation=-90, fontsize=basefont, labelpad=basefont)
back.text(.375, .535, 'Participants (n=522)', fontsize=basefont, horizontalalignment='center')

# loading ticks
loading_ax2.tick_params('x', length=basewidth*2, width=basewidth, which='major', pad=basefont*.5)
loading_ax2.xaxis.set_ticks_position('top')
loading_ax2.set_xticks(np.arange(.5,5.5,1))
loading_ax2.set_xticklabels(['Factor %s' % i for i in range(1,nfactors+1)],
                            rotation=45, ha='left', fontsize=basefont)
# participant box
back.add_patch(Rectangle((.3385,.55), width=.0115, height=.4, 
                         facecolor="none", edgecolor='grey', linewidth=basewidth*.75))
back.text(.3385, .96, 'One Participant', fontsize=basefont, 
          horizontalalignment='center', color='grey')


# legend for mds
back.text(.13, .18, 'All Task DVs (%s)' % loading_distances.shape[0], fontsize=basefont, 
          horizontalalignment='center')
back.text(.13, .15, 'Threshold', fontsize=basefont, 
          horizontalalignment='center', color=get_var_color('thresh'))
back.text(.13, .125, 'Non-Decision', fontsize=basefont, 
          horizontalalignment='center', color=get_var_color('non_decision'))
back.text(.13, .1, 'Drift Rate', fontsize=basefont, 
          horizontalalignment='center', color=get_var_color('drift'))
back.text(.13, .075, 'SSRT', fontsize=basefont, 
          horizontalalignment='center', color=get_var_color('SSRT'))
back.text(.13, .05, 'Other', fontsize=basefont, 
          horizontalalignment='center', color='grey')
#back.text(.13, .025, 'o indicates example DVs', fontsize=basefont*.75, 
#          horizontalalignment='center', color='k')
#back.text(.13, .01, 'from preceeding plots', fontsize=basefont*.75, 
#          horizontalalignment='center', color='k')
# add connecting lines between participants and loading
back.vlines(.565, .3, .42, alpha=.4, linestyle='-', linewidth=basewidth)
back.vlines(.565, .05, .2, alpha=.4, linestyle='-', linewidth=basewidth)

# arrows
# from tasks to DVs
arrowcolor = [.5,.5,.5]
back.arrow(.03,.62,.1,.06, width=basewidth/1000, color=arrowcolor)
back.arrow(.03,.62,.07,.011, width=basewidth/1000, color=arrowcolor)
back.arrow(.03,.62,.1,-.04, width=basewidth/1000, color=arrowcolor)

back.arrow(.05,.87,.08,.045, width=basewidth/1000, color=arrowcolor)
back.arrow(.05,.87,.06,-.005, width=basewidth/1000, color=arrowcolor)
back.arrow(.05,.87,.08,-.045, width=basewidth/1000, color=arrowcolor)
back.arrow(.05,.87,.1,-.075, width=basewidth/1000, color=arrowcolor)

# from participant to EFA
back.arrow(.5, .725, .05, 0, width=basewidth/200, head_width=basewidth/75, 
           facecolor='k')
back.text(.52, .735, 'EFA', fontsize=basefont, 
          horizontalalignment='center')
# from data to heatmap
back.arrow(.375, .514, 0, -.01, width=basewidth/250, head_width=basewidth/125,
           edgecolor='k', facecolor='white')
back.arrow(.75, .514, 0, -.01, width=basewidth/250, head_width=basewidth/125,
           facecolor='k')
back.text(.567, .48, 'Pairwise Distance', fontsize=basefont, 
          horizontalalignment='center')
back.text(.567, .46, 'Between DVs', fontsize=basefont, 
          horizontalalignment='center')
back.text(.567, .435, r'$1-\vert r_{DV_1,DV_2} \vert$', fontsize=basefont, 
          horizontalalignment='center')
# from heatmap to MDS
back.arrow(.375, .27, 0, -.01, width=basewidth/250, 
           head_width=basewidth/125, edgecolor='k', facecolor='white')
back.arrow(.75, .27, 0, -.01, width=basewidth/250, 
           head_width=basewidth/125, facecolor='k')
back.text(.567, .24, 'Multidimensional Scaling', fontsize=basefont, 
          horizontalalignment='center')

# figure labels
back.text(-.13, 1, 'A', fontsize=basefont*1.56255, fontweight='bold')
back.text(.12, 1, 'B', fontsize=basefont*1.56255, fontweight='bold')
back.text(.62, 1, 'C', fontsize=basefont*1.56255, fontweight='bold')
back.text(.25, .49, 'D', fontsize=basefont*1.56255, fontweight='bold')
back.text(.85, .49, 'E', fontsize=basefont*1.56255, fontweight='bold')
back.text(.25, .24, 'F', fontsize=basefont*1.56255, fontweight='bold')
back.text(.85, .24, 'G', fontsize=basefont*1.56255, fontweight='bold')


# save
f.savefig(path.join(plot_file, 'analysis_overview.%s' % ext), 
                bbox_inches='tight', 
                dpi=dpi)
