import math
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib import text as mtext
import numpy as np
import os
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
import seaborn as sns

# basic helper functions
def format_num(num, digits=2):
    def format(num):
        if num%1 != 0:
            formatted = ("{0:0." + str(digits) + "f}").format(num)
        else:
            formatted =  ("{0:0." + str(0) + "f}").format(num)
        return formatted
    try:
        return [format(i) for i in num]
    except TypeError:
        return format(num)

def format_variable_names(variables):
    """ formats a list of variable names """
    # convert non_decision
    new_vars = []
    for var in variables:
        var = var.replace('non_decision', 'non-decision')
        var = var.replace('hddm_', 'DDM-')
        var = var.replace('_cost', '-cost')
        var = var.replace('_sensitivity', '-sensitivity')
        var = var.replace('two_stage', 'two_step')
        var = var.replace('threebytwo', 'cue/task switch')
        var = var.replace('.', ': ')
        var = ' '.join(var.split('_'))
        new_vars.append(var)
    return new_vars

# basic plotting helper functions

def place_letter(ax, letter, fontsize=14, yoffset=.02, xoffset=0):
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    
    ax.text(xlim[0]-(xlim[1]-xlim[0])*xoffset, 
            ylim[1]+(ylim[1]-ylim[0])*yoffset, 
            letter, 
            fontsize=fontsize, fontweight='bold')
    
def beautify_legend(legend, colors=None, fontsize=None):
    if colors is None:
        colors = [i.get_color() for i in legend.legendHandles]
    for i, text in enumerate(legend.get_texts()):
        text.set_color(colors[i])
    for item in legend.legendHandles:
        item.set_visible(False)
    for item in legend.legendHandles:
        item.set_visible(False)
    if fontsize:
        plt.setp(legend.get_texts(), fontsize=fontsize)
    
#***************************************************
# ********* Plotting Functions **********************
#**************************************************

def DDM_plot(v,t,a, sigma = .1, n = 10, plot_n = 15, 
             subsample=1, percentiles=None, only_positive=False,
             plot_distributions=True, plot_avg=False, file = None):
    """ Make a plot of trajectories using ddm parameters (in seconds)
    
    """
    # generate trajectory
    v = v/1000
    t =  t*1000
    timesteps = np.arange(2000)
    avg_trajectory = [0]*int(t) + [i*v for i in range(len(timesteps)-int(t))]
    crosspoint = np.where(np.array(avg_trajectory)>a)[0][0]
    avg_trajectory = avg_trajectory[:crosspoint]
    trajectories = []
    while len(trajectories) < n:
        y = [0]
        for step in timesteps[1:]:
            if step < t:
                y += [0]
            else:
                y += [y[-1]+v+np.random.normal(0,sigma)]
            if y[-1] > a:
                trajectories.append((y,'correct'))
                break
            elif y[-1] < -a:
                trajectories.append((y,'fail'))
                break
    # sort trajectories by length
    trajectories = sorted(trajectories, key = lambda x: len(x[0]))
    positive_trajectories = [t[0] for t in trajectories if t[1] == 'correct']
    negative_trajectories = [t[0] for t in trajectories if t[1] == 'fail']
    correct_rts = [len(t) for t in positive_trajectories]
    incorrect_rts = [len(t) for t in negative_trajectories]
    p_correct = len(positive_trajectories)/len(trajectories)
    plot_posn = int(plot_n*p_correct)
    # rts
    if percentiles is not None:
        plot_posi = [int(len(positive_trajectories)*i) for i in percentiles]
        plot_negi = [int(len(negative_trajectories)*i) for i in percentiles]
    else:
        plot_posi = np.linspace(0,len(positive_trajectories)-1,plot_posn, dtype=int)
        plot_negi = np.linspace(0,len(negative_trajectories)-1,plot_n-plot_posn, dtype=int)
    if only_positive:
        plot_negi = []
    # plot
    sns.set_context('talk')
    plot_start = int(max(0,t-50))
    fig = plt.figure(figsize = [10,6])
    ax = fig.add_axes([0,.2,1,.6]) 
    ax.set_xticklabels([])
    max_y = 0
    for y in [positive_trajectories[i] for i in plot_posi]:
        plt.plot(timesteps[plot_start:len(y):subsample],y[plot_start::subsample], 
                           c = 'green', alpha=.8, lw=2)
        if len(y) > max_y:
            max_y = len(y)
    for y in [negative_trajectories[i] for i in plot_negi]:
        plt.plot(timesteps[plot_start:len(y):subsample],y[plot_start::subsample], 
                           c = 'red', alpha=.8, lw=2)
        if len(y) > max_y:
            max_y = len(y)
    plt.xlim([plot_start,max_y+50])
    plt.ylim([-a*1.01,a*1.01])
    plt.yticks([-a, 0, a], [-a, 0, a])
    ax.tick_params(axis='y', labelsize=20)
    plt.ylabel('Decision Variable', fontsize = 24)
    # plot average trajectory
    if plot_avg:
        plt.plot(timesteps[plot_start:len(avg_trajectory)], avg_trajectory[plot_start:],
                 linewidth=5, color='b', linestyle='--')
    axes = [ax]
    if plot_distributions:
        with sns.axes_style("white"):
            # plot correct responses
            ax2 = fig.add_axes([0,.8,1,.2]) 
            sns.kdeplot(pd.Series(correct_rts), color = 'g', ax = ax2, shade = True)
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            ax2.text(np.sum(ax2.get_xlim())*.9, ax2.get_ylim()[1]*.5, 
                     'Correct RT distribution', color='g',
                     horizontalalignment='center')
            # plot incorrect responses
            ax3 = fig.add_axes([0,.2*p_correct,1,.2*(1-p_correct)])
            ax3.set_xlabel('Time Step (ms)', fontsize = 24, labelpad=50)
            if len(incorrect_rts) > 0:
                sns.kdeplot(pd.Series(incorrect_rts), color = 'r', ax = ax3, shade = True)
                ax3.set_yticklabels([])
                ax.tick_params(axis='x', pad=40)
            ax3.set_ylim(0, ax3.get_ylim()[1])
            ax3.set_xticklabels([])
            ax3.text(np.sum(ax3.get_xlim())*.9, ax3.get_ylim()[1]*1.2, 
                     'Incorrect RT distribution', color='r',
                     horizontalalignment='center')
            ax3.invert_yaxis()
            axes+= [ax2, ax3]
                
    # normalize xmax based on longest trajectory
    xmax = len(trajectories[-1][0])+t*2
    for ai in axes:
        ai.set_xlim(right=xmax)
        ai.spines['right'].set_visible(False)
        ai.spines['top'].set_visible(False)
        ai.spines['bottom'].set_visible(False)
    # add dashed lines
    ax.hlines([a,-a],0,xmax,linestyles = 'dashed')
    ax.set_xticks(np.arange(*ax.get_xlim(), 200))
    ax.set_xticklabels([format_num(i,0) for i in np.arange(*ax.get_xlim(), 200)],
                       fontsize=20)
    ax.tick_params(bottom=False)
    if file:
        fig.savefig(file, dpi = 300, transparent=True, bbox_inches='tight')
    return fig, trajectories

#example plot command
#np.random.seed(100)
#DDM_plot(1.2, .2, 2, n=100, plot_n=1, percentiles=[.2, .4, .95], only_positive=False,
#         plot_distributions=True)

def dendroheatmap(link, dist_df, clusters=None,
                  label_fontsize=None, labels=True,
                  figsize=None, title=None, filename=None):
    """Take linkage and distance matrices and plot
    
    Args:
        link: linkage matrix
        dist_df: distance dataframe where index/columns are in the same order
                 as the input to link
        clusters: (optional) list of cluster labels created from the linkage 
                   used to parse the dendrogram heatmap
                   Assumes that clusters are contiguous along the dendrogram
        label_fontsize: int, fontsize for labels
        labels: (optional) bool, whether to show labels on heatmap
        figsize: figure size
        filename: string. If given, save to this location
    """
    row_dendr = dendrogram(link, labels=dist_df.index, no_plot = True)
    rowclust_df = dist_df.iloc[row_dendr['leaves'], row_dendr['leaves']]
    # plot
    if figsize is None:
        figsize=(16,16)
    if label_fontsize == None:
            label_fontsize = figsize[1]*.27
    sns.set_style("white")
    fig = plt.figure(figsize = figsize)
    ax1 = fig.add_axes([.16,.3,.62,.62]) 
    
    cax = fig.add_axes([0.21,0.25,0.5,0.02]) 
    sns.heatmap(rowclust_df, ax=ax1, xticklabels=False,
                yticklabels=True,
                cbar_ax=cax, 
                cbar_kws={'orientation': 'horizontal'})
    ax1.yaxis.tick_right()
    # update colorbar ticks
    cbar = ax1.collections[0].colorbar
    cbar.set_ticks([0, .5, .99])
    cbar.set_ticklabels([0, .5, ceil(dist_df.max().max())])
    cax.tick_params(labelsize=20)
    # reorient axis labels
    ax1.tick_params(labelrotation=0)
    ax1.set_yticks(ax1.get_yticks()+.25)
    ax1.set_yticklabels(rowclust_df.columns[::-1], rotation=0, 
                       rotation_mode="anchor", fontsize=label_fontsize, 
                       visible=labels)
    ax1.set_xticklabels(rowclust_df.columns, rotation=-90, 
                       rotation_mode = "anchor", ha = 'left')
    ax2 = fig.add_axes([.01,.3,.15,.62])
    plt.axis('off')
    # plot dendrogram
    row_dendr = dendrogram(link, orientation='left',  ax = ax2, 
                           color_threshold=-1,
                           above_threshold_color='gray') 
    ax2.invert_yaxis()
    if title is not None:
        ax1.set_title(title, fontsize=40)
    
    # add parse lines between trees 
    if clusters is not None:
        groups = clusters[row_dendr['leaves']][::-1]
        cuts = []
        curr = groups[0]
        for i,label in enumerate(groups[1:]):
            if label!=curr:
                cuts.append(i+1)
                curr=label
        
        for ax, color in [(ax1, 'w')]:
            y_min, y_max = ax.get_ylim()
            ticks = [(tick - y_min)/(y_max - y_min) for tick in ax.get_yticks()]
            pad = (ticks[0]-ticks[1])/2
            separations = (ticks+pad)*max(y_min, y_max)
            for c in cuts:
                ax.hlines(separations[c], 0, len(rowclust_df), colors=color,
                          linestyles='dashed') 
                
    if filename:
        fig.savefig(filename, bbox_inches='tight')
        plt.close()
    return fig

def get_dendrogram_color_fun(Z, labels, clusters, color_palette=sns.hls_palette):
    """ return the color function for a dendrogram
    
    ref: https://stackoverflow.com/questions/38153829/custom-cluster-colors-of-scipy-dendrogram-in-python-link-color-func
    Args:
        Z: linkage 
        Labels: list of labels in the order of the dendrogram. They should be
            the index of the original clustered list. I.E. [0,3,1,2] would
            be the labels list - the original list reordered to the order of the leaves
        clusters: cluster assignments for the labels in the original order
    
    """
    dflt_col = "#808080" # Unclustered gray
    color_palette = color_palette(len(np.unique(clusters)))
    D_leaf_colors = {i: to_hex(color_palette[clusters[i]-1]) for i in labels}
    # notes:
    # * rows in Z correspond to "inverted U" links that connect clusters
    # * rows are ordered by increasing distance
    # * if the colors of the connected clusters match, use that color for link
    link_cols = {}
    for i, i12 in enumerate(Z[:,:2].astype(int)):
      c1, c2 = (link_cols[x] if x > len(Z) else D_leaf_colors[x]
        for x in i12)
      link_cols[i+1+len(Z)] = c1 if c1 == c2 else dflt_col
    return lambda x: link_cols[x], color_palette

def heatmap(df):
    """
    :df: plot heatmap
    """
    plt.Figure(figsize = [16,16])
    sns.set_style("white")
    fig = plt.figure(figsize = [12,12])
    ax = fig.add_axes([.1,.2,.6,.6]) 
    cax = fig.add_axes([0.02,0.3,0.02,0.4]) 
    sns.heatmap(df, ax = ax, cbar_ax = cax, xticklabels = False)
    ax.yaxis.tick_right()
    ax.set_yticklabels(df.columns[::-1], rotation=0, rotation_mode="anchor", fontsize = 'large')
    ax.set_xticklabels(df.columns, rotation=-90, rotation_mode = "anchor", ha = 'left') 
    return fig

def save_figure(fig, loc, save_kws=None):
    """ Saves figure in location and creates directory tree if needed """
    if save_kws is None:
        save_kws = {}
    directory = os.path.dirname(loc)
    if directory != "":
        os.makedirs(directory, exist_ok=True)
    fig.savefig(loc, **save_kws)
    
class CurvedText(mtext.Text):
    """
    A text object that follows an arbitrary curve.
    https://stackoverflow.com/questions/19353576/curved-text-rendering-in-matplotlib
    """
    def __init__(self, x, y, text, axes, **kwargs):
        super(CurvedText, self).__init__(x[0],y[0],' ', **kwargs)

        axes.add_artist(self)

        ##saving the curve:
        self.__x = x
        self.__y = y
        self.__zorder = self.get_zorder()

        ##creating the text objects
        self.__Characters = []
        for c in text:
            if c == ' ':
                ##make this an invisible 'a':
                t = mtext.Text(0,0,'a', **kwargs)
                t.set_alpha(0.0)
            else:
                t = mtext.Text(0,0,c, **kwargs)

            #resetting unnecessary arguments
            t.set_ha('center')
            t.set_rotation(0)
            t.set_zorder(self.__zorder +1)

            self.__Characters.append((c,t))
            axes.add_artist(t)


    ##overloading some member functions, to assure correct functionality
    ##on update
    def set_zorder(self, zorder):
        super(CurvedText, self).set_zorder(zorder)
        self.__zorder = self.get_zorder()
        for c,t in self.__Characters:
            t.set_zorder(self.__zorder+1)

    def draw(self, renderer, *args, **kwargs):
        """
        Overload of the Text.draw() function. Do not do
        do any drawing, but update the positions and rotation
        angles of self.__Characters.
        """
        self.update_positions(renderer)

    def update_positions(self,renderer):
        """
        Update positions and rotations of the individual text elements.
        """

        #preparations

        ##determining the aspect ratio:
        ##from https://stackoverflow.com/a/42014041/2454357

        ##data limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        ## Axis size on figure
        figW, figH = self.axes.get_figure().get_size_inches()
        ## Ratio of display units
        _, _, w, h = self.axes.get_position().bounds
        ##final aspect ratio
        aspect = ((figW * w)/(figH * h))*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])

        #points of the curve in figure coordinates:
        x_fig,y_fig = (
            np.array(l) for l in zip(*self.axes.transData.transform([
            (i,j) for i,j in zip(self.__x,self.__y)
            ]))
        )

        #point distances in figure coordinates
        x_fig_dist = (x_fig[1:]-x_fig[:-1])
        y_fig_dist = (y_fig[1:]-y_fig[:-1])
        r_fig_dist = np.sqrt(x_fig_dist**2+y_fig_dist**2)

        #arc length in figure coordinates
        l_fig = np.insert(np.cumsum(r_fig_dist),0,0)

        #angles in figure coordinates
        rads = np.arctan2((y_fig[1:] - y_fig[:-1]),(x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)

        # center text
        text_width = np.sum([t.get_window_extent(renderer=renderer).width for c,t in self.__Characters])
        rel_pos = max((l_fig[-1]-text_width)//2+1, 5)
        for c,t in self.__Characters:
            #finding the width of c:
            t.set_rotation(0)
            t.set_va('center')
            bbox1  = t.get_window_extent(renderer=renderer)
            w = bbox1.width
            h = bbox1.height

            #ignore all letters that don't fit:
            if rel_pos+w/2 > l_fig[-1]:
                t.set_alpha(0.0)
                rel_pos += w
                continue

            elif c != ' ':
                t.set_alpha(1.0)

            #finding the two data points between which the horizontal
            #center point of the character will be situated
            #left and right indices:
            il = np.where(rel_pos+w/2 >= l_fig)[0][-1]
            ir = np.where(rel_pos+w/2 <= l_fig)[0][0]

            #if we exactly hit a data point:
            if ir == il:
                ir += 1

            #how much of the letter width was needed to find il:
            used = l_fig[il]-rel_pos
            rel_pos = l_fig[il]

            #relative distance between il and ir where the center
            #of the character will be
            fraction = (w/2-used)/r_fig_dist[il]

            ##setting the character position in data coordinates:
            ##interpolate between the two points:
            x = self.__x[il]+fraction*(self.__x[ir]-self.__x[il])
            y = self.__y[il]+fraction*(self.__y[ir]-self.__y[il])

            #getting the offset when setting correct vertical alignment
            #in data coordinates
            t.set_va(self.get_va())
            bbox2  = t.get_window_extent(renderer=renderer)

            bbox1d = self.axes.transData.inverted().transform(bbox1)
            bbox2d = self.axes.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0]-bbox1d[0])

            #the rotation/stretch matrix
            rad = rads[il]
            rot_mat = np.array([
                [math.cos(rad), math.sin(rad)*aspect],
                [-math.sin(rad)/aspect, math.cos(rad)]
            ])

            ##computing the offset vector of the rotated character
            drp = np.dot(dr,rot_mat)

            #setting final position and rotation:
            t.set_position(np.array([x,y])+drp)
            t.set_rotation(degs[il])

            t.set_va('center')
            t.set_ha('center')

            #updating rel_pos to right edge of character
            rel_pos += w-used
