import matplotlib.pyplot as plt
import numpy as np
from os import  path
import pandas as pd
import seaborn as sns

from dimensional_structure.EFA_plots import plot_factor_correlation, plot_heatmap_factors
from dimensional_structure.utils import get_factor_groups
from selfregulation.utils.plot_utils import format_num, format_variable_names, save_figure
from selfregulation.utils.r_to_py_utils import get_attr


def plot_demo_factor_dist(results, c, figsize=12, dpi=300, ext='png', plot_dir=None):
    DA = results.DA
    sex = DA.raw_data['Sex']
    sex_percent = "{0:0.1f}%".format(np.mean(sex)*100)
    scores = DA.get_scores(c)
    axes = scores.hist(bins=40, grid=False, figsize=(figsize*1.3,figsize))
    axes = axes.flatten()
    f = plt.gcf()
    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    axes[-1].set_xlabel('N: %s, Female Percent: %s' % (len(scores), sex_percent), 
        labelpad=20)
    if plot_dir:
        filename = 'factor_distributions_DA%s.%s' % (c, ext)
        save_figure(f, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
        
def plot_DA(results, plot_dir=None, verbose=False, size=10, dpi=300, ext='png',
             plot_task_kws={}):
    c = results.DA.results['num_factors']
    #if verbose: print("Plotting BIC/SABIC")
    #plot_BIC_SABIC(EFA, plot_dir)
    if verbose: print("Plotting Distributions")
    plot_demo_factor_dist(results, c, plot_dir=plot_dir, dpi=dpi,  ext=ext)
    if verbose: print("Plotting factor correlations")
    plot_factor_correlation(results, c, title=False, DA=True,
                            size=size, plot_dir=plot_dir, dpi=dpi,  ext=ext)
    if verbose: print("Plotting factor bars")
    plot_heatmap_factors(results, c, thresh=0, DA=True, size=size, plot_dir=plot_dir, dpi=dpi,  ext=ext)