from expanalysis.experiments.ddm_utils import load_model
from kabuki.analyze import gelman_rubin
import hddm
import numpy as np
from os import path
import pandas as pd
import pymc as pm


def plot_posteriors(model, params=None, plot_subjs=False, save=False, **kwargs):
    """
    plot the nodes posteriors
    Input:
        params (optional) - a list of parameters to plot.
        plot_subj (optional) - plot subjs nodes
        kwargs (optional) - optional keywords to pass to pm.Matplot.plot

    TODO: add attributes plot_subjs and plot_var to kabuki
    which will change the plot attribute in the relevant nodes
    """

    # should we save the figures
    kwargs.pop('last', None)

    if isinstance(params, str):
         params = [params]

    # loop over nodes and for each node if it
    for (name, node) in model.iter_non_observeds():
        if (params is None) or (name in params): # plot params if its name was mentioned
            if not node['hidden']: # plot it if it is not hidden
                plot_value = node['node'].plot
                if (plot_subjs and node['subj']): # plot if it is a subj node and plot_subjs==True
                    node['node'].plot = True
                if (params is not None) and  (name in params): # plot if it was sepecficily mentioned
                    node['node'].plot = True
                pm.Matplot.plot(node['node'], last=save, **kwargs)
                node['node'].plot = plot_value


def plot_subset_hddm_subjs(m, params, n=4):
    subjs = list(m.nodes_db.filter(regex='^t.*subj', axis=0).index)
    shuffle = list(range(len(subjs)))
    np.random.shuffle(shuffle)
    subj_params = []
    for param in params:
        subjs = list(m.nodes_db.filter(regex='%s.*subj' % param, axis=0).index)
        assert len(subjs) <= len(shuffle)*2
        if len(subjs) == len(shuffle)*2:
            tmp = [subjs[i+len(shuffle)] for i in shuffle]
            subjs = [subjs[i] for i in shuffle] + tmp
            subj_params += subjs[len(shuffle):len(shuffle)+n]
        else:
            subjs = [subjs[i] for i in shuffle]
        subj_params += subjs[0:n]
    plot_posteriors(m, subj_params)

""" 
Example Code

task = 'threebytwo'
output_loc = '/mnt/OAK/behavioral_data/mturk_complete_output'
db_path = path.join(output_loc, '%s_parallel_output' % task, '*traces*.db')
empty_path = path.join(output_loc, '%s_empty.model' % task)
#load dvs
dvs = pd.read_json(path.join(output_loc, '%s_mturk_complete_DV.json' % task))
# load model
m, models = load_model(empty_path, db_path)
# plot group level main DDM statistics (a, v, t)
plot_posteriors(m)
# plot random 4 subjects for all ddm statistics
plot_subset_hddm_subjs(m, ['^a', '^v', '^t'], n=2)
# test for convergence
out = gelman_rubin(models[1:]) # the first model has the 

"""