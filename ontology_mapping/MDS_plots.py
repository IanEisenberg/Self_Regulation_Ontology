import matplotlib.pyplot as plt
import numpy as np
from os import makedirs, path
import pandas as pd
from scipy.spatial.distance import  squareform
from sklearn.manifold import MDS
import seaborn as sns
from dimensional_structure.HCA_plots import abs_pdist
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_info, get_recent_dataset

# get dataset of interest
basedir=get_info('base_directory')
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
plot_dir = output_dir = path.join(get_info('results_directory'),
                       'ontology_reconstruction', results['task'].ID, 'Plots')
makedirs(plot_dir, exist_ok=True)


# plot
size=8
basefont = size*1.3
basemarker = size**2*4
basewidth = size*.12


# ****************************************************************************
# calculate distances 
# ****************************************************************************
participant_distances = squareform(abs_pdist(data.T))
participant_distances = results['task'].HCA.results['data']['clustered_df']
loading_distances = results['task'].HCA.results['EFA5_oblimin']['clustered_df']

# ****************************************************************************
# MDS Plots
# ****************************************************************************
colored_variables = ['digit_span.forward_span',
                      'spatial_span.forward_span',
                      'ravens.score',
                      'choice_reaction_time.hddm_drift']
colors = sns.color_palette(n_colors=4)
greys = np.array([[.5, .5, .5, .3]]*loading_distances.shape[0])

mds_colors = greys.copy()
interest_index = []
misc_index = []
color_i = 0
for i, label in enumerate(loading_distances.index):
    if label in colored_variables:
        interest_index.append(i)
        mds_colors[i] = list(colors[color_i]) + [1]
        color_i += 1
    else:
        misc_index.append(i)
mds_index = misc_index + interest_index


for color_type, coloring in [('greys', greys), 
                             ('colors', mds_colors)]:
    f, ax = plt.subplots(1,2, figsize=(size*2,size))
    participant_mds, loading_mds = ax
    
    # plot raw MDS
    np.random.seed(700)
    mds = MDS(dissimilarity='precomputed')
    mds_out = mds.fit_transform(participant_distances)
    participant_mds.scatter(mds_out[mds_index,0], mds_out[mds_index,1], 
                s=basemarker,
                marker='h',
                facecolors=coloring[mds_index],
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
                facecolors=coloring[mds_index],
                edgecolors='white',
                linewidths=basewidth/2)
    loading_mds.set_xticklabels(''); loading_mds.set_yticklabels('')
    loading_mds.tick_params(axis='both', length=0)
    loading_mds.axis('off'); 
    f.savefig(path.join(plot_dir, 'MDS_%s.png' % color_type), 
              dpi=300, transparent=True)