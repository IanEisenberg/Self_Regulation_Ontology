import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import scale
from dimensional_structure.HCA_plots import get_dendrogram_color_fun
from dimensional_structure.utils import hierarchical_cluster
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_recent_dataset, get_demographics

dataset = get_recent_dataset()
results = load_results(dataset)

demographics = get_demographics()
demo_factors = results['survey'].DA.get_scores()


def plot_dendrogram(clustering, size=10):
    link = clustering['linkage']
    labels = clustering['labels']
    link_function, colors = get_dendrogram_color_fun(link, clustering['reorder_vec'],
                                                     labels)
    
    # set figure properties
    figsize = (size, size*.6)
    with sns.axes_style('white'):
        fig = plt.figure(figsize=figsize)
        # **********************************
        # plot dendrogram
        # **********************************
        with plt.rc_context({'lines.linewidth': size*.125}):
            dendrogram(link,  link_color_func=link_function,
                       orientation='top')

def get_cluster_memberships(cluster):
    raw_index = cluster['distance_df'].index
    labels = cluster['labels']
    cluster_memberships = {k:set() for k in range(1,max(labels)+1)}
    for i,l in enumerate(labels):
        cluster_memberships[l].add(raw_index[i])
    return cluster_memberships
    
def get_avg_demos(cluster_memberships, demographics):
    cluster_demos = {}
    for cluster, members in cluster_memberships.items():
        cluster_demos[cluster] = demographics.loc[members].mean()
    return pd.DataFrame(cluster_demos).T

for name, result in results.items():
    data = result.data
    scaled_data = pd.DataFrame(scale(data), index=data.index, columns=data.columns)
    scores = result.EFA.get_scores()
    raw_cluster = hierarchical_cluster(scaled_data)
    score_cluster = hierarchical_cluster(scores)

    cluster_memberships = get_cluster_memberships(score_cluster)
    cluster_demos = get_avg_demos(cluster_memberships, demographics)
    cluster_demo_factors = get_avg_demos(cluster_memberships, demo_factors)
    
    



