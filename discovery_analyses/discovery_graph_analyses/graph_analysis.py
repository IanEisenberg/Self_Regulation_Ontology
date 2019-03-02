#get utils
from selfregulation.utils.graph_utils import calc_connectivity_mat, find_intersection, get_fully_connected_threshold
from selfregulation.utils.graph_utils import  Graph_Analysis, threshold_proportional_sign
from selfregulation.utils.plot_utils import dendroheatmap
from selfregulation.utils.utils import get_behav_data

import bct
import igraph
import numpy as np
import pandas as pd
import seaborn as sns

# get dependent variables
graph_data = get_behav_data(file = 'taskdata_imputed.csv')
    

# ************************************
# ************ Connectivity Matrix *******************
# ************************************

spearman_connectivity = calc_connectivity_mat(graph_data, edge_metric = 'spearman')
distance_connectivity = calc_connectivity_mat(graph_data, edge_metric = 'distance')

print('Finished creating connectivity matrices')
# ************************************
# ********* Heatmaps *******************
# ************************************
# dendrogram heatmap
fig, column_order = dendroheatmap(spearman_connectivity, labels = True)

# ************************************
# ********* Graphs *******************
# ************************************
# signed spearman graph
thresholds = get_fully_connected_threshold(spearman_connectivity)
plot_t = thresholds['proportional']

GA = Graph_Analysis()
GA.setup(data = spearman_connectivity,
         thresh_func = threshold_proportional_sign,
         community_alg = bct.modularity_louvain_und_sign)
seed = 1337
gamma = np.arange(0,3,.2)
mod_scores = []
layout = None
reference=None
intersections = []
communities = []
for g in gamma:
    if layout==None:
        layout='circle'
    mod = GA.calculate_communities(gamma=g, seed=seed)
    mod_scores.append(mod)
    #GA.set_visual_style(layout=layout, plot_threshold = plot_t)
    #GA.display()
    if reference!=None:
        intersections.append(find_intersection(GA.G.vs['community'], reference))
    reference = GA.G.vs['community']
    communities.append(reference)

sns.plt.plot(gamma,mod_scores)
sns.plt.plot(gamma, [np.max(c) for c in communities])

subgraph_GA = GA.return_subgraph_analysis()
subgraph_GA.calculate_communities()
subgraph_GA.set_visual_style(layout='circle')
subgraph_GA.display()




    
# distance graph
thresholds = get_fully_connected_threshold(spearman_connectivity)
plot_t = thresholds['proportional']

GA = Graph_Analysis()
GA.setup(data = spearman_connectivity,
         thresh_func = threshold_proportional_sign,
         community_alg = bct.modularity_louvain_und_sign)
GA.calculate_communities()
GA.set_visual_style(layout='circle', plot_threshold = plot_t)
GA.display()



# signed graph
t = 1
thresholds = get_fully_connected_threshold(spearman_connectivity)
plot_t = thresholds['proportional']
t_f = bct.threshold_proportional
c_a = bct.modularity_louvain_und                                           

# circle layout                                                  
G_spearman, connectivity_mat, visual_style = Graph_Analysis(spearman_connectivity, community_alg = c_a, thresh_func = t_f,
                                                     reorder = False, threshold = t,  layout = 'kk', 
                                                     plot_threshold = plot_t, print_options = {'lookup': {}}, 
                                                    plot_options = {'inline': False})
# signed graph
t = 1
t_f = threshold_proportional_sign
c_a = bct.modularity_louvain_und_sign                                               

# circle layout                                                  
G_spearman, connectivity_mat, visual_style = Graph_Analysis(spearman_connectivity, community_alg = c_a, thresh_func = t_f,
                                                     reorder = True, threshold = t,  layout = 'circle', 
                                                     plot_threshold = t, print_options = {'lookup': {}}, 
                                                    plot_options = {'inline': False})

# *********************************
# Task Analysis
# ********************************
def node_importance(v):
    return (v['subgraph_eigen_centrality'], v['community'])

def integrate_over_measures(lst):
    return np.sum(lst)

def integrate_over_communities(lst):
    return np.sum(lst)
    
def get_task_importance(G):
    tasks = np.unique(list(map(lambda x: x.split('.')[0], G.vs['name'])))
    communities = range(1, max(G.vs['community'])+1)
    task_importance_df = pd.DataFrame(index = tasks, columns = ['comm' + str(c) for c in communities] + ['num_measures'])
    for task in tasks:
        measure_importance = [node_importance(v) for v in G.vs if v['name'].split('.')[0] == task]
        community_importance = [integrate_over_measures([m[0] for m in measure_importance if m[1] == c]) for c in communities]
        task_importance_df.loc[task,:] = community_importance + [len(measure_importance)]
    return task_importance_df
    
df = get_task_importance(G_spearman)
n_comms = df.shape[1]-1
task_df = df.filter(regex = '^(?!.*survey).*$', axis = 0)
sorted_list = (task_df.iloc[:,0:n_comms]).apply(lambda x: integrate_over_communities(x), axis = 1).sort_values(ascending = False)

#plot task_df
plot_df = task_df.iloc[:,0:n_comms]
plot_df.loc[:,'task'] = plot_df.index
plot_df = pd.melt(plot_df, id_vars = 'task', var_name = 'community', value_name = 'sum centrality')
sns.set_context('poster')
sns.barplot(x = 'community', y = 'sum centrality', data = plot_df, hue = 'task')
leg = sns.plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', ncol=1)
sns.plt.savefig('Plots/task_importance.pdf', bbox_inches = 'tight', pad_inches = 2, dpi = 300)

with sns.color_palette(['b','g','r','gray']):
    ax = sns.barplot(x = 'task', y = 'sum centrality', data = plot_df, ci = False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, rotation_mode = "anchor", ha = 'left')
    sns.plt.savefig('Plots/community_importance.png', dpi = 300, bbox_inches = 'tight', pad_inches = 2)

# save graph
# distance graph
t = 1
plot_t = .1
em = 'distance'
t_f = bct.threshold_proportional

layout = None
ref_community = None
for gamma in np.arange(.5,2,.05):
    c_a = lambda x: bct.community_louvain(x, gamma = gamma)
    layout = layout or 'kk'
    G_w, connectivity_adj, visual_style = Graph_Analysis(distance_connectivity, community_alg = c_a, ref_community = ref_community,
                                                         thresh_func = t_f, threshold = t, plot_threshold = plot_t, 
                                                         layout = layout,
                                                         print_options = {'lookup': {}, 'file': 'Plots/gamma=' + str(gamma) + '_weighted_distance.txt'}, 
                                                        plot_options = {'inline': False, 'target': 'Plots/gamma=' + str(gamma) + '_weighted_distance.pdf'})
    if type(layout) != igraph.layout.Layout:
        layout = visual_style['layout']
    ref_community = G_w.vs['community']
                                            
        
# other graph exploration
plot_data = pd.DataFrame([G_spearman.vs['part_coef_pos'], 
                          G_spearman.vs['subgraph_eigen_centrality'],
                          G_spearman.vs['community']], index = ['Participation', 'Community Centrality', 'Community']).T
sns.set_context('poster')
sns.lmplot('Participation', 'Community Centrality', data = plot_data, hue = 'Community', fit_reg = False, size = 10, scatter_kws={"s": 100})
sns.plt.ylim([-.05,1.05])
sns.plt.savefig('/home/ian/tmp/plot1.png',dpi = 300)                                                


subgraph = community_reorder(get_subgraph(G_w,2))
print_community_members(subgraph)
subgraph_visual_style = get_visual_style(subgraph, vertex_size = 'eigen_centrality')
plot_graph(subgraph, visual_style = subgraph_visual_style, layout = 'circle', inline = False)












