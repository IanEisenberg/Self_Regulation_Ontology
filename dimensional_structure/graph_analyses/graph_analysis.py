#get utils
from selfregulation.utils.graph_utils import get_adj, \
    find_intersection, get_fully_connected_threshold, remove_island_variables
from selfregulation.utils.graph_utils import  Graph_Analysis, threshold, \
    threshold_proportional_sign
from selfregulation.utils.utils import get_behav_data, get_info

import bct
import igraph
import numpy as np
from os.path import join, exists
from os import makedirs
import pandas as pd
import seaborn as sns

# generic variables
save_plots = False
plot_dir = join(get_info('base_directory'),'dimensional_structure','Plots')

# get dependent variables
graph_data = get_behav_data(file = 'taskdata_imputed.csv')  



def run_graph_analysis(adj_dict, save_plots=False):
    """
    Takes in a dictionary with two keys: "name" and "adj", specifying
    an adjacency matrix (as a dataframe) and its corresponding name
    """
    def plot_name(name):
        return join(plot_dir,adj_name,name)
        
    adj_name = adj_dict['name']
    adj = adj_dict['adj']
    # if saving plots, make sure directory exists
    if save_plots: 
        makedirs(join(plot_dir,adj_name), exist_ok=True)
    
    # ************************************
    # ********* Graphs *******************
    # ************************************
    seed = 1337
    community_alg = bct.modularity_louvain_und_sign
    # exclude variables with no correlations with the rest of the graph
    adj = remove_island_variables(adj)
            
    # create graph object
    GA = Graph_Analysis()
    GA.setup(adj = adj,
             community_alg = community_alg)
    
    # search for ideal gamma value
    gamma = np.arange(0,3,.1)
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
        if reference!=None:
            intersections.append(find_intersection(GA.G.vs['community'],
                                                   reference))
        reference = GA.G.vs['community']
        communities.append(reference)
    
    # plot modularity index vs gamma
    fig = sns.plt.figure()
    sns.plt.plot(gamma,mod_scores,'o-')
    sns.plt.xlabel('Gamma')
    sns.plt.ylabel('Modularity')
    if save_plots: fig.savefig(plot_name('gamma_vs_modularity.pdf'),
                                bbox_inches='tight')
    
    # calculate the mean number of nodes in each community for each gamma value
    size_per_gamma = []
    # iterative over communities identified with different gammas
    for comm in communities: 
        nodes_per_comm=[np.sum(np.equal(c,comm)) 
                        for c in range(1,np.max(comm)+1)]
        # exclude signle communities, following Ashourvan et al. 2017
        nodes_per_comm = [i for i in nodes_per_comm if i!=1]
        size_per_gamma+=[np.mean(nodes_per_comm)]
         
    fig = sns.plt.figure()                       
    sns.plt.plot(gamma, [np.max(c) for c in communities], 'o-', 
                         label='# Communities')
    sns.plt.plot(gamma, size_per_gamma, 'o-', 
                 label='Mean Size of Communities')
    sns.plt.legend()
    sns.plt.xlabel('Gamma')
    if save_plots: fig.savefig(plot_name('community_stats.pdf'),
                               bbox_inches='tight')
    
    # use best gamma
    best_gamma = gamma[np.argmax(mod_scores)]
    GA.calculate_communities(gamma=best_gamma, seed=seed, reorder=True)
    
    # plot communities of graph in dendrohistogram
    fig = sns.plt.figure(figsize=[20,16])
    sns.heatmap(GA.graph_to_dataframe(GA.G),square=True)
    if save_plots: fig.savefig(plot_name('graph_community_heatmap.pdf'),
                               bbox_inches='tight')
# ****************************************************
# ************ Connectivity Matrix *******************
# ****************************************************

spearman_connectivity = get_adj(graph_data, edge_metric = 'spearman')
distance_connectivity = get_adj(graph_data, edge_metric = 'distance')
gamma = 0
glasso_connectivity = get_adj(graph_data, edge_metric = 'EBICglasso',
                              gamma=gamma)

print('Finished creating connectivity matrices')

# ***************************************************
# ********* Distribution of Edges *******************
# ***************************************************
edge_mats = [{'name': 'spearman', 'adj': spearman_connectivity},
             {'name': 'distance', 'adj': distance_connectivity},
             {'name': 'glasso', 'adj': glasso_connectivity}]
fig = sns.plt.figure(figsize=[12,8])
fig.suptitle('Distribution of Edge Weights', size='x-large')
for i,mat in enumerate(edge_mats):
    sns.plt.subplot(1,3,i+1)
    sns.plt.hist(mat['adj'].replace(1,0).values.flatten(), bins =100)
    sns.plt.title(mat['name'])
sns.plt.tight_layout()
sns.plt.subplots_adjust(top=0.85)
if save_plots: fig.savefig(join(plot_dir,'connectivity_distributions.pdf'),
                           bbox_inches='tight')

# ***************************************
# ********* Select Connectivity Matrix **
# ***************************************
for adj_dict in edge_mats:
    run_graph_analysis(adj_dict, True)
    














"""               
# plot graph
layout='circle'
GA.set_visual_style(layout=layout,plot_adj=True)
if save_plots: GA.display(print_options={'file': join(plot_dir, adj_name,
                                                      'gamma_%s_%s_graph.txt'
                                                      % (best_gamma, layout))}, 
                          plot_options={'inline': False, 
                                        'target': join(plot_dir, adj_name,
                                                       'gamma_%s_%s_graph.pdf'
                                                       % (best_gamma, layout))})        

subgraph_GA = GA.return_subgraph_analysis(community=3)
subgraph_GA.calculate_communities()
subgraph_GA.set_visual_style(layout='kk', plot_adj=True)
subgraph_GA.display()







    
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

"""







