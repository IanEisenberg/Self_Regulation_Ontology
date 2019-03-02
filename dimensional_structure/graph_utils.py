import bct
import igraph
from itertools import combinations 
import numpy as np
import pandas as pd
from pprint import pprint
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from selfregulation.utils.r_to_py_utils import qgraph_cor
from sklearn.metrics.cluster import normalized_mutual_info_score

#work around for spyder bug in python 3
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

# Utilities
def calc_small_world(G):
    # simulate random graphs with same number of nodes and edges
    sim_out = simulate(rep = 10000, fun = lambda: gen_random_graph(n = len(G.vs), m = len(G.es)))
    # get average C and L for random
    C_random = np.mean([i[1] for i in sim_out])
    L_random = np.mean([i[2] for i in sim_out])
    # calculate relative clustering and path length vs random networks
    Gamma = G.transitivity_undirected()/C_random
    Lambda = G.average_path_length()/L_random
    # small world coefficient
    Sigma = Gamma/Lambda
    return (Sigma, Gamma, Lambda)

def distcorr(X, Y):
    """ Compute the distance correlation function
    
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    X,Y = zip(*[v for i,v in enumerate(zip(X,Y)) if not np.any(np.isnan(v))])
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

def distcorr_mat(M):
    n = M.shape[1]
    corr_mat = np.ones([n,n])
    for i in range(n):
        for j in range(i):
            distance_corr = distcorr(M[:,i], M[:,j])
            corr_mat[i,j] = corr_mat[j,i] =  distance_corr
    return corr_mat
    
def gen_random_graph(n = 10, m = 10, template_graph = None):
    if template_graph:
        G = template_graph.copy()
        G.rewire() # bct.randomizer_bin_und works on binary adjrices
    else:
        #  Generates a random binary graph with n vertices and m edges
        G = igraph.Graph.Erdos_Renyi(n = n, m = m)    
    # get cluster coeffcient. Transitivity is closed triangles/total triplets
    c = G.transitivity_undirected() 
    # get average (shortest) path length
    l = G.average_path_length()
    return (G,c,l,c/l)

def get_percentile_weight(W, percentile):
    return np.percentile(np.abs(W[np.tril_indices_from(W, k = -1)]),percentile)
    
def pairwise_MI(data):
    columns = data.columns
    MI_df = pd.DataFrame(index = columns, columns = columns)
    for c1,c2 in combinations(columns, 2):
        cleaned = data[[c1,c2]].dropna()
        MI = normalized_mutual_info_score(cleaned[c1], cleaned[c2])
        MI_df.loc[c1,c2] = MI
        MI_df.loc[c2,c1] = MI
    return MI_df.astype(float)
    
def simulate(rep = 1000, fun = lambda: gen_random_graph(100,100)):
    output = []
    for _ in range(rep):
        output.append(fun())
    return output
          
    
def threshold_proportional_sign(W, threshold):
    sign = np.sign(W)
    thresh_W = bct.threshold_proportional(np.abs(W), threshold)
    W = thresh_W * sign
    return W
        
# community functions      
def get_adj(data, edge_metric = 'pearson', **kwargs):
    """ 
    Creates a connectivity matrix from a dataframe using a specified metric.
    Options:
        pearson
        abs_pearson
        spearman
        distance (distance correlation)
        MI (mutual information)
        EBICglasso (graphical lasso from the qgraph package)
        corauto (from the qgraph package, based on lavcor from lavaan)
    """
    assert edge_metric in ['pearson','spearman','MI','EBICglasso', 'corauto', 'abs_pearson','abs_spearman', 'distance'], \
        'Invalid edge metric passed. Must use "pearson", "spearman", "distance", "EBICglasso", "curauto" or "MI" '
    if edge_metric == 'MI':
        adj = pd.DataFrame(pairwise_MI(data))
    elif edge_metric == 'distance':
        adj = pd.DataFrame(distcorr_mat(data.as_matrix()))
    elif edge_metric == 'EBICglasso':
        adj, tuning_param = qgraph_cor(data, True, kwargs.get('gamma',0))
        print('Using tuning param %s for EBICglasso' % tuning_param)
    elif edge_metric == 'corauto':
        adj = qgraph_cor(data)
    else:
        *qualifier, edge_metric = edge_metric.split('_')
        if (qualifier or [None])[0] == 'abs':
            adj = abs(data.corr(method = edge_metric))
        else:
            adj = data.corr(method = edge_metric)
    adj.columns = data.columns
    adj.index = data.columns
    return adj

def construct_relational_tree(intersections, proportional=False):
    G = igraph.Graph()
    layer_start = 0
    colors = ['red','blue','green','violet']*4
    for intersection in intersections:
        if proportional:
            intersection = intersection/intersection.sum(axis=0)
        curr_color = colors.pop()
        origin_length = intersection.shape[0]
        target_length = intersection.shape[1]
        if len(G.vs)==0:
            G.add_vertices(origin_length)
        G.add_vertices(target_length)
        for i in range(origin_length):
            for j in range(target_length):
                G.add_edge(i+layer_start,j+origin_length+layer_start,weight=intersection[i,j],color = curr_color)
        layer_start+=intersection.shape[0]
    igraph.plot(G, layout = 'rt', **{'inline': False, 
                                     'vertex_label': range(len(G.vs)), 
                                     'edge_width':[w for w in G.es['weight']],
                                     'edge_color': G.es['color'],
                                     'bbox': (1000,1000)})
    #G.write_dot('test.dot')

def find_intersection(community, reference):
    ref_lists = [[i for i,c in enumerate(reference) if c==C] for C in np.unique(reference)]
    comm_lists = [[i for i,c in enumerate(community) if c==C] for C in np.unique(community)]
    # each element relates to a community
    intersection = [[len(set(ref).intersection(comm)) for ref in ref_lists] for comm in comm_lists]
    return np.array(intersection).T

def get_fully_connected_threshold(adj):
    '''Get a threshold above the initial value such that the graph is fully connected
    '''
    threshold_mat = adj.values.copy()
    np.fill_diagonal(threshold_mat,0)
    abs_threshold = np.min(np.max(threshold_mat, axis = 1))
    proportional_threshold = np.mean(threshold_mat>=(abs_threshold-.001))
    return {'absolute': abs_threshold, 'proportional': proportional_threshold}            

def remove_island_variables(adj):
    """
    Remove variables with no correlation with any other variable
    """
    return adj.loc[(adj.sum()!=1),(adj.sum()!=1)]
    
def threshold(adj, threshold_func, threshold):
    adj_matrix = threshold_func(adj.values,threshold)
    adj_df = pd.DataFrame(adj_matrix, columns=adj.columns, index=adj.index)
    return adj_df




    
# Graph Analysis Class Definition

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 22:40:47 2016

@author: ian
"""

class Graph_Analysis(object):
    def __init__(self):
        self.adj = None
        self.G = None
        self.node_order = []
        self.weight = True
        self.community_alg = bct.community_louvain
        self.ref_community = None
        self.visual_style = None
        self.print_options = {}
        self.plot_options = {}
        
    def setup(self, adj, weighted=True,
              community_alg=None, 
              ref_community=None):
        """
        Creates and displays graphs of a data matrix.
        
        Parameters
        ----------
        data: pandas DataFrame
            data to use to create the graph
        thresh_func: function that takes in a connectivity matrix and thresholds
            any algorithm that returns a connectivity matrix of the same size as the original may be used.
            intended to be used with bct.threshold_proportional or bct.threshold_absolute
        community_alg: function that takes in a connectivity matrix and returns community assignment
            intended to use algorithms from brain connectivity toolbox like commnity_louvain or 
            modularity_und. Must return a list of community assignments followed by Q, the modularity
            index
        threshold: float 0 <= x <= 1, optional
            the proportion of weights to keep (to be passed to bct.threshold_proportion)
        weight: bool, optional
            if True, creates a weighted graph (vs. a binary)
            
        """
        assert type(adj) == pd.core.frame.DataFrame
        self.adj = adj.replace(1,0) # remove self edges
        self.community_alg = community_alg
        self.ref_community = ref_community
        # numpy matrix used for graph functions
        graph_mat = self.adj.values
        
        # check if binary
        if weighted==False:
            graph_mat = np.ceil(graph_mat)
            G = igraph.Graph.Adjacency(graph_mat.tolist(), mode = 'undirected')
        else:
            G = igraph.Graph.Weighted_Adjacency(graph_mat.tolist(), mode = 'undirected')
        # label vertices of G
        G.vs['id'] = range(len(G.vs))
        G.vs['name'] = adj.columns
        # set class variables
        self.graph_mat = graph_mat
        self.G = G
        self.node_order = list(range(graph_mat.shape[0]))
            
    def calculate_communities(self, reorder=False, **kwargs):
        assert self.community_alg is not None, \
            print("Community algorithm has not been set!")
        G = self.G
        graph_mat = self.graph_mat
        # calculate community structure
        comm, mod = self.community_alg(graph_mat, **kwargs)
        # if there is a reference, relabel communities based on their closest association    
        if self.ref_community:
            comm = self._relabel_community(comm,self.ref_community)
        # label vertices of G
        G.vs['community'] = comm
        G.vs['within_module_degree'] = bct.module_degree_zscore(graph_mat,comm)
        if np.min(graph_mat) < 0:
            participation_pos, participation_neg = bct.participation_coef_sign(graph_mat, comm)
            G.vs['part_coef_pos'] = participation_pos
            G.vs['part_coef_neg'] = participation_neg
        else:
            G.vs['part_coef'] = bct.participation_coef(graph_mat, comm)
        if reorder:
            self.reorder()
        # calculate subgraph (within-community) characteristics
        self._subgraph_analysis()
        return mod
    
    def calculate_centrality(self):
        G = self.G
        if self.weight:
            G.vs['eigen_centrality'] = G.eigenvector_centrality(directed = False, weights = G.es['weight'], scale=False)
        else:
            G.vs['eigen_centrality'] = G.eigenvector_centrality(directed = False)
            
    def create_visual_style(self, G, layout='kk', layout_graph = None, 
                            vertex_size = None, size = 1000, labels = None):
        """
        Creates an appropriate visual style for a graph. 
        
        Parameters
        ----------
        G: igraph object
            graph that the visual style will be based on
        layout: igraph.layout or str ('kk', 'circle', 'grid' or other igraph layouts), optional
            Determines how the graph is displayed. If a string is provided, assume it is
            a specification of a type of igraph layout using the graph provided. If an
            igraph.layout is provided it must conform to the number of vertices of the graph
        vertex_size: str, optional
            if defined, uses this graph metric to determine vertex size
        size: int, optinal
            determines overall size of graph display. Other sizes (vertex size, font size)
            are proportional to this value. Figures are always square
        labels: list the size of the number of vertices, optional
            Used to label vertices. Numbers are used if no labels are provided
        avg_num_edges: int > 1
            thresholds the edges on the graph so each node has, on average, avg_num_edges
            
        Returns
        ----------
        visual_style: dict
            the dictionary of attribute values to be used by igraph.plot
        """
        def make_layout():
            if type(layout) == igraph.layout.Layout:
                graph_layout = layout
                if layout_graph:
                    display_threshold = min([abs(i) for i in layout_graph.es['weight']])
                else:
                    display_threshold = 0
            elif type(layout) == str:
                if layout_graph:
                    graph_layout = layout_graph.layout(layout)
                    display_threshold = min([abs(i) for i in layout_graph.es['weight']])
                else:
                    graph_layout = G.layout(layout)
                    display_threshold = 0
            return graph_layout, display_threshold
        
        def get_vertex_colors():
            # color by community and within-module-centrality
            # each community is a different color palette, darks colors are more central to the module
            if 'community' in G.vs.attribute_names():
                community_count = np.max(G.vs['community'])
                if community_count <= 6 and 'subgraph_eigen_centrality' in G.vs.attribute_names():
                    num_colors = 20.0
                    palettes = ['Blues','Reds','Greens','Greys','Purples','Oranges']
                    
                    min_degree = np.min(G.vs['within_module_degree'])
                    max_degree = np.max(G.vs['within_module_degree']-min_degree)
                    within_degree = [(v-min_degree)/max_degree for v in G.vs['within_module_degree']]
                    within_degree = np.digitize(within_degree, bins = np.arange(0,1,1/num_colors))
                    
                    vertex_color = [sns.color_palette(palettes[v['community']-1], int(num_colors)+1)[within_degree[i]] for i,v in enumerate(G.vs)]
                else:
                    palette = sns.cubehelix_palette(max(community_count,10))
                    vertex_color = [palette[v['community']-1] for v in G.vs]
            else:
                vertex_color = 'red'
            return vertex_color
        
        def set_edges(visual_style):
            # normalize edges for weight
            edges = [w/np.max(G.es['weight']) for w in G.es['weight']]
            if 'weight' in G.es.attribute_names():
                thresholded_weights = [w if abs(w) > display_threshold else 0 for w in edges]
                if layout_graph!=None:
                    if min(layout_graph.es['weight'])>0:
                        thresholded_weights = [w if w > display_threshold else 0 for w in edges]
                visual_style['edge_width'] = [abs(w)**2.5*size/300.0 for w in edges]
                if np.sum([e<0 for e in G.es['weight']]) > 0:
                    visual_style['edge_color'] = [['#3399FF','#696969','#FF6666'][int(np.sign(w)+1)] for w in edges]
                else:
                    visual_style['edge_color'] = '#696969'
            
            
        G = self.G
        graph_layout, display_threshold = make_layout()
        vertex_color = get_vertex_colors()
        # set up visual style dictionary. Vertex label sizes are proportional to the total size
        visual_style = {'layout': graph_layout, 
                        'vertex_color': vertex_color, 
                        'vertex_label_size': size/130.0,
                        'bbox': (size,size),
                        'margin': size/20.0,
                        'inline': False}
        set_edges(visual_style)
        
        if vertex_size in G.vs.attribute_names():
            visual_style['vertex_size'] = [c*(size/60.0)+(size/100.0) for c in G.vs[vertex_size]]
        else:
            print('%s was not an attribute of G. Could not set vertex size!' % vertex_size)
        if labels:
            visual_style['vertex_label'] = labels
        
        return visual_style
        
    def display(self, plot=True, verbose=True,  print_options=None, plot_options=None):
        if verbose:
            if print_options==None:
                print_options = {}
            try:
                self._print_community_members(**print_options)
            except KeyError:
                print('Communities not detected! Run calculate_communities() first!')
        if plot:
            assert self.visual_style!=None, 'Must first call set_visual_style() !'
            if plot_options is None:
                plot_options = {}
            self._plot_graph(**plot_options)
                
    def get_subgraph(self, community = 1):
        G = self.G
        assert set(['community']) <=  set(G.vs.attribute_names()), \
            'No communities found! Call calculate_communities() first!'
        subgraph = G.induced_subgraph([v for v in G.vs if v['community'] == community])
        subgraph.vs['community'] = subgraph.vs['subgraph_community']
        subgraph.vs['eigen_centrality'] = subgraph.vs['subgraph_eigen_centrality']
        del subgraph.vs['subgraph_community']
        del subgraph.vs['subgraph_eigen_centrality']
        return subgraph
        
    def return_subgraph_analysis(self, community = 1):
        subgraph = self.graph_to_dataframe(self.get_subgraph(community))
        subgraph_GA=Graph_Analysis()
        subgraph_GA.setup(adj=subgraph,
                          community_alg=self.community_alg)
        return subgraph_GA
              
    
    def graph_to_dataframe(self, G=None):
        if G==None:
            G = self.G
        matrix = self._graph_to_matrix(G)
        graph_dataframe = pd.DataFrame(data = matrix, columns = G.vs['name'], index = G.vs['name'])
        return graph_dataframe
    
    def reorder(self, reorder_index=None):
        """
        Reorders nodes in graph (and corresponding entries of graph_mat)
        to reflect community assignment
        """
        G = self.G
        # if no reorder index given, sort by community then by
        # centrality within that community
        if reorder_index==None:
            community = G.vs['community']
            subgraph_centrality = G.vs['subgraph_eigen_centrality']
            sort_list = list(zip(community,subgraph_centrality))
            reorder_index = sorted(range(len(sort_list)), 
                                   key=lambda e: (sort_list[e][0],
                                                  -sort_list[e][1]))
        
        # hold graph attributes:
        attribute_names = G.vs.attributes()
        attribute_values = [G.vs[a] for a in attribute_names]
        attribute_df = pd.DataFrame(attribute_values, index = attribute_names).T
        sorted_df = attribute_df.reindex(reorder_index).reset_index()
        
        # rearrange connectivity matrix
        graph_mat = self._graph_to_matrix(G)
        graph_mat = graph_mat[:,reorder_index][reorder_index]
    
        # make a binary version if not weighted
        if 'weight' in G.es.attribute_names():
            G = igraph.Graph.Weighted_Adjacency(graph_mat.tolist(), mode = 'undirected')
        else:
            G = igraph.Graph.Adjacency(graph_mat.tolist(), mode = 'undirected')
        for name in attribute_names:
            G.vs[name] = sorted_df[name]
        self.G = G
        self.graph_mat = self._graph_to_matrix(G)
        self.node_order = reorder_index
    
    def save_graph(self, filename, f='graphml'):
        self.G.save(open(filename,'wb'), format=f)

    def set_visual_style(self, layout ='kk',  labels='auto', plot_adj=True,
                         size=6000):
        """
        layout: str: 'kk', 'circle', 'grid' or other igraph layouts, optional
        Determines how the graph is displayed
        """
        if layout=='circle':
            self.reorder()
        layout_graph = None
        # plot_adj removes negative c.onnections and thresholds to
        # aid layout
        if plot_adj==True:
            # remove negative values
            plot_adj = threshold(self.adj,
                     threshold_func = bct.threshold_absolute,
                     threshold = 0)
            # only allow the top variables to effect the layout
            plot_adj = threshold(plot_adj,
                                 threshold_func = bct.threshold_proportional,
                                 threshold = .2)
            # check if binary graph
            if set(np.unique(plot_adj.values))==set([0,1]):
                layout_graph = igraph.Graph.Adjacency(plot_adj.values.tolist(), mode = 'undirected')
            else:
                layout_graph = igraph.Graph.Weighted_Adjacency(plot_adj.values.tolist(), mode = 'undirected')
        if labels=='auto':
            labels = self.G.vs['id']
        self.visual_style = self.create_visual_style(self.G, layout = layout, 
                                                     layout_graph = layout_graph, 
                                                     vertex_size = 'eigen_centrality', 
                                                     labels = labels,
                                                     size = size)
  
    def _graph_to_matrix(self, G):
        if 'weight' in G.es.attribute_names():
            graph_mat = np.array(G.get_adjacency(attribute = 'weight').data)
        else:
            graph_mat = np.array(G.get_adjacency().data)
        return graph_mat
    
    def _plot_graph(self, G=None, visual_style=None, **kwargs):
        if G==None:
            G=self.G
        if visual_style==None:
            visual_style = self.visual_style
        visual_style.update(**kwargs)
        fig = igraph.plot(G, **visual_style)
        return fig
    
    def _print_community_members(self, G=None, lookup = {}, file = None):
        if G==None:
            G=self.G
        if file:
            f = open(file,'w')
        else:
            f = None
            
        print('Key: Node index, Subgraph Eigen Centrality, Measure, Eigenvector centrality', file = f)
        for community in np.unique(G.vs['community']):
            #find members
            members = [lookup.get(v['name'],v['name']) for v in G.vs if v['community'] == community]
            # ids and total degree
            ids = [v['id'] for v in G.vs if v['community'] == community]
            eigen = ["{0:.2f}".format(v['eigen_centrality']) for v in G.vs if v['community'] == community]
            #sort by within degree
            within_degrees = ["{0:.2f}".format(v['subgraph_eigen_centrality']) for v in G.vs if v['community'] == community]
            to_print = list(zip(ids, within_degrees,  members, eigen))
            to_print.sort(key = lambda x: -float(x[1]))
            
            print('Members of community ' + str(community) + ':', file = f)
            pprint(to_print, stream = f)
            print('', file = f)
        
    def _relabel_community(community, reference):
        ref_lists = [[i for i,c in enumerate(reference) if c==C] for C in np.unique(reference)]
        comm_lists = [[i for i,c in enumerate(community) if c==C] for C in np.unique(community)]
        relabel_dict = {}
        for ci,comm in enumerate(comm_lists):
            best_match = None
            best_count = 0
            for ri,ref in enumerate(ref_lists):
                count = len(set(ref).intersection(comm))
                if count > best_count:
                    best_count = count
                    best_match = ri + 1
            if best_match in relabel_dict.values():
                best_match = max(relabel_dict.values()) + 1
            relabel_dict[ci+1] = best_match
        return [relabel_dict[c] for c in community]
    
    def _subgraph_analysis(self):
        G = self.G
        community_alg = self.community_alg
        assert set(['community','id']) <=  set(G.vs.attribute_names()), \
            'Graph must have "community" and "id" as a vertex attributes'
        for c in np.unique(G.vs['community']):
            subgraph = G.induced_subgraph([v for v in G.vs if v['community'] == c])
            subgraph_mat = self._graph_to_matrix(subgraph)
            if 'weight' in G.es.attribute_names():
                subgraph.vs['eigen_centrality'] = subgraph.eigenvector_centrality(directed = False, weights = subgraph.es['weight'])
            else:
                subgraph.vs['eigen_centrality'] = subgraph.eigenvector_centrality(directed = False)
            G.vs.select(lambda v: v['id'] in subgraph.vs['id'])['subgraph_eigen_centrality'] = subgraph.vs['eigen_centrality']
            if community_alg:
                comm, Q = community_alg(subgraph_mat)
                subgraph.vs['community'] = comm
                G.vs.select(lambda v: v['id'] in subgraph.vs['id'])['subgraph_community'] = subgraph.vs['community']

    
        
