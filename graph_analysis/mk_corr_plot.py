import sys,os
import scipy.stats
import random
import pickle
import pandas,numpy,seaborn
from selfregulation.utils.utils import get_info,get_behav_data,get_var_category
import fancyimpute
import matplotlib.pyplot as plt
import networkx

pthresh=0.1
dataset='Complete_10-27-2017'
basedir=get_info('base_directory')
outdir='maxcorr'
if not os.path.exists(outdir):
    os.mkdir(outdir)
datafile=os.path.join(basedir,
        'Data/Derived_Data/%s/behavdata_imputed_cleaned.csv'%dataset)
if os.path.exists(datafile):
    df=pandas.read_csv(datafile,index_col=0)
    print('using existing datafile')

abcc=df.corr().abs()
ccrand=numpy.loadtxt('ccrand_max.txt')
thresh=scipy.stats.scoreatpercentile(ccrand,100*(1-pthresh))
print('p<%0.3f: r>%0.3f'%(pthresh,thresh))
abcc[abcc<thresh]=0
abcc=numpy.triu(abcc,1)

G=networkx.from_numpy_matrix(abcc).to_undirected()
for i in G.nodes():
    G.node[i]['name']=df.columns[i]
    G.node[i]['type']=get_var_category(df.columns[i])

degree=G.degree()
for i in degree.keys():
    if degree[i]<1:
        G.remove_node(i)
# drop components with only two members
remove_nodes=[]
for sg in networkx.connected_components(G):
    if len(sg)<3:
        for i in sg:
            remove_nodes.append(i)
for i in remove_nodes:
    G.remove_node(i)

networkx.write_graphml(G,'corr_thresh%0.3f.graphml'%thresh)
