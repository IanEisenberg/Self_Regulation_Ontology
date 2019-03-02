#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

load scores for different dimensionalities and see how they relate
across levels

Created on Sat Sep  3 15:22:55 2016

@author: poldrack
"""

import numpy,pandas
import networkx as nx


compdesc={}
compdesc[1]=['persistence']
compdesc[2]=['persistence','risktaking']
compdesc[3]=['persistence','physical risk','impulsivity']
compdesc[4]=['persistence','physical risk','impulsivity','internalizing']
compdesc[5]=['manic impulsivity','physical risk','impulsivity','internalizing',
            'persistence']
compdesc[6]=['impulsivity','physical risk','life impulsivity',
            'internalizing','persistence','planning/sentimentality']
compdesc[8]=['manic impulsivity','physical risk','dysfunctional risk',
            'internalizing','persistence','unknown','life impulsivity',
            'mindfulness']

compnums=[1,2,3,4,5,6,8]
scores={}
for ncomps in compnums:
    scores[ncomps]=pandas.read_csv('mirt_scores_%ddims.tsv'%ncomps,
                        delimiter='\t',header=None)

corrs={}
g=nx.DiGraph()

for i in range(len(compnums)-1):
    s1=scores[compnums[i]]
    for j in range(i+1,i+2):
        s2=scores[compnums[j]]
        corrs[(i+1,j+1)]=numpy.zeros((compnums[i],compnums[j]))
        for x in range(compnums[i]):
            xnode='%d-%x'%(compnums[i],x+1)
            if not xnode in g.nodes():
                g.add_node(xnode)
                g.node[xnode]['level']=compnums[i]
            g.node[xnode]['desc']=xnode+':'+compdesc[compnums[i]][x]
            for y in range(compnums[j]):
                ynode='%d-%x'%(compnums[j],y+1)
                if not ynode in g.nodes():
                    g.add_node(ynode)
                    g.node[ynode]['level']=compnums[j]
                g.node[ynode]['desc']=ynode+':'+compdesc[compnums[j]][y]
                corrs[(i+1,j+1)][x,y]=numpy.corrcoef(s1.loc[:,x+1],s2.loc[:,y+1])[0,1]
            matchnum=numpy.argmax(numpy.abs(corrs[(i+1,j+1)][x,:]))
            g.add_edge(xnode,'%d-%x'%(compnums[j],matchnum+1))
            print('%d-comp%d: matches %d-comp%d (%0.3f)'%(compnums[i],x+1,compnums[j],matchnum+1,corrs[(i+1,j+1)][x,matchnum]))
                
        
# for nodes with no inputs, find  largest input and add link

indegree=g.in_degree()
for k in indegree.keys():
    if indegree[k]>0 or k.split('-')[0]=='1':
        continue
    i,j=[int(x) for x in k.split('-')]
    try:
        c=corrs[(i-1,i)]

    except:
        i=i-1
        c=corrs[(i-1,i)]
    crow=c[:,i-1]
    matchnum=numpy.argmax(numpy.abs(crow))
    newedge='%d-%d'%(i-1,matchnum+1)
    g.add_edge(newedge,k)
    #print(k,i,j,matchnum,newedge,crow)
    

        
    

shells=[]
for i in compnums:
    s=[]
    for n in g.nodes():
        if int(n.split('-')[0])==i:
            s.append(n)
    shells.append(s)

    
nx.draw(g,pos=nx.shell_layout(g,shells))
nx.write_graphml(g,'relations.graphml')