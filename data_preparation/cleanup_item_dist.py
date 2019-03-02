#!/usr/bin/env python3
"""
function to clean up item distribution for surveys
collapsing any categories that do not have sufficient responses
"""

import numpy,sys

def get_respdist(f):
    f=f.dropna()
    u=f.unique()
    u.sort()
    h=[numpy.sum(f==i) for i in u]
    return u,h

def cleanup_item_dist(c,data,minresp=4,verbose=False,
                        drop_bad_middle=False):

    d=data[c].copy()

    u,h=get_respdist(d)
    if verbose:
        print(u)
        print(h)
    u=numpy.array(u)
    h=numpy.array(h)

    csl=numpy.cumsum(h)
    csr=numpy.cumsum(h[::-1])

    badlocs=numpy.where(h<minresp)[0]

    if len(badlocs)==0:
        if verbose:
            print('no bad locations found for',c)
        return d,False
    else:
        if verbose:
            print('bad locations for %s'%c,badlocs)

    if len(u)==2:
        # if it's dichotomous and has a bad option, then drop it
        if verbose:
            print('dropping dichotomous variable:',c)
        return d,True

    if numpy.sum(h<minresp)>0:
        left_cutoff=u[numpy.min(numpy.where(h>minresp))]
        left_replace=u[u<left_cutoff]
        left_replacement=u[u==left_cutoff][0]
        if verbose:
            print('replacing left responses:',left_replace)
        d[d<left_cutoff]=left_replacement
    else:
        if verbose:
            print('no bad left responses')
        left_replace=[]

    if numpy.sum(h<minresp)>0:
        right_cutoff=u[numpy.max(numpy.where(h>minresp))]
        right_replace=u[u>right_cutoff]
        right_replacement=u[u==right_cutoff][0]
        d[d>right_cutoff]=right_replacement
        if verbose:
            print('replacing right responses:',right_replace)
    else:
        if verbose:
            print('no bad right responses')
        right_replace=[]

    unew,hnew=get_respdist(d)

    if len(numpy.where(numpy.array(hnew)<minresp)[0])>0:
        # bad middle category
        if verbose:
            print('dropping responses in bad middle category:',c)
        if drop_bad_middle:
            return d,True

    return d,False
