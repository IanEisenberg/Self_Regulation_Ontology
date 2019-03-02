"""
utility functions for jupyter notebook
on prediction results
"""

import pandas,numpy
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

def get_pval(target,null,allvars,datasets,acc):
    data=[]
    vars=list(allvars.keys())
    vars.sort()
    for v in vars:
        #print(target,null,v)
        if not v in acc[target][allvars[v]]['scores_cv'] or not v in acc[null][allvars[v]]['scores_cv']:
            data.append([allvars[v],numpy.nan,numpy.nan,numpy.nan,numpy.nan,numpy.nan])
            continue
        targdist=acc[target][allvars[v]]['scores_cv'][v].dropna()
        targmean=targdist.mean()
        nulldist=acc[null][allvars[v]]['scores_cv'][v].dropna()
        nullmean=nulldist.mean()
        targstd=targdist.std()
        pval=1-scipy.stats.percentileofscore(nulldist,targmean)/100.
        if targstd>0:
            #es=(targmean-nullmean)/targstd
            es=targmean-nullmean
        else:
            es=numpy.nan
        insample=acc[target][allvars[v]]['scores_insample'][v].mean()
        data.append([allvars[v],targmean,nullmean,es,insample,pval])
    df=pandas.DataFrame(data,index=vars,columns=['Measure','Target mean','Null Mean','Effect size','In-sample','p_unc'])
    return(df)


def get_importances(v,dt,features,nfeats=3):
    if not v in features[dt]:
        print(v,'is not in features for',dt)
        return None

    #print(dt,'importances for:',v)
    imp=pandas.DataFrame({'importance':(features[dt][v].abs()>0).mean(0)})
    imp['mean']=features[dt][v].mean(0)
    imp=imp.sort_values(by=['importance','mean'],ascending=False)

    if nfeats>(imp.shape[0]):
        nfeats=imp.shape[0]
    topfeats=imp.iloc[:nfeats]
    topfeats=topfeats.query('importance>0')
    return topfeats

def get_importance_list(sigp,dt,features):
    implist=[]
    for v in sigp.index:
        i=get_importances(v,dt,features)
        implist.append([list(i.index)])
    df=pandas.DataFrame({'top features':implist})
    df.index=sigp.index
    return df

# plot var for all datasets
def plotvars(v,pvals,datasets,allvars,plotcutoff=True,
            plotbaseline=False):

    df=[]
    errors=[]
    ds=[]
    for k in datasets:
        if not allvars[v] in acc[k]:
            continue
        if not v in acc[k][allvars[v]]['scores_cv']:
            continue
        targdist=acc[k][allvars[v]]['scores_cv'][v].dropna()
        df.append(targdist.mean())
        ds.append(k)
        errors.append(targdist.std())
    df=pandas.DataFrame({'mean':df},index=ds)
    errors=pandas.DataFrame({'mean':errors},index=ds)
    if allvars[v]=='AUROC':
        df.plot.bar(yerr=errors,legend=False,ylim=(0.45,numpy.max(df.values)*1.1))
    else:
        df.plot.bar(yerr=errors,legend=False)
    plt.title(v)
    plt.ylabel(allvars[v]+' +/- SE across CV runs')
    if plotcutoff:
        cutoff=acc['baseline'][allvars[v]]['scores_cv'][v].dropna().quantile(0.95)
        plt.plot([0,1000],[cutoff,cutoff],'k--',linewidth=0.5)
    if plotbaseline:
        baseline=acc['baseline'][allvars[v]]['scores_cv'][v].dropna().mean()
        plt.plot([0,1000],[baseline,baseline],'k--',linewidth=0.5)
