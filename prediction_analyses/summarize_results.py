import os,glob,sys
import pickle
import numpy
import scipy.stats
from statsmodels.sandbox.stats.multicomp import multipletests
analysis=sys.argv[1]
assert analysis in ['prediction','regression']


datadir='%s_outputs'%analysis
pvals={}
pvals_fdr={}
outfile=open("%s_results.txt"%analysis,'w')
for clf in ['lasso','forest']:
    if not clf in pvals:
        pvals[clf]={}
        pvals_fdr[clf]={}
    for ds in ['all','survey','task']:
        if not ds in pvals[clf]:
            pvals[clf][ds]={}
            pvals_fdr[clf][ds]={}
        # load data
        if os.path.exists('%s_%s_%s_data.pkl'%(analysis,clf,ds)):
            truedata,permuted=pickle.load(open('%s_%s_%s_data.pkl'%(analysis,clf,ds),'rb'))
        else:
            allfiles=glob.glob(os.path.join(datadir,'prediction_%s_%s*.pkl'%(ds,clf)))
            if len(allfiles)==0:
                continue
            print(clf,ds,'found %d files'%len(allfiles))
            true=[i for i in allfiles if not i.find('shuffle')>-1]
            shuf=[i for i in allfiles if i.find('shuffle')>-1]
            #print(true,shuf)
            truedata=pickle.load(open(true[0],'rb'))
            permuted={}
            for k in truedata[0].keys():
                permuted[k]=[]
            for f in shuf:
                tmp=pickle.load(open(f,'rb'))
                for k in tmp[0]:
                    permuted[k].append(tmp[0][k])
            pickle.dump((truedata,permuted),open('%s_%s_%s_data.pkl'%(analysis,clf,ds),'wb'))

        # put into a matrix
        keys=list(truedata[0].keys())
        keys.sort()

        permuted_data=numpy.zeros((len(keys),len(permuted[list(permuted.keys())[0]])))

        for i,k in enumerate(keys):
            permuted_data[i,:]=permuted[list(permuted.keys())[0]]
        # get individual variable pvals
        pvals[clf][ds]=[]
        for i in range(permuted_data.shape[0]):
            data_nonan=[x for x in permuted_data[i,:] if not numpy.isnan(x)]
            pvals[clf][ds].append((100-scipy.stats.percentileofscore(data_nonan,truedata[0][keys[i]]))/100.)
        _,pvals_fdr[clf][ds],_,_=multipletests(pvals[clf][ds],method='fdr_bh')
        for i in range(permuted_data.shape[0]):
            if pvals_fdr[clf][ds][i]<0.05:
                outfile.write(' '.join([clf,ds,keys[i],'%f'%truedata[0][keys[i]],'%f'%pvals_fdr[clf][ds][i]])+'\n')
        # FWE max cutoff
        perm_max=numpy.nanmax(permuted_data,1)
        cutoff=scipy.stats.scoreatpercentile(perm_max,95)
        print(analysis,clf,ds,'cutoff=%0.3f'%cutoff)
outfile.close()
