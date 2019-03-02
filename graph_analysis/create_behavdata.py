import sys,os
import random
import pickle
import pandas,numpy
from selfregulation.utils.utils import get_info,get_behav_data
import fancyimpute
import matplotlib.pyplot as plt

dataset='Complete_10-27-2017'
basedir=get_info('base_directory')
nruns=int(sys.argv[1])
if len(sys.argv)>2:
    outdir=sys.argv[2]
else:
    outdir='maxcorr'
if not os.path.exists(outdir):
    os.mkdir(outdir)
datafile=os.path.join(basedir,
        'Data/Derived_Data/%s/behavdata_imputed_cleaned.csv'%dataset)
if os.path.exists(datafile):
    df=pandas.read_csv(datafile)
    print('using existing datafile')
else:
    if not os.path.exists(os.path.dirname(datafile)):
        os.mkdir(os.path.dirname(datafile))
    behavdata=get_behav_data(dataset)
    filter_by_icc=False

    if filter_by_icc:
        icc_threshold=0.25
        icc_boot=pandas.read_csv('../Data/Retest_09-27-2017/bootstrap_merged.csv')
        icc=icc_boot.groupby('dv').mean().icc

        for v in behavdata.columns:
            if icc.loc[v]<icc_threshold:
                del behavdata[v]

    behavdata_imputed=fancyimpute.SoftImpute().fit_transform(behavdata.values)
    df=pandas.DataFrame(behavdata_imputed,columns=behavdata.columns)

    for dropvar in ['kirby_mturk.percent_patient',
                    'kirby_mturk.exp_discount_rate',
                    'kirby_mturk.hyp_discount_rate',
                    'kirby_mturk.exp_discount_rate_medium',
                    'kirby_mturk.exp_discount_rate_small',
                    'kirby_mturk.exp_discount_rate_large',
                    'probabilistic_selection_mturk.positive_learning_bias',
                    'shift_task_mturk.nonperseverative_errors',
                    'stop_signal_mturk.omission_errors',
                    'stop_signal_mturk.SSRT',
                    'stroop_mturk.incongruent_errors']:
            del df[dropvar]

    abcc=df.corr().abs()
    ct=numpy.triu(abcc.values,1)
    nw=numpy.where(ct>=0.90)
    for i in range(len(nw[0])):
        print(df.columns[nw[0][i]],df.columns[nw[1][i]])
    print('saving',datafile)
    df.to_csv(datafile)

plot_heatmap=False
if plot_heatmap:
    plt.figure(figsize=(16,16))
    g=seaborn.clustermap(cc,method='ward')
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation = 0, fontsize = 6)
    plt.savefig('corr_hmap.png')

# shuffle, recompute, and store maximum for each run

def col_shuffle(df,test=False):
    """
    shuffle data within each column
    """
    if test:
        return(df)
    df_shuf=df.copy()
    for i in range(df.shape[1]):
        numpy.random.shuffle(df_shuf.iloc[:,i])
    return(df_shuf)

maxcc=numpy.zeros(nruns)
for i in range(nruns):
    df_tmp=col_shuffle(df)
    abcc_tmp=df_tmp.corr().abs()
    maxcc[i]=numpy.triu(abcc_tmp.values,1).max()
    print(i,maxcc[i])
h='%08x'%random.getrandbits(32)
numpy.savetxt(os.path.join(outdir,'ccrand_max_%s.txt'%h),maxcc)
