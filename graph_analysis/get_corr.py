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
outdir=sys.argv[2]
datafile=sys.argv[3]

if not os.path.exists(outdir):
    os.mkdir(outdir)
#datafile=os.path.join(basedir,
#        'Data/Derived_Data/%s/behavdata_imputed_cleaned.csv'%dataset)

df=pandas.read_csv(datafile)

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
