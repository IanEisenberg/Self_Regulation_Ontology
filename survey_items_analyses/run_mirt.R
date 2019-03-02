library(mirt)
library(parallel)
mirtCluster(19)
args <- commandArgs(TRUE)

if (length(args)<1) {
ncomps=2
} else {
  ncomps=as.integer(args[1])
}


d=read.csv('../Data/Derived_Data/Combined_12-15-2016/surveydata_fixed_minfreq40.csv')
d$worker=NULL
verbosearg=FALSE
modeltype='graded'
m=mirt(d,ncomps,SE=TRUE,technical=list(MHRM_SE_draws=5000,MAXQUAD=100000,NCYCLES=10000),verbose=verbosearg,method='MHRM',itemtype=modeltype)

save(m,file=sprintf('output/mirt_%ddims_%s.Rdata',ncomps,modeltype))
