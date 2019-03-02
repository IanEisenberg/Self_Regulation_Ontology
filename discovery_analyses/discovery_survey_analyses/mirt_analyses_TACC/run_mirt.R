library(mirt)
library(parallel)
mirtCluster(23)
args <- commandArgs(TRUE)

if (length(args)<1) {
ncomps=2
} else {
  ncomps=as.integer(args[1])
}


d=read.csv('../../Data/Derived_Data/Discovery_9-26-16/surveydata_fixed_minfreq20.csv')
d$worker=NULL
verbosearg=FALSE
modeltype='graded'
m=mirt(d,ncomps,SE=TRUE,TOL=0.01,technical=list(MHRM_SE_draws=5000,MAXQUAD=100000,NCYCLES=10000),verbose=verbosearg,method='MHRM',itemtype=modeltype)

save(m,file=sprintf('mirt_%ddims_%s.Rdata',ncomps,modeltype))
