library(mirt)
library(parallel)
mirtCluster(23)

args <- commandArgs(TRUE)

if (length(args)<1) {
ncomponents=2
} else {
  ncomponents=as.integer(args[1])
}

d=read.csv('../../Data/Derived_Data/Discovery_9-26-16/surveydata_fixed_minfreq20.csv')
d$worker=NULL
nsubs=dim(d)[1]
nfolds=2
for (runnum in 1:10){

# need to find balanced folds that have the same response options
ctr=1
good_cv=FALSE
while (!good_cv) {
  cvidx=kronecker(array(1,nsubs),c(1:nfolds))
  cvidx=sample(cvidx[1:nsubs])
  respcats=apply(d[cvidx==1,], 2, function(x) length(unique(x)))
  good_cv=TRUE
  for (i in 2:nfolds){
    tstresp=apply(d[cvidx==i,], 2, function(x) length(unique(x)))
    if (sum(tstresp==respcats)!=length(respcats)) {good_cv=FALSE}
  }
  ctr=ctr+1
  if (ctr%%100 == 0) {cat(sprintf('try %d\n',ctr))}
}
cat('found good split')

ll=array(0,nfolds)
for (cvfold in 1:nfolds){
  train_data=d[cvidx!=cvfold,]
  test_data=d[cvidx==cvfold,]
  m=mirt(train_data,ncomponents,SE=TRUE,TOL=0.01,technical=list(MHRM_SE_draws=5000,MAXQUAD=100000,NCYCLES=10000),verbose=verbosearg,method='MHRM',itemtype=modeltype)

  sv <- mod2values(m)
  sv$est <- FALSE #fix starting values to last model
  mod2 <- mirt(test_data, ncomponents, pars = sv)
  ll[cvfold]=logLik(mod2)
}
write(ll,file=sprintf('cvdata/ll_%03d_%d.txt',ncomponents,runnum))
}
