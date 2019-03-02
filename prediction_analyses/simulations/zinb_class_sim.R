# simulate zero-inflated NB data
# compare ZINB regression to dichotomization + classification

library("mpath")
library("zic")
library("pscl")
library(glmnet)

args <- commandArgs(trailingOnly = TRUE)
require(doMC)
ncores=4
registerDoMC(cores=ncores)

rand_zinb=function(X,b1=2,b2=4,nb_size=4,nb_prob=0.03){
  pcount=0
  # ensure at least 2.5% nonzero responses
  #print('making data...')
  while (sum(pcount)<(dim(X)[1]*0.025)){
    #print('trying...')
    beta1=array(0,dim=dim(X)[2])
    beta2=array(0,dim=dim(X)[2])
    beta1[1:2]=b1
    beta2[3:4]=b2
    a1 =X%*%beta1
    a2 =X%*%beta2
    pzero = exp(a2)/(1+exp(a2))
    pcount = round(exp(a1)*(1-pzero)) + rnbinom(dim(X)[1],nb_size,nb_prob)
    pcount[pcount<0]=0
  }
  #print(sprintf("mean nonzero: %f",mean(round(pcount)>0)))
  return(pcount)
}

b1=as.numeric(args[1])
shuffle=as.numeric(args[2])
#b1=3.0
#shuffle=FALSE
nruns=1000
nobs=500
nvars=20
zinbcor=array(NA,dim=nruns)
bincor=array(NA,dim=nruns)
bincor_count=array(NA,dim=nruns)
regcor_count=array(NA,dim=nruns)
pcount_nonzero_sum=array(NA,dim=nruns)

for (r in 1:nruns) {
  #print(sprintf('run %d',r))
  X<-matrix(runif(nobs*nvars), ncol=nvars) 
  
  pcount=rand_zinb(X,b1=b1)
  if (shuffle) {pcount=pcount[sample(length(pcount))]}
  
  nfolds=4
  nsets=ceiling(dim(X)[1]/nfolds)
  # ensure cross validation folds have positive examples

  fold=kronecker(rep(1,nsets),seq(1,nfolds))[1:dim(X)[1]]
  goodfolds=0
  while (!goodfolds){
    goodfolds=1
    fold=sample(fold)
    for (f in 1:nfolds){
      if (sum(pcount[fold==f])<1){
        goodfolds=0
        print("bad fold")
      }
    }
  }
  pred_count_zinb=array(NA,dim=dim(X)[1])
  pred_bin=array(NA,dim=dim(X)[1])
  pred_resp_lassobin=array(NA,dim=dim(X)[1])
  pred_resp_lassoreg=array(NA,dim=dim(X)[1])
  
  for (f in 1:nfolds) {
    d=as.data.frame(X[fold!=f,])
    d$y=pcount[fold!=f]
    
    # z-inflated negative binomial model on count data
    count_cv=cv.zipath(y~.|.,d,family='negbin',nlambda=20,n.cores=1,,plot.it=FALSE)
    z=zipath(y~.|.,d,family='negbin',
              lambda.count = count_cv$lambda.optim$count,
             lambda.zero = count_cv$lambda.optim$zero)
    pred_count_zinb[fold==f]=predict(z,as.data.frame(X[fold==f,]),type='response')
    
    # lasso logistic regression fit to dichotomized data
    l=cv.glmnet(X[fold!=f,],as.integer(pcount[fold!=f]>0),parallel=TRUE,type.measure="auc",family='binomial')
    pred_bin[fold==f]=as.integer(predict(l,newx=X[fold==f,], s="lambda.min",type='class'))
    # use weights from model on binary data to predict counts
    pred_resp_lassobin[fold==f]=predict(l,newx=X[fold==f,], s="lambda.min",type='response')
    
    # lasso logistic regression fit to count data (i.e. misspecified model)
    l=cv.glmnet(X[fold!=f,],pcount[fold!=f],family='gaussian',parallel=TRUE)
    # use weights from model on binary data to predict counts
    pred_resp_lassoreg[fold==f]=predict(l,newx=X[fold==f,], s="lambda.min",type='response')
    
  }
  zinbcor[r]=cor(pred_count_zinb,pcount)
  bincor[r]=cor(pred_bin,as.integer(pcount>0))
  bincor_count[r]=cor(pred_resp_lassobin,pcount)
  regcor_count[r]=cor(pred_resp_lassoreg,pcount)
  pcount_nonzero_sum[r]=sum(pcount>0)
  print(sprintf('%d %f %d %d %f %f %f %f',shuffle,b1,r,pcount_nonzero_sum[r],zinbcor[r],bincor[r],bincor_count[r],regcor_count[r]))
}
if (shuffle) {
  save(zinbcor,bincor,bincor_count,regcor_count,file=sprintf('shuf_%f.Rdata',b1))
} else {
  save(zinbcor,bincor,bincor_count,regcor_count,file=sprintf('noshuf_%f.Rdata',b1))
}
