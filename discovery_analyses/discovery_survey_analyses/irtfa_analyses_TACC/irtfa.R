library(psych)
args <- commandArgs(TRUE)

if (length(args)<1) {
  ncomps=4
} else {
  ncomps=as.integer(args[1])
}


d=read.csv('../../Data/Derived_Data/Discovery_9-26-16/surveydata.csv')
d$worker=NULL
d[,363]=NULL # drop row with non-integer values
d$X=NULL
itrfa.result=irt.fa(d,nfactors=ncomps,plot=FALSE)

save(irtfa.result,file=sprintf('irtfa_%ddims.Rdata',ncomps))
