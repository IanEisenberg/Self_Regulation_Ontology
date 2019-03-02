# summarize results from mirt analysis
library(mirt)
compnums=c(1,2,3,4,5,6,8,9,10)
converged=array(dim=length(compnums))
for (i in 1:length(compnums)) {
  ncomps=compnums[i]
  load(sprintf('mirt_%ddims.Rdata',ncomps))
  converged[i]=m@OptimInfo$converged
  s=summary(m,verbose=FALSE)
  scores=s$rotF
  write.table(scores,file=sprintf('mirt_scores_%ddims.tsv',ncomps),sep='\t',quote=FALSE,col.names=FALSE)
  vnames=read.csv('variable_key.txt',sep='\t',header=FALSE)
  
  for (i in 1:ncomps) {
    s=sort(scores[,i])
    sd=sort(scores[,i],decreasing=TRUE)
    n=names(s)
    nd=names(sd)
    cat('\n')
    cat(sprintf('ncomps=%d,component %d\n',ncomps,i))
    for (j in 1:3){
      
      cat(n[j],as.character(vnames[as.character(vnames$V1)==n[j],]$V2), "\n")
    }
    for (j in 1:3){
      
      cat(nd[j],as.character(vnames[as.character(vnames$V1)==nd[j],]$V2), "\n")
    }
    cat('\n')
  }
  
}