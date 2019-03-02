# find the best fitting link function for each variable
# pick the one with minimum AIC

library(pscl)
demogdata=read.table('../Data/Derived_Data/Complete_10-08-2017/demogdata_for_prediction.csv',
                     sep=',',header=TRUE)

models=c('gaussian','poisson','NB','ZIpoisson','ZINB')
demogdata$X=NULL
varnames=names(demogdata)
varinfo=c()
for (varname in varnames){
  
  ydata=demogdata[,varname]
  ydata=ydata[!is.na(ydata)]
  if (length(unique(ydata))==2) {
    print(varname)
    print('binary')
    varinfo=rbind(varinfo,c(varname,'binary'))
    
    next
  }
  pois=tryCatch({
    modelPoisson <- glm(formula = ydata ~ 1,
                        family  = poisson(link = "log"))  
    AIC(modelPoisson)
    }, error = function(e) {
      Inf
    })
  gaus=tryCatch({
    modelGaussian <- glm(formula = ydata ~ 1,
                         family  = gaussian)  
    AIC(modelGaussian)
    }, error = function(e) {
      Inf
    })
  nb=tryCatch({
    modelNB <- glm.nb(formula = ydata ~ 1)
    AIC(modelNB)
  }, error = function(e) {
    Inf
  })
  # for datasets with at least 10% zeros, try zero-inflated models
  if (mean(ydata==0)>0.1){
  
    zi_nb=tryCatch({
      modelZINB <- zeroinfl(formula = ydata ~ 1,
                            dist    = "negbin")
      AIC(modelZINB)
    }, error = function(e) {
      Inf
    })
    
    zi_pois=tryCatch({
      modelZIpoisson <- zeroinfl(formula = ydata ~ 1,
                                 dist    = "poisson")
      AIC(modelZIpoisson)
    }, error = function(e) {
      Inf
    })
  } else {
    zi_pois=Inf
    zi_nb=Inf
  } 
  aicvals=c(gaus,pois,nb,zi_pois,zi_nb)
  
  print(varname)
  print(aicvals)
  print(models[which(aicvals==min(aicvals))])
  varinfo=rbind(varinfo,c(varname,models[which(aicvals==min(aicvals))]))
}

write.table(varinfo,file='demographic_model_type.txt',quote=FALSE,
            sep='\t',row.names=FALSE,col.names=FALSE)
