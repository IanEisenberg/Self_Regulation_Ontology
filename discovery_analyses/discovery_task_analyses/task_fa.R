# FA on behav data
library(psych)
library(softImpute)
library(missForest)

source("http://goo.gl/UUyEzD") # outlierKD



file='/Users/poldrack/code/Self_Regulation_Ontology/Data/Discovery_10-14-2016/meaningful_variables.csv'
d=read.table(file,sep=',',header=TRUE)
worker=d$X
d$X=NULL


taskdata=d
surveydata=d
for (n in colnames(d)) {
  if (length(grep('survey',n))>0){
    taskdata[n]=NULL
  } else {
    surveydata[n]=NULL
  }
}

# remove bad variables

taskdata=subset(taskdata,select=-c(kirby.hyp_discount_rate,kirby.percent_patient,
                                         discount_titrate.hyp_discount_rate_glm  ,                  
                                         discount_titrate.hyp_discount_rate_nm ,                    
                                         discount_titrate.percent_patient ))

# get names of tasks
tasknames=c()
for (n in colnames(taskdata)){
  tasknames=c(tasknames,unlist(strsplit(n, "[.]"))[1])
}
tasks=unique(tasknames)

# set outliers to NA
# use a conservative cutoff of 3*IQR
remove_outliers <- function(x, na.rm = TRUE, cutoff=3,...) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H <- cutoff * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  y
}

taskdata_clean=apply(taskdata,2,remove_outliers)


# make pair panel plots for uncleaned data
pdf('panel_plots_uncleaned.pdf')
for (n in tasks){
  taskcols=n==tasknames
  if (sum(taskcols)>1) {
    pairs.panels(taskdata[,taskcols])
  }
}
dev.off()

# impute missing values for task data
mf=missForest(taskdata)
cat(sprintf('Imputation error: %f',mf$OOBerror))
taskimp=mf$ximp
row.names(taskimp)=worker

write.table(taskimp,'../Data/Derived_data/Discovery_10-14-2016/taskdata_imputed.csv',
            sep=',',quote=FALSE)
# make pair panel plots
pdf('panel_plots_cleaned_imputed.pdf')
for (n in tasks){
  taskcols=n==tasknames
  if (sum(taskcols)>1) {
    pairs.panels(taskimp[,taskcols])
  }
}
dev.off()

# remove weakly correlated variables
c=cor(taskimp)
c[c==1]=0
m=apply(c,1,max)
corthresh=0.33
taskimp_reduced=taskimp[,m>=corthresh]
cat(sprintf('dropping %d variables with max correlation below %f',
            sum(m<corthresh),corthresh))



# do factor analysis
for (ndims in 1:8) {
  print('')
  print(sprintf('FA: %d dimensions',ndims))
  fa.result=fa(taskimp_reduced,ndims,fm='ml')
  vars=names(taskimp_reduced)
  for (i in 1:ndims){
    print(sprintf('dimension %d',i))
    o=order(fa.result$loadings[,i],decreasing=TRUE)
    for (j in 1:3){
      print(sprintf('%s: %f',vars[o[j]],fa.result$loadings[o[j],i]))
    }
    o=order(fa.result$loadings[,i])
    for (j in 3:1){
      print(sprintf('%s: %f',vars[o[j]],fa.result$loadings[o[j],i]))
    }
    
    print('')
  }
}


#  do CCA
do_cca=FALSE
if (do_cca) {
  library(PMA)
  tasknames=colnames(taskdata)
  surveynames=colnames(surveydata)

  # impute missing values for survey data
  mf=missForest(surveydata)
  cat(sprintf('Survey data imputation error: %f',mf$OOBerror))
  surveyimp=mf$ximp
  
  pmt=CCA.permute(taskimp,surveyimp)
  cca.result=CCA(taskimp,surveyimp,penaltyx=0.15,
                 penaltyz = 0.2,K=5,niter=50,
                 typex='standard',typez='standard')
  
  for (k in 1:5){
    print(k)
    print('tasks')
    print(tasknames[cca.result$u[,k]!=0])
    print('surveys')
    print(surveynames[cca.result$v[,k]!=0])
    print('')
  }

}


