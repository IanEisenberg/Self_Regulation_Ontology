#  run canonical correlation on task/survey data

library(CCA)

source ('../utils/utils.R')

dataset=get_info('dataset')
basedir=get_info('base_directory')
taskdata=get_task_data('Complete_12-15-2016')
taskvars=names(taskdata)
# get names of tasks
tasknames=c()
for (n in colnames(taskdata)){
  tasknames=c(tasknames,unlist(strsplit(n, "[.]"))[1])
}
tasks=unique(tasknames)
surveydata=read.table('../Data/Complete_12-15-2016/meaningful_variables_imputed_for_task_selection.csv',sep=',',header=TRUE)
surveynames=c()
for (n in names(surveydata)) {
  if (grepl('survey',n)){
    surveynames=c(surveynames,n)
  }
}

surveydata=subset(surveydata,select=surveynames)
surveydata=subset(surveydata,select=c(-cognitive_reflection_survey.correct_proportion,-cognitive_reflection_survey.intuitive_proportion,-holt_laury_survey.safe_choices))
cca.result=cc(taskdata,surveydata)

nruns=1000
corshuf=matrix(NA,nrow=length(cca.result$cor),ncol=nruns)

for (i in 1:nruns){
  print(i)
  surveydata_shuf=surveydata[sample(dim(surveydata)[1]),]
  cca.result.shuffle=cc(taskdata,surveydata_shuf)
  corshuf[,i]=cca.result.shuffle$cor
  
}

pvals=array(NA,dim=dim(corshuf)[1])
for (i in 1:dim(corshuf)[1]){
  pvals[i]=mean(corshuf[i,]>cca.result$cor[i])
  print(sprintf('dim %d: p = %0.3f',i,pvals[i]))
  
  
}
