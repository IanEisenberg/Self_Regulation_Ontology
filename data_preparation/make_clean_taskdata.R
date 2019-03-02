#  make cleaned data files

source ('../utils/utils.R')

dataset=get_info('dataset')
d=get_behav_data(dataset,use_EZ = TRUE)
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

taskdata=subset(taskdata,select=-c(kirby.hyp_discount_rate,
                                   discount_titrate.hyp_discount_rate_glm  ,                  
                                   discount_titrate.hyp_discount_rate_nm ,                    
                                   angling_risk_task_always_sunny.keep_adjusted_clicks,
                                   angling_risk_task_always_sunny.release_adjusted_clicks))
rownames(taskdata)=worker

# set outliers to NA
# use a conservative cutoff of 3*IQR
for (cutoff in c(2,2.5,3)) {
  remove_outliers <- function(x, na.rm = TRUE,...) {
    qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
    H <- cutoff * IQR(x, na.rm = na.rm)
    y <- x
    y[x < (qnt[1] - H)] <- NA
    y[x > (qnt[2] + H)] <- NA
    y
  }

  taskdata_clean=apply(taskdata,2,remove_outliers)
  cat(sprintf('removed %d outlier observations',sum(is.na(taskdata_clean))-sum(is.na(taskdata))))
  outfile=sprintf('/Users/poldrack/code/Self_Regulation_Ontology/Data/%s/taskdata_clean_cutoff%0.2fIQR.csv',
                  dataset,cutoff)
  write.table(taskdata_clean,outfile,
              sep=',',quote=FALSE)
  
}
