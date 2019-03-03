pkgTest <- function(x)
  {
    if (!require(x,character.only = TRUE))
    {
      install.packages(x,dep=TRUE,repos="http://cran.rstudio.com")
    }
    if (!require(x,character.only = TRUE))
    {
        source("http://bioconductor.org/biocLite.R")
        biocLite(x)
    }
        if(!require(x,character.only = TRUE)) stop("Package not available from CRAN or bioconductor - something must be wrong")
  }

pkgTest('e1071')
pkgTest('caret')
pkgTest('jsonlite')
library(e1071)
library(caret)
library(jsonlite)
library(pscl)

print_confusion_matrix <- function(y_true, y_pred){
  return(caret::confusionMatrix(y_pred, y_true))
}

#Usage: get_behav_data('Discovery_09-26-2016')
#Note difference in date notation!
#Returns df with dv's for tasks (as opposed to surveys)
#use_EZ=FALSE reads in data file without diffusion contrasts
get_behav_data <- function(dataset, use_EZ=TRUE){
  basedir <- get_info('base_directory')
  if(use_EZ==T){
    datafile <- paste0(basedir, 'Data/', dataset, '/meaningful_variables.csv')
  }
  else {
    datafile <- paste0(basedir, 'Data/', dataset, '/meaningful_variables_noEZ_contrasts.csv')
  }
  d <- read.csv(datafile, row.names=1)
  return(d)
}

# get the cleaned up task data
get_task_data <- function(dataset){
  basedir <- get_info('base_directory')
  datafile <- paste0(basedir, 'Data/', dataset, '/taskdata_imputed_for_task_selection.csv')
  d <- read.csv(datafile, row.names=1,sep=',',header=TRUE)
  return(d)
}

#Usage: get_info('base_directory')
#Relative path assumes you are in your /Data directory
#item can be any field in Self_Regulation_Settings.txt file
get_info <- function(item,infile='../Self_Regulation_Settings.txt'){

  if(file.exists(infile)){
    infodict <- suppressWarnings(read.table(infile, sep = ":"))
    infodict$V2 <- gsub(" ", "", infodict$V2)
    if(item %in% infodict$V1){
      return(as.character(infodict[infodict$V1 == item, "V2"]))
    }
    else {
      print('infodict does not include requested item')
    }

  }
  else{
    print('You must first create a Self_Regulation_Settings.txt file')
  }
}

#Usage: dataset <- get_single_dataset('Discovery_9-26-16','alcohol_drugs')
#dataset[[1]] for survey data
#dataset[[2]] for metadata
#To view metadata in dataset[[2]]
#try as.data.frame(dataset[[2]]) for a flattened dataframe
#or to work with list dataset[[2]]$WidthdrawalSymptoms
get_single_dataset <- function(dataset, survey){
  base_dir <- get_info('base_directory')
  infile <- paste0(base_dir, "data/Derived_Data/", dataset, "/surveydata/", survey,".tsv")
  mdfile <- paste0("data/Derived_Data/", dataset, "/metadata/", survey,".json")
  data <- read.csv(infile, row.names=1, sep='\t')
  metadata <- load_metadata(mdfile, base_dir)
  return(list(data, metadata))
}

#Usage: survey_data <- get_survey_data('Discovery_9-26-16')
#survey_data[[1]] for surveydata.csv (response file)
#survey_data[[2]] for surveyitem_key.txt
get_survey_data <- function(dataset){
  basedir<- get_info('base_directory')
  d=get_behav_data(dataset)
  surveydata=d
  for (n in colnames(surveydata)) {
    if (length(grep('survey',n))<1){
      surveydata[n]=NULL
    }
  }
  keyfile <- paste0(basedir, "data/Derived_Data/", dataset, "/surveyitem_key.txt")
  surveykey <- read.table(keyfile, sep='\t')
  return (list(surveydata,surveykey))
}

#Usage: md <- load_metadata('data/Derived_Data/Discovery_09-26-16/metadata/alcohol_drugs.json', '/Users/zeynepenkavi/Dropbox/PoldrackLab/Self_Regulation_Ontology/Data/')
#Output is list
#Access json fields by md$json_field_name
load_metadata <- function(variable, basedir){
  outfile <- paste0(basedir, variable)
  metadata <- fromJSON(outfile, simplifyVector = FALSE)
  return(metadata)
}



#Usage: varinfo <- get_vartypes(demographics, True)
#Output is dataframe
get_vartypes = function(demogdata, verbose=TRUE) {
        
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
      
      if (verbose == TRUE) {
          print(varname)
          print(mean(ydata==0))
          print(aicvals)
          print(models[which(aicvals==min(aicvals))])
      }
      varinfo=rbind(varinfo,c(varname,models[which(aicvals==min(aicvals))]))
    }
    return(varinfo)
}