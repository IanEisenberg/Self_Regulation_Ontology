#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

path <- args[1]
t1_df <- args[2]
t2_df <- args[3]
var_list <- args[4]
output_dir <- args[5]
n <- as.numeric(args[6])
job_num <- as.numeric(args[7])

#load packages
library(tidyverse)
library(RCurl)
library(psych)

from_gh = TRUE

#load data

t1_df = read.csv(paste0(path, t1_df))
t2_df = read.csv(paste0(path, t2_df))
var_list = read.table(paste0(path, var_list, '.txt'))
var_list = as.character(var_list[,1])

if('sub_id' %in% names(t1_df)==FALSE){
  t1_df$X <- as.character(t1_df$X)
  names(t1_df)[which(names(t1_df) == 'X')] <-'sub_id'
}

if('sub_id' %in% names(t2_df)==FALSE){
  t2_df$X <- as.character(t2_df$X)
  names(t2_df)[which(names(t2_df) == 'X')] <-'sub_id'
}

t1_df = t1_df %>% select('sub_id', var_list)
t2_df = t2_df %>% select('sub_id', var_list)

#helper functions

file_names = c('sem.R', 'get_numeric_cols.R', 'match_t1_t2.R', 'get_retest_stats.R', 'make_rel_df.R')

helper_func_path = 'https://raw.githubusercontent.com/zenkavi/SRO_Retest_Analyses/master/code/helper_functions/'
for(file_name in file_names){
  eval(parse(text = getURL(paste0(helper_func_path,file_name), ssl.verifypeer = FALSE)))
}

# boot function

bootstrap_reliability = function(.t1_df=t1_df, .t2_df=t2_df, metrics = c('spearman', 'pearson', 'var_breakdown', 'partial_eta', 'sem','icc2.1', 'icc3.k'), dv_var, worker_col="sub_id"){
  
  indices = sample(1:150, 150, replace=T)
  sampled_t1_df = t1_df[indices,]
  sampled_t2_df = t2_df[indices,]
  
  out_df = make_rel_df(t1_df = sampled_t1_df, t2_df = sampled_t2_df, metrics = metrics, sample="bootstrap")
  
  return(out_df)
}

# generate random seed
cur_seed <- sample(1:2^15, 1)
set.seed(cur_seed)

print('Everything imported and defined. Starting output_df...')

# bootstrap sampled dataset for given n times
output_df = plyr::rdply(n, bootstrap_reliability())

print('Done with output_df. Saving...')

# add seed info
output_df$seed <- cur_seed

# save output
write.csv(output_df, paste0(output_dir, 'bootstrap_output_',job_num,'.csv'))

print(paste0('Output saved in ', paste0(output_dir, 'bootstrap_output_',job_num,'.csv')))
