#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

#Usage:
#Rscript --vanilla calculate_hddm_kl.R ppc_data_dir

data_dir <- args[1]

library(tidyverse)
library(entropy)
sem <- function(x) {sd(x, na.rm=T) / sqrt(length(x))}

file_list = list.files(data_dir, pattern = "ppc_data")

print('Starting loop...')

for(f in file_list){
  
  print(paste0('Reading in ', f))
  
  ppc_data = read.csv(paste0(data_dir, f))
  
  kl_fname = gsub('ppc_data', 'kl', f)
  
  print(paste0('Processing ppc data for', f))

  kl_df = ppc_data %>%
    group_by() %>%
    mutate(rt = ifelse(rt<0, 0.00000000001, rt),
           rt_sampled = ifelse(rt_sampled<0, 0.00000000001, rt_sampled)) %>%
    group_by(node, sample) %>%
    do(data.frame(kl = KL.plugin(.$rt, .$rt_sampled)))
  
  print('Wiriting out first file')
  
  write.csv(kl_df, paste0(data_dir, kl_fname))
  
  kl_sum_fname = gsub('kl', 'kl_summary', kl_fname)
  
  print('Summarizing at subject level')
  
  kl_sum_df = kl_df %>%
    group_by(node) %>%
    summarise(mean_kl = mean(kl),
              sem_kl = sem(kl))
  
  print('Wiriting out second file')
  
  write.csv(kl_sum_df, paste0(data_dir, kl_sum_fname))
}