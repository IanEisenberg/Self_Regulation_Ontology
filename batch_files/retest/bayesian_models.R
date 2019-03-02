#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

output_path <- args[1]

#load packages
library(RCurl)
library(MCMCglmm)

from_gh = TRUE

file_names = c('boot_rel_data.R')

if(from_gh){
  library(RCurl)
  
  workspace_scripts = 'https://raw.githubusercontent.com/zenkavi/SRO_Retest_Analyses/master/code/workspace_scripts/'
  
  test_data_path = 'https://raw.githubusercontent.com/zenkavi/Self_Regulation_Ontology/master/Data/Complete_03-29-2018/'
  
  retest_data_path = 'https://raw.githubusercontent.com/zenkavi/Self_Regulation_Ontology/retest_scripts/Data/Retest_03-29-2018/'
  
  for(file_name in file_names){
    eval(parse(text = getURL(paste0(workspace_scripts,file_name), ssl.verifypeer = FALSE)))
  }
} else{
  workspace_scripts = '/Users/zeynepenkavi/Dropbox/PoldrackLab/SRO_Retest_Analyses/code/workspace_scripts/'
  
  test_data_path = '/Users/zeynepenkavi/Documents/PoldrackLabLocal/Self_Regulation_Ontology/Data/Complete_03-29-2018/'
  
  retest_data_path = '/Users/zeynepenkavi/Documents/PoldrackLabLocal/Self_Regulation_Ontology/Data/Retest_03-29-2018/'
  
  for(file_name in file_names){
    source(paste0(workspace_scripts,file_name))
  }
}

#########################
## Bayesian models ####
#########################
#summary(lmerTest::lmer(icc2.1 ~  task + (1|dv), boot_df))

print('Running icc2.1_by_task_model')

icc2.1_by_task_model = MCMCglmm(icc2.1 ~ task, random = ~dv, data=boot_df)

saveRDS(icc2.1_by_task_model, paste0(output_path, 'icc2.1_by_task_model.rds'))

print('Saving ',paste0(output_path, 'icc2.1_by_task_model.rds'))

#summary(lmerTest::lmer(icc3.k ~  task + (1|dv), boot_df))
print('Running icc3.k_by_task_model')

icc3.k_by_task_model = MCMCglmm(icc3.k ~ task, random = ~dv, data=boot_df)

saveRDS(icc3.k_by_task_model, paste0(output_path, 'icc3.k_by_task_model.rds'))

print('Saving ',paste0(output_path, 'icc3.k_by_task_model.rds'))

# summary(lmerTest::lmer(var_subs_pct~task+(1|dv),tmp%>%select(-var_ind_pct,-var_resid_pct)))

print('Running var_subs_pct_by_task_model')

var_subs_pct_by_task_model = MCMCglmm(var_subs_pct ~ task, random = ~dv, data=boot_df)

saveRDS(var_subs_pct_by_task_model, paste0(output_path, 'var_subs_pct_by_task_model.rds'))

print('Saving ',paste0(output_path, 'var_subs_pct_by_task_model.rds'))

# summary(lmerTest::lmer(var_ind_pct~task+(1|dv),tmp%>%select(-var_subs_pct,-var_resid_pct)))

print('Running var_ind_pct_by_task_model')

var_ind_pct_by_task_model = MCMCglmm(var_subs_pct ~ task, random = ~dv, data=boot_df)

saveRDS(var_ind_pct_by_task_model, paste0(output_path, 'var_ind_pct_by_task_model.rds'))

print('Saving ',paste0(output_path, 'var_ind_pct_by_task_model.rds'))

# summary(lmerTest::lmer(var_resid_pct~task+(1|dv),tmp%>%select(-var_subs_pct,-var_ind_pct)))
print('Running var_resid_pct_by_task_model')

var_resid_pct_by_task_model = MCMCglmm(var_subs_pct ~ task, random = ~dv, data=boot_df)

saveRDS(var_resid_pct_by_task_model, paste0(output_path, 'var_resid_pct_by_task_model.rds'))

print('Saving ',paste0(output_path, 'var_resid_pct_by_task_model.rds'))

tmp = measure_labels %>%
  mutate(dv = as.character(dv)) %>%
  filter(task == 'task',
         dv %in% meaningful_vars) %>%
  left_join(boot_df[,c("dv", "icc2.1", "icc3.k")], by = 'dv') %>%
  separate(dv, c('task_name', 'extra_1', 'extra_2'), sep = '\\.',remove=FALSE) %>%
  select(-extra_1, -extra_2)

# summary(lm(icc2.1 ~ num_all_trials, data = tmp))
# summary(lmerTest::lmer(icc2.1 ~ num_all_trials + (1|dv), data = tmp))
print('Running icc2.1_by_num_trials_model')

icc2.1_by_num_trials_model = MCMCglmm(icc2.1 ~ num_all_trials, random = ~dv, data=tmp)

saveRDS(icc2.1_by_num_trials_model, paste0(output_path, 'icc2.1_by_num_trials_model.rds'))

print('Saving ',paste0(output_path, 'icc2.1_by_num_trials_model.rds'))

# summary(lm(icc2.1 ~ num_all_trials, data = tmp))
# summary(lmerTest::lmer(icc2.1 ~ num_all_trials + (1|dv), data = tmp))
print('Running icc3.k_by_num_trials_model')

icc3.k_by_num_trials_model = MCMCglmm(icc2.1 ~ num_all_trials, random = ~dv, data=tmp)

saveRDS(icc3.k_by_num_trials_model, paste0(output_path, 'icc3.k_by_num_trials_model.rds'))

print('Saving ',paste0(output_path, 'icc3.k_by_num_trials_model.rds'))

tmp = measure_labels %>%
  mutate(dv = as.character(dv),
         contrast = ifelse(overall_difference == "difference", "contrast", "non-contrast")) %>%
  filter(ddm_task == 1,
         rt_acc != 'other') %>%
  drop_na() %>%
  left_join(boot_df[,c("dv", "icc2.1", "icc3.k")], by = 'dv')

# summary(lmerTest::lmer(icc ~ raw_fit + (1|dv) ,tmp %>% filter(contrast == "non-contrast")))
print('Running icc_by_rawfit_noncon_model')

icc_by_rawfit_noncon_model = MCMCglmm(icc2.1 ~ raw_fit, random = ~dv, data=tmp %>% filter(contrast == "non-contrast"))

saveRDS(icc_by_rawfit_noncon_model, paste0(output_path, 'icc_by_rawfit_noncon_model.rds'))

print('Saving ',paste0(output_path, 'icc_by_rawfit_noncon_model.rds'))

# summary(lmerTest::lmer(icc ~ raw_fit + (1|dv) ,tmp %>% filter(contrast == "contrast")))
print('Running icc_by_rawfit_con_model')

icc_by_rawfit_con_model = MCMCglmm(icc2.1 ~ raw_fit, random = ~dv, data=tmp %>% filter(contrast == "contrast"))

saveRDS(icc_by_rawfit_con_model, paste0(output_path, 'icc_by_rawfit_con_model.rds'))

print('Saving ',paste0(output_path, 'icc_by_rawfit_con_model.rds'))