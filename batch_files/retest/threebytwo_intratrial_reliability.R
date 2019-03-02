#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

library(dplyr)
library(tidyr)

test_data_path = args[1]
retest_data_path = args[2]
t = args[3]

retest_workers = c('s198', 's409', 's473', 's286', 's017', 's092', 's403', 's103','s081', 's357', 's291', 's492', 's294', 's145', 's187', 's226','s368', 's425', 's094', 's430', 's376', 's284', 's421', 's034','s233', 's027', 's108', 's089', 's196', 's066', 's374', 's007','s509', 's365', 's305', 's453', 's504', 's161', 's441', 's205','s112', 's218', 's129', 's093', 's180', 's128', 's170', 's510','s502', 's477', 's551', 's307', 's556', 's121', 's237', 's481','s259', 's467', 's163', 's111', 's427', 's508', 's190', 's091','s207', 's484', 's449', 's049', 's336', 's212', 's142', 's313','s369', 's165', 's028', 's216', 's346', 's083', 's391', 's388','s384', 's275', 's442', 's505', 's098', 's456', 's209', 's372','s179', 's168', 's084', 's329', 's373', 's065', 's277', 's026','s011', 's063', 's507', 's005', 's495', 's501', 's032', 's326','s396', 's420', 's469', 's244', 's359', 's110', 's383', 's254','s060', 's339', 's380', 's471', 's206', 's182', 's500', 's314','s285', 's086', 's012', 's097', 's149', 's192', 's173', 's262','s273', 's402', 's015', 's014', 's085', 's489', 's071', 's062','s042', 's009', 's408', 's184', 's106', 's397', 's451', 's269','s295', 's265', 's301', 's082', 's238', 's328', 's334')

#dv's
calc_break_dvs = function(df, breaks=c(seq(0, 440, 10)[c(-1, -45)], 439)){
  
  #filter out pratice trials and do any other preprocessing necessary
  df = df %>%
    filter(exp_stage != "practice") %>%
    mutate(CTI = as.factor(CTI))
  
  #function to calculate task specific dv's for a given subset of raw response data
  calc_dvs = function(df){
    
    df = df %>%
      mutate(correct_shift = lag(correct),
             task_switch_shift = lag(task_switch))
    
    missed_percent = mean(df$rt == -1)
    
    df = df %>%
      filter(rt != -1)
    
    df_correct = df %>%
      filter(correct == "True" & correct_shift == "True") 
    
    dvs = data.frame(acc = mean(ifelse(df$correct == "True",1,0)),
                     avg_rt_error = median(df$rt[df$correct == "False"]),
                     std_rt_error = sd(df$rt[df$correct == "False"]),
                     avg_rt = median(df_correct$rt),
                     std_rt = sd(df_correct$rt),
                     missed_percent = missed_percent)
    
    
    if(nrow(df_correct)>0){
      cue_switch_cost_rt_df = df_correct %>% group_by(CTI, cue_switch) %>% summarise(cue_switch_cost = median(rt)) %>% complete(cue_switch, CTI) %>% distinct() %>% filter(cue_switch %in% c("stay", "switch")) %>% ungroup() %>% spread(cue_switch,cue_switch_cost) %>% group_by(CTI) %>% summarise(cue_switch_cost_rt = switch-stay) %>% mutate(CTI = as.numeric(as.character(CTI)))
      
      if(nrow(cue_switch_cost_rt_df)>0){
        dvs$cue_switch_cost_rt_100 = cue_switch_cost_rt_df$cue_switch_cost_rt[cue_switch_cost_rt_df$CTI == 100]
        dvs$cue_switch_cost_rt_900 = cue_switch_cost_rt_df$cue_switch_cost_rt[cue_switch_cost_rt_df$CTI == 900]
      }
      
      else{
        dvs$cue_switch_cost_rt_100 = NA
        dvs$cue_switch_cost_rt_900 = NA
      }
      
      task_switch_rt = df_correct %>% mutate(task_switch = factor(ifelse(grepl("switch",task_switch), T, F))) %>% group_by(CTI, task_switch) %>% summarise(rt = median(rt)) %>% complete(task_switch, CTI) %>% distinct() %>% ungroup() %>% filter(task_switch == T)
      
      cue_switch_rt = df_correct %>% group_by(CTI, cue_switch) %>% summarise(rt = median(rt)) %>% complete(cue_switch, CTI)  %>% distinct() %>%  ungroup() %>% filter(cue_switch == "switch")
      
      if(nrow(task_switch_rt)>0 & nrow(cue_switch_rt)>0){
        dvs$task_switch_cost_rt_100 = task_switch_rt$rt[task_switch_rt$CTI == 100] - cue_switch_rt$rt[cue_switch_rt$CTI == 100]
        dvs$task_switch_cost_rt_900 = task_switch_rt$rt[task_switch_rt$CTI == 900] - cue_switch_rt$rt[cue_switch_rt$CTI == 900]
      }
      else{
        dvs$task_switch_cost_rt_100 = NA
        dvs$task_switch_cost_rt_900 = NA
      }
      
    }
    else {
      dvs$cue_switch_cost_rt_100 = NA
      dvs$cue_switch_cost_rt_900 = NA
      dvs$task_switch_cost_rt_100 = NA
      dvs$task_switch_cost_rt_900 = NA
    }
    
    cue_switch_cost_acc_df = df %>% group_by(CTI, cue_switch) %>% summarise(acc = mean(ifelse(correct == "True",1, 0))) %>% complete(cue_switch, CTI)  %>% distinct() %>% filter(cue_switch %in% c("stay", "switch")) %>% spread(cue_switch, acc) %>% ungroup() %>% group_by(CTI) %>% summarise(cue_switch_cost_acc = switch-stay)
    
    
    if(nrow(cue_switch_cost_acc_df)>0){
      dvs$cue_switch_cost_acc_100 = cue_switch_cost_acc_df$cue_switch_cost_acc[cue_switch_cost_acc_df$CTI == 100]
      dvs$cue_switch_cost_acc_900 = cue_switch_cost_acc_df$cue_switch_cost_acc[cue_switch_cost_acc_df$CTI == 900]
    }
    else{
      dvs$cue_switch_cost_acc_100 = NA
      dvs$cue_switch_cost_acc_900 = NA 
    }
    
    task_switch_acc = df %>% mutate(task_switch = factor(ifelse(grepl("switch",task_switch), T, F))) %>% group_by(CTI, task_switch) %>% summarise(acc = mean(ifelse(correct == "True",1,0))) %>% complete(task_switch, CTI)  %>% distinct() %>%  ungroup() %>% filter(task_switch == T)
    
    cue_switch_acc = df %>% group_by(CTI, cue_switch) %>% summarise(acc = mean(ifelse(correct == "True",1,0))) %>% complete(cue_switch, CTI)  %>% distinct() %>% ungroup() %>% filter(cue_switch == "switch")
    
    if(nrow(task_switch_acc)>0&nrow(cue_switch_acc)>0){
      dvs$task_switch_cost_acc_100 = task_switch_acc$acc[task_switch_acc$CTI == 100] - cue_switch_acc$acc[cue_switch_acc$CTI == 100]
      dvs$task_switch_cost_acc_900 = task_switch_acc$acc[task_switch_acc$CTI == 900] - cue_switch_acc$acc[cue_switch_acc$CTI == 900]
    }
    else{
      dvs$task_switch_cost_acc_100 = NA
      dvs$task_switch_cost_acc_900 = NA
    }
    
    return(dvs)
  }
  
  out = data.frame()
  
  #anonymous within function for loop that subset the number of trials for each increment for one subject's data and calculates the dv's for that subset
  for(i in 1:length(breaks)){
    tmp = df[1:breaks[i],]
    dvs = calc_dvs(tmp)
    out = rbind(out, dvs)
  }
  
  # calculate dv's for the remaining trials
  if(nrow(df)!=440 & nrow(df)%%10 != 0){
    dvs = calc_dvs(df)
    out = rbind(out, dvs)
  }
  
  #assign row names to a column to keep track of number of trial increments for later merge
  out$breaks = row.names(out)
  
  return(out)
}

if(t=="t1"){
  
  #read raw task data
  t1_tbt = read.csv(gzfile(paste0(test_data_path,'Individual_Measures/threebytwo.csv.gz'), 'rt'))
  #filter data for retest workers
  t1_tbt = t1_tbt[as.character(t1_tbt$worker_id) %in% retest_workers,]
  
  #Create blank output df that will be filled with each subject's dv's 
  t1_dvs = data.frame()
  #Loop through each subject
  for(i in 1:length(unique(t1_tbt$worker_id))){
    #Print index to be trace output files progress
    print(i)
    #extract data for current worker
    cur_worker = unique(t1_tbt$worker_id)[i]
    df = t1_tbt %>% filter(worker_id == cur_worker)
    #calculate dv's for each 10 trial increment for given subject
    sub_dvs = calc_break_dvs(df)
    #add subject identifier to output
    sub_dvs$sub_id = cur_worker
    #append current subject's calculated dv's for each subset of trials to the output dataframe
    t1_dvs = rbind(t1_dvs, sub_dvs)
  }
  
  write.csv(t1_dvs, paste0(retest_data_path, 'Local/t1_tbt_dvs.csv'))
  
}

if(t=="t2"){
  
  t2_tbt = read.csv(gzfile(paste0(retest_data_path,'Individual_Measures/threebytwo.csv.gz'), 'rt'))
  t2_tbt = t2_tbt[as.character(t2_tbt$worker_id) %in% retest_workers,]
  
  t2_dvs = data.frame()
  for(i in 1:length(unique(t2_tbt$worker_id))){
    print(i)
    cur_worker = unique(t2_tbt$worker_id)[i]
    df = t2_tbt %>% filter(worker_id == cur_worker)
    sub_dvs = calc_break_dvs(df)
    sub_dvs$sub_id = cur_worker
    t2_dvs = rbind(t2_dvs, sub_dvs)
  }
  
  write.csv(t2_dvs, paste0(retest_data_path, 'Local/t2_tbt_dvs.csv'))
}

