set -e
for task in adaptive_n_back attention_network_task choice_reaction_time directed_forgetting dot_pattern_expectancy local_global_letter motor_selective_stop_signal recent_probes shape_matching simon stim_selective_stop_signal stop_signal stroop threebytwo
do
  for samples in 10 25 50 100 250
  do
    sed -e "s/{TASK}/$task/g" -e "s/{MODEL_DIR}/\/oak\/stanford\/groups\/russpold\/users\/ieisenbe\/Self_Regulation_Ontology\/behavioral_data\/mturk_retest_output\/hddm_refits\//g" -e "s/{SUB_ID_DIR}/\/oak\/stanford\/groups\/russpold\/users\/zenkavi\/Self_Regulation_Ontology\/Data\/Retest_03-29-2018\/t1_data\/Individual_Measures\//g" -e "s/{OUT_DIR}/\/oak\/stanford\/groups\/russpold\/users\/ieisenbe\/Self_Regulation_Ontology\/behavioral_data\/mturk_retest_output\/hddm_fitstat\/num_samples_tests\//g" -e "s/{SUBSET}/refit/g" -e "s/{PARALLEL}/yes/g" -e "s/{HDDM_TYPE}/hierarchical/g" -e "s/{SAMPLES}/$samples/g" calculate_hddm_fitstat_samples.batch | sbatch
  done
done
