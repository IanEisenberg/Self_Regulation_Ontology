for exp_id in adaptive_n_back attention_network_task choice_reaction_time directed_forgetting dot_pattern_expectancy local_global_letter motor_selective_stop_signal recent_probes shape_matching  simon stim_selective_stop_signal stop_signal stroop threebytwo
do
  for proptrials in 0.25 0.5 0.75
  do
    for subset in retest complete
    do
      sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/$subset/g" -e "s/{PROPTRIALS}/$proptrials/g" -e "s/{RAND}/no/g" calculate_exp_DVs_proptrials.batch | sbatch --qos=long --time=168:00:00
    done
  done
done
