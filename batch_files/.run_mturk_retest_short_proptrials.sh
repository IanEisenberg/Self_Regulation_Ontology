set -e
for exp_id in angling_risk_task_always_sunny bickel_titrator columbia_card_task_cold columbia_card_task_hot dietary_decision digit_span discount_titrate go_nogo hierarchical_rule information_sampling_task keep_track kirby probabilistic_selection psychological_refractory_period_two_choices ravens simple_reaction_time spatial_span tower_of_london
do
  for proptrials in 0.25 0.5 0.75
  do
    for subset in retest complete
    do
      sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/$subset/g" -e "s/{PROPTRIALS}/$proptrials/g" -e "s/{RAND}/no/g" calculate_exp_DVs_proptrials.batch | sbatch --time=02:00:00 --cpus-per-task=4
    done
  done
done
