cd /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/hddm_flat/subject_fits
for EXP_ID in adaptive_n_back attention_network_task choice_reaction_time directed_forgetting dot_pattern_expectancy local_global_letter motor_selective_stop_signal recent_probes shape_matching simon stim_selective_stop_signal stop_signal stroop threebytwo
do
	for SUBSET in _retest _t1
    do
    find . -type f -name "${EXP_ID}${SUBSET}_s*hddm_flat.csv" -exec cat > "../${EXP_ID}${SUBSET}_hddm_flat.csv" {} \;
    awk '!a[$0]++' "../${EXP_ID}${SUBSET}_hddm_flat.csv" > "../${EXP_ID}${SUBSET}_hddm_flat_clean.csv"
    rm ../${EXP_ID}${SUBSET}_hddm_flat.csv
    mv ../${EXP_ID}${SUBSET}_hddm_flat_clean.csv ../${EXP_ID}${SUBSET}_hddm_flat.csv
    done
done