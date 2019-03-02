for subset in retest t1
do
    for exp_id in adaptive_n_back attention_network_task choice_reaction_time directed_forgetting dot_pattern_expectancy local_global_letter motor_selective_stop_signal recent_probes shape_matching simon stim_selective_stop_signal stop_signal stroop threebytwo
    do
    	for sub_id in s005 s007 s009 s011 s012 s014 s015 s017 s026 s027 s028 s032 s034 s042 s049 s060 s062 s063 s065 s066 s071 s081 s082 s083 s084 s085 s086 s089 s091 s092 s093 s094 s097 s098 s103 s106 s108 s110 s111 s112 s121 s128 s129 s142 s145 s149 s161 s163 s165 s168 s170 s173 s179 s180 s182 s184 s187 s190 s192 s196 s198 s205 s206 s207 s209 s212 s216 s218 s226 s233 s237 s238 s244 s254 s259 s262 s265 s269 s273 s275 s277 s284 s285 s286 s291 s294 s295 s301 s305 s307 s313 s314 s326 s328 s329 s334 s336 s339 s346 s357 s359 s365 s368 s369 s372 s373 s374 s376 s380 s383 s384 s388 s391 s396 s397 s402 s403 s408 s409 s420 s421 s425 s427 s430 s441 s449 s451 s453 s456 s467 s469 s471 s473 s477 s481 s484 s489 s492 s495 s500 s501 s502 s504 s505 s507 s508 s509 s510 s551 s556
    	do
    		if [[ "$subset" == retest ]]; then
        	echo "python calculate_hddm_flat.py ${exp_id} ${sub_id} /oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_03-29-2018/Individual_Measures/ ${subset} /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/hddm_flat/subject_fits/" >> flat_tasks_list.sh
        	elif [[ "$subset" == t1 ]]; then
        	echo "python calculate_hddm_flat.py ${exp_id} ${sub_id} /oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_03-29-2018/t1_data/Individual_Measures/ ${subset} /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/hddm_flat/subject_fits/" >> flat_tasks_list.sh
        	fi
        done
    done
done
