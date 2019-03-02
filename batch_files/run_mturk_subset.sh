set -e
#for exp_id in directed_forgetting local_global_letter shape_matching stop_signal threebytwo 
for exp_id in angling_risk_task_always_sunny
do
    sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/mturk_complete/g" calculate_exp_DVs.batch | sbatch --time=168:00:00
    sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/mturk_retest/g" calculate_exp_DVs.batch | sbatch --time=168:00:00
    sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/fmri_followup/g" calculate_exp_DVs.batch | sbatch --time=168:00:00

done

