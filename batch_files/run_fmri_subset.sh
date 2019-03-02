set -e
for exp_id in motor_selective_stop_signal
do
    sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/fmri_followup/g" calculate_exp_DVs.batch | sbatch --time=168:00:00

done

