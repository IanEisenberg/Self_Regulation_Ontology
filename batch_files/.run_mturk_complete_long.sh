set -e
for exp_id in shift_task two_stage_decision 
do
sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/mturk_complete/g" calculate_exp_DVs.batch | sbatch --time=48:00:00 --cpus-per-task=8
done
