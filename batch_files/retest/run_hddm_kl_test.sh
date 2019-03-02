set -e
for ppc_data_dir in t1_flat retest_flat
do
sed -e "s/{PPC_DATA_DIR}/$ppc_data_dir/g" calculate_hddm_kl.batch | sbatch
done
