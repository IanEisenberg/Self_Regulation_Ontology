cd /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/bootstrap_output/hddm_flat
cat *.csv > /oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_03-29-2018/Local/flat_bootstrap_merged.csv
cd /oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_03-29-2018/Local
awk '!a[$0]++' flat_bootstrap_merged.csv > flat_bootstrap_merged_clean.csv
rm flat_bootstrap_merged.csv
mv flat_bootstrap_merged_clean.csv ./flat_bootstrap_merged.csv
gzip flat_bootstrap_merged.csv
