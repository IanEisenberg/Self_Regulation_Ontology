cd /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/bootstrap_output/hddm_refits
cat *.csv > /oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_03-29-2018/Local/refits_bootstrap_merged.csv
cd /oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_03-29-2018/Local
awk '!a[$0]++' refits_bootstrap_merged.csv > refits_bootstrap_merged_clean.csv
rm refits_bootstrap_merged.csv
mv refits_bootstrap_merged_clean.csv ./refits_bootstrap_merged.csv
gzip refits_bootstrap_merged.csv
