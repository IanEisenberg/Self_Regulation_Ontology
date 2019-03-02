cd /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/bootstrap_output/demog
cat *.csv > /oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_03-29-2018/Local/demog_boot_merged.csv
cd /oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_03-29-2018/Local
awk '!a[$0]++' demog_boot_merged.csv > demog_boot_merged_clean.csv
rm demog_boot_merged.csv
mv demog_boot_merged_clean.csv ./demog_boot_merged.csv
gzip demog_boot_merged.csv
