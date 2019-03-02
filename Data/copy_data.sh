sshfs sherlock:/scratch/users/ieisenbe/ /mnt/Sherlock_Scratch
#yes | cp -rf /mnt/Sherlock_Scratch/Self_Regulation_Ontology/Data/worker_lookup.json Local
#yes | cp -rf /mnt/Sherlock_Scratch/Self_Regulation_Ontology/Data/worker_counts.json Local
#yes | cp -rf /mnt/Sherlock_Scratch/Self_Regulation_Ontology/Data/worker_pay.json Local
yes | cp -rf /mnt/Sherlock_Scratch/Self_Regulation_Ontology/Data/*DV.json Local
yes | cp -rf /mnt/Sherlock_Scratch/Self_Regulation_Ontology/Data/*DV_valence.json Local
#yes | cp -rf /mnt/Sherlock_Scratch/Self_Regulation_Ontology/Data/*post*pkl Local
#yes | cp -rf /mnt/Sherlock_Scratch/Self_Regulation_Ontology/Data/*discovery*raw*pkl Local
sudo umount /mnt/Sherlock_Scratch
