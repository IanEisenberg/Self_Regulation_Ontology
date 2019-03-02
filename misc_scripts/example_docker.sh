# example commands to run parts of SRO analysis using docker.

# DATA PREP SCRIPTS
# assumes you have a $HOME/tmp folder

docker run --rm  -ti sro

# calculate exp dvs
data_loc=$HOME/tmp/
output=$HOME/tmp
docker run --rm  \
--mount type=bind,src=$data_loc,dst=/Data \
--mount type=bind,src=$output,dst=/output \
-ti sro \
python batch_files/helper_funcs/calculate_exp_DVs.py grit_scale_survey mturk_complete --out_dir /output

# download data
data_loc=$HOME/tmp
expfactory_loc=$HOME/tmp/expfactory
docker run --rm  \
--mount type=bind,src=$data_loc,dst=/Data \
--mount type=bind,src=$expfactory_loc/docs,dst=/expfactory_token
-ti sro python data_preparation/fmri_followup_download_data.py --job all

# save data
data_loc=$HOME/tmp
output_loc=$HOME/tmp
docker run --rm  \
--mount type=bind,src=$data_loc,dst=/Data \
--mount type=bind,src=$output_loc,dst=/SRO/Data \
-ti sro python data_preparation/save_data.py

# run analysis with example data
output_loc=$HOME/tmp 
exp_id=adaptive_n_back
subset=example
docker run --rm  \
--mount type=bind,src=$output_loc,dst=/output \
-ti sro python /SRO/batch_files/helper_funcs/calculate_exp_DVs.py ${exp_id} ${subset} --out_dir /output --hddm_samples 100 --hddm_burn 50 

# run analysis with mounted data
data_loc=$HOME/tmp
output_loc=$HOME/tmp
exp_id=adaptive_n_back
subset=mturk_complete # replace with subset name
docker run --rm  \
--mount type=bind,src=$data_loc,dst=/Data \
--mount type=bind,src=$output_loc,dst=/output \
-ti sro python /SRO/batch_files/helper_funcs/calculate_exp_DVs.py ${exp_id} ${subset} --out_dir /output --hddm_samples 100 --hddm_burn 50

# DATA ANALYSIS SCRIPTS
# enter into docker environment for development
scripts_loc=~/Experiments/Self_Regulation_Ontology/dimensional_structure
data_loc=~/Experiments/Self_Regulation_Ontology/Data
results_loc=~/Experiments/Self_Regulation_Ontology/Results

docker run --rm  \
--mount type=bind,src=$scripts_loc,dst=/SRO/dimensional_structure \
--mount type=bind,src=$data_loc,dst=/SRO/Data \
--mount type=bind,src=$results_loc,dst=/Results \
-ti sro 


# enter into docker environment with mounted data
data_loc=~/Experiments/Self_Regulation_Ontology/Data
results_loc=~/Experiments/Self_Regulation_Ontology/Results

docker run --rm  \
--mount type=bind,src=$data_loc,dst=/SRO/Data \
--mount type=bind,src=$results_loc,dst=/Results \
-ti sro 

# generate dimensional results
data_loc=~/Experiments/Self_Regulation_Ontology/Data
results_loc=~/Experiments/Self_Regulation_Ontology/Results

docker run --rm  \
--mount type=bind,src=$data_loc,dst=/SRO/Data \
--mount type=bind,src=$results_loc,dst=/Results \
-ti sro python /SRO/dimensional_structure/generate_results.py -bootstrap -shuffle_repeats 10 -plot_backend Agg

# just prediction
docker run --rm  \
--mount type=bind,src=$data_loc,dst=/SRO/Data \
--mount type=bind,src=$results_loc,dst=/Results \
-ti sro python /SRO/dimensional_structure/generate_results.py -bootstrap -shuffle_repeats 10 -plot_backend Agg -no_analysis -no_plot -no_group_plot -no_group_analysis