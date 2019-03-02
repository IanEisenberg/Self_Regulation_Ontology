## Synopsis

These batch files handle processing of UH2 behavioral data

## Pipeline

To go from downloaded Uh2 data to a data release (complete with DVs calculated for each experiment,
follow the steps in this pipeline. The pipeline follows the following steps, where [data] refers to the target:


### Post Process Data after it has already been downloaded
**post_process_[data]** - runs [data]_download_data.py in data_preparation

This function defaults to running post process on previously downloaded data
(also downloaded using [data]_download.py)
This function will default to saving/looking in "data_directory" specified in the settings. 
If not found, it will save in "base_directory"/"Data"
Creates [data]_data_post.pkl

(optional)
concat_mturk_data.batch - runs concat_mturk_data.py

takes post processed discovery and validation data and combines them
into one large dataset
Creates mturk_complete_data_post.pkl


### Calculate DVs from post processed data
**calculate_[data]_DVs.batch** - runs calculate_exps_DVs.py

Runs the DV calculation script separately for each experiment for the
particular dataset. Calculate_exp_DVs.py calls "get_exp_DVs" from 
expanalysis. This function defaults to setting "group_fun=True",
which, for example, calculates HDDM parameters for many tasks.
This can significantly increase the length of the DV calculations

Calculate_exp_DVs.py takes 2 required arguments: the exp_id,
and the the name of the dataset to label
the output DV files (e.g. mturk_complete). To optional arguments
are the name of an output folder (defaults to data directory, but
should be set to something else) where the DVs will be saved (temporarily, see below),
and a flag "no_group", which will set the use_group_fun to False

### Concatenate the DVs
**concatenate_[data]_DVs.py**

This is just a python function - no need to create a job. The
previous step created many DV files. Do put them into one large
dataset we have to concatenate them. This does that and saves
the resulting dataset in output.

### cleanup_[data].sh

This is just a batch script - no need to create a job.
This is a hard-coded function that moves the outputs of the hddm
fits and the concatenated DV file to Data.

### Create a Data directory where files of interest are created
**save_[data]** - runs [data]_save_data.py in data_preparation

This function create a new Data entry (e.g. Complete_10-10-10) in [base_directory]/Data
It expects to find a "base_directory" and "data_directory" in the settings file.
In the data directory should be the outputs of the above functions
