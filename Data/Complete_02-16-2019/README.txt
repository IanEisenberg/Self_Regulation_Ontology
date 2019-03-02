demographics_survey.csv: demographic information from expfactory-surveys

alcohol_drug_survey.csv: alcohol, smoking, marijuana and other drugs from expfactory-surveys

ky_survey.csv: mental health and neurological/health conditions from expfactory-surveys

items.csv.gz: gzipped csv of all item information across surveys

subject_x_items.csv: reshaped items.csv such that rows are subjects and columns are individual items

Individual Measures: directory containing gzip compressed files for each individual measures

DV_valence.csv: Subjective assessment of whether each variable's 'natural' direction implies 'better' self regulation

variables_exhaustive.csv: all variables calculated for each measure

meaningful_variables_noDDM.csv: subset of exhaustive data to only meaningful variables with DDM parameters removed

meaningful_variables_EZ.csv: subset of exhaustive data to only meaningful variables with rt/acc parameters removed (replaced by EZ DDM params)

meaningful_variables_hddm.csv: subset of exhaustive data to only meaningful variables with rt/acc parameters removed (replaced by hddm DDM params)

meaningful_variables.csv: Same as meaningful_variables_hddm.csv

meaningful_variables_clean.csv: same as meaningful_variables.csv with skewed variables transformed and then outliers removed 

meaningful_variables_imputed.csv: meaningful_variables_clean.csv after imputation with missForest

taskdata*.csv: taskdata are the same as meaningful_variables excluded surveys. Note that imputation is performed on the entire dataset including surveys

short*.csv: short versions are the same as long versions with variable names shortened using variable_name_lookup.csv

