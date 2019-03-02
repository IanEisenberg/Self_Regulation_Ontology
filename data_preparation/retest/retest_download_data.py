from expanalysis.experiments.utils import remove_duplicates, result_filter
from expanalysis.results import get_filters, get_result_fields
from expanalysis.results import Result
from os import path, makedirs
import pickle

from selfregulation.utils.utils import get_info

#set token and data directory
token = get_info('expfactory_token', infile='/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Self_Regulation_Retest_Settings.txt')

data_dir=path.join('/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/','Retest_01-23-2018', 'Local')

if not path.exists(data_dir):
    makedirs(data_dir)


# Set up filters
filters = get_filters()
drop_columns = ['battery_description', 'experiment_reference', 'experiment_version', \
         'experiment_name','experiment_cognitive_atlas_task']
for col in drop_columns:
    filters[col] = {'drop': True}

# Strip token from specified file
f = open(token)
access_token = f.read().strip()

# Set up variables for the download request
battery = 'Self Regulation Retest Battery' 
url = 'http://www.expfactory.org/new_api/results/62/'
file_name = 'mturk_retest_data.json'

fields = get_result_fields()

# Create results object
results = Result(access_token, filters = filters, url = url)

# Clean filters from results objects
results.clean_results(filters)

# Extract data from the results object
data = results.data

# Remainder of download_data
data = result_filter(data, battery = battery)
remove_duplicates(data)
data = data.query('worker_id not in ["A254JKSDNE44AM", "A1O51P5O9MC5LX"]') # Sandbox workers
data.reset_index(drop = True, inplace = True) 

# Save data
data.to_json(path.join(data_dir, file_name))
pickle.dump(data, open(path.join(data_dir, 'mturk_retest_data.pkl'), 'wb'), -1)

