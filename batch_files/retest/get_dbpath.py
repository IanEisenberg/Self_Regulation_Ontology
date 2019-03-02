from os import path
import os

def get_dbpath():

    model_dir = os.environ['MODEL_DIR']
    task = os.environ['TASK']
    subset = os.environ['SUBSET']
    hddm_type = os.environ['HDDM_TYPE']
    parallel = os.environ['PARALLEL']

    if(hddm_type == 'flat'):
        model_path = model_dir
        
    if(hddm_type == 'hierarchical' and parallel == 'yes'):
        model_path = path.join(model_dir, task+'_parallel_output')

    if(hddm_type == 'hierarchical' and parallel == 'no'):
        model_path = path.join(model_dir, task)

    return model_path
