import pandas as pd
from os import path
from selfregulation.utils.utils import get_behav_data
from selfregulation.utils.data_preparation_utils import remove_outliers
from selfregulation.utils.data_preparation_utils import transform_remove_skew


def get_retest_comparison_data():
    subsets = ['meaningful_variables_noDDM.csv', 'meaningful_variables_EZ.csv',
               'meaningful_variables_hddm.csv']
    dataset = pd.DataFrame()
    for subset in subsets:
        df = get_behav_data(file=subset)
        df_clean = remove_outliers(df)
        df_clean = transform_remove_skew(df_clean)
        drop_columns = set(dataset) & set(df_clean)
        df_clean.drop(labels=drop_columns, axis=1, inplace=True)
        dataset = pd.concat([dataset, df_clean], axis=1)
    return dataset

def anonymize_retest_data(data, data_dir):
    if path.exists(path.join(data_dir, 'worker_lookup.json')):
        old_worker_lookup = pd.read_json(path.join(data_dir, 'worker_lookup.json'), typ='series')
        complete_workers = (data.groupby('worker_id').count().finishtime>=62)
        complete_workers = list(complete_workers[complete_workers].index)
        workers = data.groupby('worker_id').finishtime.max().sort_values().index
        new_ids = []
        for worker in workers:
            if worker in complete_workers:
                new_ids.append(old_worker_lookup[old_worker_lookup == worker].index[0])
            else:
                new_ids.append(worker)
        data.replace(workers, new_ids, inplace = True)
        return{x: y for x, y in zip(new_ids, workers)}
    else:
        print('worker_lookup.json not in data directory')

