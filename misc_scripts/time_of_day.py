from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import scale
import statsmodels.formula.api as smf
from selfregulation.utils.utils import get_behav_data

def convert_to_time(date_str):
    dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    time = dt.time()
    return (time.hour-5)%24 # convert from GMT to CST

def plot_time_effects(measure_DVs, melted_DVs, title=None):
    f, (ax1,ax2) = plt.subplots(1, 2, figsize=(16,8))
    for name in measure_DVs.columns[:-2]:
        sns.regplot('hour', name, data=measure_DVs, lowess=True, label=name,
                    ax=ax1, scatter_kws={'s': 100, 'alpha': .4})
    ax1.legend()
    sns.boxplot('split_time', 'value', hue='variable', data=melted, ax=ax2)
    if title:
        plt.suptitle(title, fontsize=18)
    plt.show()
    
verbose=True
# load data    
behav_data = get_behav_data(file='meaningful_variables_imputed.csv')
measures = np.unique([i.split('.')[0] for i in behav_data.columns])
time_effects = {}

for measure_name in measures[0:10]:
    measure = get_behav_data(file='Individual_Measures/%s.csv.gz' % measure_name)
    measure_DVs = behav_data.filter(regex=measure_name)
    measure_DVs.columns = [i.split('.')[1] for i in measure_DVs.columns]
    # scale
    measure_DVs = pd.DataFrame(scale(measure_DVs), index=measure_DVs.index, columns=measure_DVs.columns)
    
    finishtimes = measure.groupby('worker_id').finishtime.apply(lambda x: np.unique(x)[0])
    daytime = finishtimes.apply(convert_to_time)
    daytime.name='hour'
    measure_DVs = pd.concat([measure_DVs, daytime], axis=1)
    # add on time split in half and melt
    split_time = measure_DVs.hour>measure_DVs.hour.median()
    measure_DVs = measure_DVs.assign(split_time = split_time)
    melted = measure_DVs.melt(value_vars=measure_DVs.columns[:-2],
                             id_vars='split_time')
    
    # basic regression
    time_effects[measure_name] = {}
    if verbose: print('Measure Name\n', ['*']*79)
    for name in measure_DVs.columns[:-2]:
        rs = smf.ols(formula = '%s ~ hour' % name, data=measure_DVs).fit()
        time_effects[measure_name][name] = {'Beta': rs.params.hour,
                                            'pvalue': rs.pvalues.hour}
        if verbose:
            print(name, '\nBeta: %s' % rs.params.hour, 
                  '\nPvalue: %s\n' % rs.pvalues.hour)
    if verbose:
        # plot
        plot_time_effects(measure_DVs, melted, title=measure_name)
        input('Press Enter to Continue...')