

import math
import matplotlib.pyplot as plt
import numpy as np
from os import path
import pandas as pd
import seaborn as sns
from selfregulation.utils.plot_utils import beautify_legend, format_num, format_variable_names
from selfregulation.utils.utils import filter_behav_data, get_behav_data, get_demographics, get_info

# correlation of ravens and literature
# replication of "Intelligence and socioeconomic success: A meta-analytic
# review of longitudinal research"

base_dir = get_info('base_directory')
ext= 'png'
data = get_behav_data()     
demographics = get_demographics()                    
data = data.loc[demographics.index]     
# get dataframe of intelligence measure (raven's progressive matrices) and demographics)                              
df = pd.concat([data.filter(regex='raven'), demographics], axis=1)

# get raven's reliability
reliability = get_behav_data(dataset='Retest_02-03-2018', file='bootstrap_merged.csv.gz')
raven_reliability = reliability.groupby('dv').icc.mean().filter(regex='raven')[0]
# demographic reliabilities 
demo_reliabilities = [1.0]*demographics.shape[1]

# correlations
correlations = df.corr().filter(regex='raven').sort_values(by='ravens.score').iloc[:-1]
correlations.insert(0, 'target_reliability', demo_reliabilities)
adjusted = correlations['ravens.score']/(raven_reliability*correlations['target_reliability'])**.5
correlations.insert(0, 'adjusted_correlation', adjusted)

# adjust based on literature values
lit_raven_reliability = .83
lit_demo_reliability = correlations.target_reliability.copy()
lit_demo_reliability['HighestEducation'] = .88
lit_demo_reliability['HouseholdIncome'] = .83
correlations.insert(0, 'lit_target_reliability', lit_demo_reliability)
lit_adjusted = correlations['ravens.score']/(lit_raven_reliability*correlations['lit_target_reliability'])**.5
correlations.insert(0, 'lit_adjusted_correlation', lit_adjusted)


# correlation of ravens and PCA of tasks
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

data = get_behav_data(file='meaningful_variables_imputed.csv')     
task_data = filter_behav_data(data, 'task').drop('ravens.score', axis=1)
pca = PCA(2)
transformed = pca.fit_transform(scale(task_data))
ravens_data = data['ravens.score']
df = pd.DataFrame(transformed).assign(ravens = ravens_data.tolist())
sns.heatmap(abs(df.corr()))
