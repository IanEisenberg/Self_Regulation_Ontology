import os
import numpy as np
from selfregulation.utils.utils import get_behav_data
import seaborn as sns

#Make Plot Directy if it doesn't exist
if not os.path.exists('Plots'):
    os.mkdir('Plots')

# get DV df
DV_df = get_behav_data()
tasks = np.unique(DV_df.columns.map(lambda x: x.split('.')[0]))

for task in tasks:
    subset = DV_df.filter(regex = '^%s' % task)
    subset = subset.dropna(how = 'all').dropna(axis = 1)
    sns.set(font_scale = 1.5)
    p = sns.pairplot(subset, kind = 'reg', size = 5, diag_kws = {'bins': 50})
    p.savefig('Plots/%s_pair_plot.pdf' % task, dpi = 300)

