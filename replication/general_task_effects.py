import matplotlib.pyplot as plt
import numpy as np
from os import path
import pandas as pd
import seaborn as sns
from selfregulation.utils.plot_utils import beautify_legend
from selfregulation.utils.utils import get_behav_data, get_info

base_dir = get_info('base_directory')
ext= 'png'
data = get_behav_data()

# our data 
dpx = get_behav_data(file='Individual_Measures/dot_pattern_expectancy.csv.gz')
dpx = dpx.query('exp_stage != "practice"')
N = len(dpx.worker_id.unique())
acc = 1-dpx.query('rt!=-1').groupby(['worker_id', 'condition']).correct.mean()
acc_stats = acc.groupby('condition').agg(["mean","std"])

rt = dpx.groupby(['worker_id', 'condition']).rt.mean()
rt_stats = rt.groupby('condition').agg(["mean","std"])

# literature data
#replicate "The neural circuitry supporting goal maintenance during cognitive control: a comparison of expectancy AX-CPT and dot probe expectancy paradigms"
literature_data = {'acc': {'mean': {'AX': .98, 'AY': .84, 'BX': .92, 'BY': .98},
                           'std': {'AX': .01, 'AY': .13, 'BX': .08, 'BY': .03}},
                   'rt': {'mean': {'AX': 512, 'AY': 677, 'BX': 505, 'BY': 497},
                          'std': {'AX': 137, 'AY': 162, 'BX': 193, 'BY': 160}}}
lit_acc_stats = pd.DataFrame.from_dict(literature_data['acc'])
lit_acc_stats.loc[:, 'mean'] = 1-lit_acc_stats.loc[:,'mean']
lit_rt_stats = pd.DataFrame.from_dict(literature_data['rt'])
lit_N = 26
# plot
sns.set_context('poster')
f, axes = plt.subplots(2,2, figsize=(16, 12))
axes = f.get_axes()
axes[0].errorbar(range(rt_stats.shape[0]), 
                 acc_stats.loc[:,'mean'], 
                 yerr=acc_stats.loc[:,'std']/(N**.5),
                 color='#D3244F', linewidth=5, elinewidth=3, 
                 label='Our Data')
axes[0].set_xlim([-.5, 3.5])
axes[0].set_ylim([0, .22])
axes[0].set_ylabel(r'Mean $\pm$ SEM error rate', fontsize=20)
axes[0].set_yticks(np.arange(0,.24,.02))
axes[0].set_xticks(range(rt_stats.shape[0]))
axes[0].set_xticklabels(['AX', 'AY', 'BX', 'BY'])
axes[0].grid(axis='y')
# plot reaction time
axes[2].errorbar(range(rt_stats.shape[0]), 
                 rt_stats.loc[:,'mean'], 
                 yerr=rt_stats.loc[:,'std']/(N**.5),
                 color='#D3244F', linewidth=5, elinewidth=3)
axes[2].set_xlim([-.5, 3.5])
axes[2].set_ylim([400, 750])
axes[2].set_ylabel(r'Mean $\pm$ SEM reaction time', fontsize=20)
axes[2].set_xticks(range(rt_stats.shape[0]))
axes[2].set_xticklabels(['AX', 'AY', 'BX', 'BY'])
axes[2].set_xlabel('Trial Type', fontsize=20)
axes[2].grid(axis='y')

# plot literature
axes[0].errorbar(range(lit_rt_stats.shape[0]), 
                 lit_acc_stats.loc[:,'mean'], 
                 yerr=lit_acc_stats.loc[:,'std']/(lit_N**.5),
                 color='#29A6F0', linewidth=5, elinewidth=3,
                 label='Lopez-Garcia et al')
leg = axes[0].legend(handlelength=0)
beautify_legend(leg, colors=['#D3244F','#29A6F0'])
# plot reaction time
axes[2].errorbar(range(lit_rt_stats.shape[0]), 
                 lit_rt_stats.loc[:,'mean'], 
                 yerr=lit_rt_stats.loc[:,'std']/(lit_N**.5),
                 color='#29A6F0', linewidth=5, elinewidth=3)
# plot comparison to literature
axes[1].scatter(lit_acc_stats.loc[:,'mean'],
                acc_stats.loc[:, 'mean'], color='k')
max_val = max(max(axes[1].get_xlim()), max(axes[1].get_ylim()))
axes[1].plot([0, max_val], [0, max_val], linestyle='--', color='k')
axes[1].set_ylabel('Our Values', fontsize=20)

axes[3].scatter(lit_rt_stats.loc[:,'mean'],
                rt_stats.loc[:, 'mean'], color='k')
min_val = min(min(axes[3].get_xlim()), min(axes[3].get_ylim()))
max_val = max(max(axes[3].get_xlim()), max(axes[3].get_ylim()))
axes[3].plot([min_val, max_val], [min_val, max_val], linestyle='--', color='k')
axes[3].set_xlabel('Literature Values', fontsize=20)
axes[3].set_ylabel('Our Values', fontsize=20)

plt.subplots_adjust(hspace=.3, wspace=.3)
save_dir = path.join(base_dir, 'Results', 'replication', 'Plots', 'DPX.%s' % ext)
f.savefig(save_dir, dpi=300, bbox_inches='tight')
plt.close()