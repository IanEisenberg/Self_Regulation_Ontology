import matplotlib.pyplot as plt
import numpy as np
from os import path
import pandas as pd
import seaborn as sns
from shutil import copyfile
from selfregulation.utils.utils import (get_behav_data, get_info, get_recent_dataset, 
                                        get_retest_data, get_var_category)
from selfregulation.utils.plot_utils import format_num
sns.set_palette("Set1", 8, .75)

base_dir = get_info('base_directory')
ext = 'pdf'
dpi = 300

# Raw Data Plots
"""
# Load data if plots need to be regenerated

post_process_data_loc = ''
data = pd.load_pickle(post_process_data_loc)

# plt total time on tasks
(data.groupby('worker_id').ontask_time.sum()/3600).hist(bins=40, 
                                                        grid=False, 
                                                        density=True,
                                                        figsize=(12,8))
plt.xlabel('Time (Hours)')
plt.title('Total Time on Tasks', weight='bold')



# plot distribution of times per task
tasks = data.experiment_exp_id.unique()
N = len(tasks)

f, axes = plt.subplots(3,1,figsize=(16,20))
for i in range(3):
    for exp in tasks[i*N//3: (i+1)*N//3]:
        task_time = data.query('experiment_exp_id == "%s"' % exp).ontask_time/3600
        task_time.name = ' '.join(exp.split('_'))
        if not pd.isnull(task_time.sum()):
            sns.kdeplot(task_time, linewidth=3, ax=axes[i])
    axes[i].set_xlim(0,1)
    axes[i].legend(ncol=3)
plt.xlabel('Time (Hours)')
"""

# Worker statistics plots
"""
# Load worker completions if plot needs to be regenerated
worker_completion_loc = '/mnt/OAK/behavioral_data/admin/worker_counts.json'
worker_completions = json.load(open(worker_completion_loc, 'r'))
with sns.plotting_context('poster'):
    save_dir = path.join(base_dir, 'Results', 'data_collection', 'Plots', 'worker_completions.%s' % ext)
    completion_rate = np.mean(np.array(list(worker_completions.values())) ==63)
    completion_rate = format_num(completion_rate*100, 1)
    analyzed_rate = 522/len(worker_completions)
    analyzed_rate = format_num(analyzed_rate*100, 1)
    plt.figure(figsize=(12,8))
    plt.hist(worker_completions.values(), bins=40, width=5)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.text(5, 400, 'Completion Rate: %s' % completion_rate, size=20)
    ax.text(5, 350, 'Passed QC: %s' % analyzed_rate, size=20)
    plt.xlabel('Number of Tasks Completed', fontsize=20)
plt.savefig(save_dir, dpi=300, bbox_inches='tight')
"""

# ****************************************************************************
# plot psychometric reliability
# ****************************************************************************

sns.set_context('poster')
dataset = get_recent_dataset()
meaningful_vars = get_behav_data(dataset=dataset, 
                                 file='meaningful_variables_imputed.csv').columns
meaningful_vars = [i.replace('.logTr','') for i in meaningful_vars]
meaningful_vars = [i.replace('.ReflogTr','') for i in meaningful_vars]
retest_data = get_retest_data(dataset=dataset.replace('Complete','Retest'))
# only select meaningful variables
retest_data = retest_data.query('dv in %s' % list(meaningful_vars))

# create reliability dataframe
measure_cat = [get_var_category(v).title() for v in retest_data.index]
retest_data.loc[:,'Measure Category'] = measure_cat
Survey_N = np.sum(retest_data.loc[:, 'Measure Category']=='Survey')
Task_N = len(retest_data)-Survey_N

def plot_retest_data(retest_data, size=4.6, save_dir=None):
    colors = [sns.color_palette('Reds_d',3)[0], sns.color_palette('Blues_d',3)[0]]
    f = plt.figure(figsize=(size,size*.75))
    # plot boxes
    with sns.axes_style('white'):
        box_ax = f.add_axes([.15,.1,.8,.5]) 
        sns.boxplot(x='icc3.k', y='Measure Category', ax=box_ax, data=retest_data,
                    palette={'Survey': colors[0], 'Task': colors[1]}, saturation=1,
                    width=.5, linewidth=size/4)
    box_ax.text(0, 1, '%s Task measures' % Task_N, color=colors[1], fontsize=size*2)
    box_ax.text(0, 1.2, '%s Survey measures' % Survey_N, color=colors[0], fontsize=size*2)
    box_ax.set_ylabel('Measure category', fontsize=size*2, labelpad=size)
    box_ax.set_xlabel('Intraclass correlation coefficient', fontsize=size*2, labelpad=size)
    box_ax.tick_params(labelsize=size*1.5, pad=size, length=2)
    [i.set_linewidth(size/5) for i in box_ax.spines.values()]

    # plot distributions
    dist_ax = f.add_axes([.15,.6,.8,.4]) 
    dist_ax.set_xlim(*box_ax.get_xlim())
    dist_ax.set_xticklabels('')
    dist_ax.tick_params(length=0)
    for i, (name, g) in enumerate(retest_data.groupby('Measure Category')):
        sns.kdeplot(g['icc3.k'], color=colors[i], ax=dist_ax, linewidth=size/3, 
                    shade=True, legend=False)
    dist_ax.set_ylim((0, dist_ax.get_ylim()[1]))
    dist_ax.axis('off')
    if save_dir:
        plt.savefig(save_dir, dpi=dpi, bbox_inches='tight')

save_dir = path.join(base_dir, 'Results', 'data_collection', 'Plots', 'ICC_distplot.%s' % ext)
size=4.6
plot_retest_data(retest_data, size, save_dir)

# also save to psych ontology folder
paper_dir = path.join(base_dir, 'Results', 'Psych_Ontology_Paper', 'Plots', 'FigS01_test-retest.%s' % ext)
copyfile(save_dir, paper_dir)