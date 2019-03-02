import matplotlib.pyplot as plt
from os import path
import pandas as pd
import seaborn as sns
from selfregulation.utils.plot_utils import beautify_legend, format_num, format_variable_names
from selfregulation.utils.utils import get_behav_data, get_demographics, get_info

base_dir = get_info('base_directory')
ext= 'png'
data = get_behav_data()

# *************************************************************************
# Successful replications
# ************************************************************************
# two_stage
two_stage_df = get_behav_data(file='Individual_Measures/two_stage_decision.csv.gz')
# subset two subjects who passed quality control
successful_two_stage = data.filter(regex='two_stage').dropna(how='any').index
two_stage_df = two_stage_df.query('worker_id in %s' % list(successful_two_stage))
two_stage_df = two_stage_df.query('rt_first != -1 and feedback_last in [0,1]')
colors = sns.hls_palette(2)
plot_df = (1-two_stage_df.groupby(['worker_id','stage_transition_last','feedback_last']).switch.mean()).reset_index()
plot_df.feedback_last = plot_df.feedback_last.replace({0:'Unrewarded', 1:'Rewarded'})
plot_df.stage_transition_last = \
    plot_df.stage_transition_last.replace({'infrequent':'Rare', 'frequent':'Common'})
# shift
shift_df = get_behav_data(file='Individual_Measures/shift_task.csv.gz')
# subset two subjects who passed quality control
successful_shift = data.filter(regex='shift').dropna(how='any').index
shift_df = shift_df.query('worker_id in %s' % list(successful_shift))
shift_df = shift_df.query('rt != -1')
shift_df = shift_df.groupby(['worker_id','trials_since_switch']).correct.mean().reset_index() 
    
# plot
f, axes = plt.subplots(1,2,figsize=(20,8))
# two stage
sns.barplot(x='feedback_last', y='switch', hue='stage_transition_last', 
            data=plot_df, 
            order=['Rewarded', 'Unrewarded'],
            hue_order=['Common', 'Rare'],
            palette=colors,
            ax=axes[0])
axes[0].set_xlabel('')
axes[0].set_ylabel('Stay Probability', fontsize=24)
axes[0].set_title('Two Step Task', y=1.04, fontsize=30)
axes[0].set_ylim([.5,1])
axes[0].tick_params(labelsize=20)
leg = axes[0].get_legend()
leg.set_title('')
beautify_legend(leg, colors=colors, fontsize=20)

#shift
sns.pointplot('trials_since_switch', 'correct', data=shift_df, ax=axes[1])
axes[1].set_xticks(range(0,25,5))
axes[1].set_xticklabels(range(0,25,5))
axes[1].set_xlabel('Trials After Change-Point', fontsize=24)
axes[1].set_ylabel('Percent Correct', fontsize= 24)
axes[1].set_title('Shift Task', y=1.04, fontsize=30)
axes[1].tick_params(labelsize=20)
save_dir = path.join(base_dir, 'Results', 'replication', 'Plots', 'successful_learning_tasks.%s' % ext)
f.savefig(save_dir, dpi=300, bbox_inches='tight')
plt.close()

# *************************************************************************
# Unsuccessful replications
# ************************************************************************
# hierarchical
hierarchical_df = get_behav_data(file='Individual_Measures/hierarchical_rule.csv.gz')
successful_hierarchical = data.filter(regex='hierarchical_rule').dropna(how='any').index
hierarchical_df = hierarchical_df.query('worker_id in %s' % list(successful_hierarchical))
hierarchical_df = hierarchical_df.query('rt != -1 ')

# probabilistic_selection
prob_select_df = get_behav_data(file='Individual_Measures/probabilistic_selection.csv.gz')
successful_prob_select = data.filter(regex='probabilistic_selection').dropna(how='any').index
prob_select_df = prob_select_df.query('worker_id in %s' % list(successful_prob_select))
prob_select_df = prob_select_df.query('rt != -1 ')
prob_select_df.query('rt!=-1').groupby('worker_id').correct.mean().mean()

# plot
f, axes = plt.subplots(1,2,figsize=(20,8))
# hierarchical
hierarchical_df.query('trial_num>=259 and rt!=-1').groupby('worker_id') \
    .correct.mean().hist(bins=20, ax=axes[0])

axes[0].set_xlabel('Accuracy in last 100 trials')
axes[0].set_ylabel('# of Participants', fontsize=24)
axes[0].set_title('Hierarchical Rule Task', y=1.04, fontsize=30)
axes[0].tick_params(labelsize=20)

#probabilistic selection

save_dir = path.join(base_dir, 'Results', 'replication', 'Plots', 'unsuccessful_learning_tasks.%s' % ext)
f.savefig(save_dir, dpi=300, bbox_inches='tight')
plt.close()