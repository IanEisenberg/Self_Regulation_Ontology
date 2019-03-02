import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import scale

from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_recent_dataset, get_behav_data

# lookup
fmri_contrasts = ['ANT_orienting_network',
 'ANT_conflict_network',
 'ANT_response_time',
 'CCTHot_EV',
 'CCTHot_risk',
 'CCTHot_response_time',
 'discountFix_subjective_value',
 'discountFix_LL_vs_SS',
 'discountFix_response_time',
 'DPX_BX-BY',
 'DPX_AY-BY',
 'DPX_AY-BX',
 'DPX_BX-AY',
 'DPX_response_time',
 'motorSelectiveStop_crit_stop_success-crit_go',
 'motorSelectiveStop_crit_stop_failure-crit_go',
 'motorSelectiveStop_crit_go-noncrit_nosignal',
 'motorSelectiveStop_noncrit_signal-noncrit_nosignal',
 'motorSelectiveStop_crit_stop_success-crit_stop_failure',
 'motorSelectiveStop_crit_stop_failure-crit_stop_success',
 'motorSelectiveStop_crit_stop_success-noncrit_signal',
 'motorSelectiveStop_crit_stop_failure-noncrit_signal',
 'stroop_incongruent-congruent',
 'stroop_response_time',
 'surveyMedley_response_time',
 'twoByTwo_cue_switch_cost_100',
 'twoByTwo_cue_switch_cost_900',
 'twoByTwo_task_switch_cost_100',
 'twoByTwo_task_switch_cost_900',
 'twoByTwo_response_time',
 'WATT3_search_depth']
 
fmri_ontology_mapping = {
     'ANT_conflict_network': 'attention_network_task.conflict_hddm_drift',
     'ANT_orienting_network': 'attention_network_task.orienting_hddm_drift',
     'ANT_response_time': 'attention_network_task.avg_rt',
     'DPX_AY-BY': 'dot_pattern_expectancy.AY-BY_hddm_drift',
     'DPX_BX-BY': 'dot_pattern_expectancy.BX-BY_hddm_drift',
     'DPX_response_time': 'dot_pattern_expectancy.avg_rt',
     'motorSelectiveStop_crit_go-noncrit_nosignal': 'motor_selective_stop_signal.proactive_control_hddm_drift',
     'motorSelectiveStop_noncrit_signal-noncrit_nosignal': 'motor_selective_stop_signal.reactive_control_hddm_drift',
     'motorSelectiveStop_crit_stop_success-crit_stop_failure': 'motor_selective_stop_signal.SSRT',
     'stroop_incongruent-congruent': 'stroop.stroop_hddm_drift',
     'stroop_response_time': 'stroop.avg_rt',
     'twoByTwo_cue_switch_cost_100': 'threebytwo.cue_switch_cost_hddm_drift',
     'twoByTwo_task_switch_cost_100': 'threebytwo.task_switch_cost_hddm_drift',
     'twoByTwo_response_time': 'threebytwo.avg_rt',
 }

# helper functions
def run_linear(data, scores, clf=RidgeCV(fit_intercept=False)):
    """
    Run Knearest neighbor using precomputed distances to create an ontological mapping
    
    Args:
        data: dataframe with variables to reconstruct as columns
        scores: ontological scores
        clf: linear model that returns coefs
    """
    y=scale(data)
    clf.fit(scores, y)

    out = clf.coef_
    if len(out.shape)==1:
        out = out.reshape(1,-1)
    out = pd.DataFrame(out, columns=scores.columns)
    out.index = data.columns
    return out

# do mapping
dataset = get_recent_dataset()
# load ontology
results = load_results(datafile=dataset)
task_loadings = results['task'].EFA.get_loading()
task_scores = results['task'].EFA.get_scores()
# load all DVs
all_DVs = get_behav_data(file='variables_exhaustive.csv', dataset=dataset)



contrast_loadings = {}
for contrast, relation in fmri_ontology_mapping.items():
    # if relation is in the direct mapping
    if relation.lstrip('-') in task_loadings.index:
        task_loading = task_loadings.loc[relation.lstrip('-')]
        if relation[0] == '-':
            task_loading = task_loading*-1
    # otherwise, reconstruct!
    else:
        unmapped_data = all_DVs.loc[:,relation.lstrip('-')]
        missing = unmapped_data[unmapped_data.isnull()].index
        task_loading = run_linear(pd.DataFrame(unmapped_data.drop(missing)), 
                                  task_scores.drop(missing)).iloc[0,:]
    contrast_loadings[contrast] = task_loading

# calculate new loadings for DVs not in ontology already



contrast_loadings = pd.DataFrame(contrast_loadings).T
#contrast_loadings.to_pickle('/home/ian/Experiments/Self_Regulation_Ontology_fMRI/fmri_analysis/scripts/tmp.pkl')