from expanalysis.experiments.processing import calc_exp_DVs

from math import floor, ceil
import numpy
import pandas 
import statsmodels.formula.api as smf
import statsmodels.api as sm
from selfregulation.utils.utils import get_behav_data


def group_decorate(group_fun = None):
    """ Group decorate is a wrapper for multi_worker_decorate to pass an optional group level
    DV function
    :group_fun: a function to apply to the entire group that returns a dictionary with DVs
    for each subject (i.e. fit_HDDM)
    """
    def multi_worker_decorate(fun):
        """Decorator to ensure that dv functions (i.e. calc_stroop_DV) have only one worker
        :func: function to apply to each worker individuals
        """
        def multi_worker_wrap(group_df, use_check = False, use_group_fun = True):
            exps = group_df.experiment_exp_id.unique()
            group_dvs = {}
            if len(group_df) == 0:
                return group_dvs, ''
            if len(exps) > 1:
                print('Error - More than one experiment found in dataframe. Exps found were: %s' % exps)
                return group_dvs, ''
            # remove practice trials
            if 'exp_stage' in group_df.columns:
                group_df = group_df.query('exp_stage != "practice"')
            # remove workers who haven't passed some check
            if 'passed_check' in group_df.columns and use_check:
                group_df = group_df[group_df['passed_check']]
            # apply group func if it exists
            if group_fun and use_group_fun:
                group_dvs = group_fun(group_df)
            # apply function on individuals
            for worker in pandas.unique(group_df['worker_id']):
                df = group_df.query('worker_id == "%s"' %worker)
                dvs = group_dvs.get(worker, {})
                try:
                    worker_dvs, description = fun(df, dvs)
                    group_dvs[worker] = worker_dvs
                except:
                    print('%s DV calculation failed for worker: %s' % (exps[0], worker))
            return group_dvs, description
        return multi_worker_wrap
    return multi_worker_decorate
    
@group_decorate()
def calc_dietary_decision_DV(df, dvs = {}):
    """ Calculate dv for dietary decision task. Calculate the effect of taste and
    health rating on choice
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df[~ pandas.isnull(df['taste_diff'])].reset_index(drop = True)
    df = df.query('mouse_click != "-1"')
    rs = smf.ols(formula = 'coded_response ~ health_diff + taste_diff', data = df).fit()
    dvs['health_sensitivity'] = {'value':  rs.params['health_diff'], 'valence': 'Pos'} 
    dvs['taste_sensitivity'] = {'value':  rs.params['taste_diff'], 'valence': 'Neg'} 
    description = """
        Both taste and health sensitivity are calculated based on the decision phase.
        On each trial the participant indicates whether they would prefer a food option
        over a reference food. Their choice is regressed on the subjective health and
        taste difference between that option and the reference item. Positive values
        indicate that the option's higher health/taste relates to choosing the option
        more often
    """
    return dvs,description
    
# get data
df = get_behav_data(dataset = 'Discovery_11-20-2016', file = 'Individual_Measures/dietary_decision.csv.gz')
demo = get_behav_data(dataset = 'Discovery_11-20-2016', file = 'demographic_targets.csv')

# calc DVs
DVs, description = calc_dietary_decision_DV(df)
for key,val in DVs.items():
    for subj_key in val.keys():
        val[subj_key]=val[subj_key]['value']
DVs = pandas.DataFrame.from_dict(DVs).T
        
# do it the simpler way     
DV, valence, description = calc_exp_DVs(df)