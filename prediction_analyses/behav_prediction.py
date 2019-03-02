"""
perform prediction on demographic data

use different strategy depending on the nature of the variable:
- lasso classification (logistic regression) for binary variables
- lasso regression for normally distributed variables
- lasso-regularized zero-inflated poisson regression for zero-inflated variables
-- via R mpath library using rpy2

compare each model to a baseline with age and sex as regressors

TODO:
- add metadata including dataset ID into results output
- break icc thresholding into separate method
- use a better imputation method than SimpleFill
"""

import sys,os
import random
import pickle
import importlib
import traceback

import numpy

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from selfregulation.utils.utils import get_info
import selfregulation.prediction.behavpredict_V1 as behavpredict
importlib.reload(behavpredict)

import argparse
import fancyimpute

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v',"--verbose", help="increase output verbosity",
                        default=0, action='count')
    parser.add_argument('-c',"--classifier", help="classifier",
                            default='lasso')

    parser.add_argument("--report_features", help="print features",
                        action='store_true')
    parser.add_argument("--print_report", help="print report at the end",
                        action='store_true')
    parser.add_argument('-s',"--shuffle", help="shuffle target variable",
                        action='store_true')
    parser.add_argument('-i',"--icc_threshold", help="threshold for ICC filtering",
                        type=float,default=0.25)
    parser.add_argument("--freq_threshold", help="threshold for binary variable frequency",
                        type=float,default=0.1)
    parser.add_argument("--no_baseline_vars",
                        help="don't include baseline vars in task/survey model",
                        action='store_true')
    parser.add_argument("--debugbreak",
                        help="break after setting up class",
                        action='store_true')
    parser.add_argument("--demogfile",
                        help="use data from file for demog vars",
                        default=None)
    parser.add_argument('-d',"--dataset", help="dataset for prediction",
                            required=True)
    parser.add_argument('-j',"--n_jobs", help="number of processors",type=int,
                            default=2)
    parser.add_argument('-w',"--workdir", help="working directory")
    parser.add_argument('-r',"--resultsdir", help="results directory")
    parser.add_argument("--singlevar", nargs='*',help="run with single variables")
    parser.add_argument('--imputer',help='imputer to use',
                            default='SimpleFill')
    parser.add_argument("--smote_threshold", help="threshold for applying smote (distance from 0.5)",
                        type=float,default=0.05)
    args=parser.parse_args()

    # parameters to set

    if args.resultsdir is None:
        try:
            output_base=get_info('results_directory')
        except:
            output_base='.'
    else:
        output_base=args.resultsdir
    output_dir=os.path.join(output_base,'prediction_outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #assert args.dataset in ['survey','mirt','task','all','baseline']
    assert args.classifier in ['lasso','rf']
    # don't regress out baseline vars for baseline model
    if args.dataset=='baseline' or args.no_baseline_vars:
        baselinevars=False
        if args.verbose:
            print("turning off inclusion of baseline vars")
    else:
        baselinevars=True
        if args.verbose:
            print("including baseline vars in survey/task models")


    # skip several variables because they crash the estimation tool
    bp=behavpredict.BehavPredict(verbose=args.verbose,
         drop_na_thresh=100,n_jobs=args.n_jobs,
         skip_vars=['RetirementPercentStocks',
         'HowOftenFailedActivitiesDrinking',
         'HowOftenGuiltRemorseDrinking',
         'AlcoholHowOften6Drinks'],
         output_dir=output_dir,shuffle=args.shuffle,
         classifier=args.classifier,
         add_baseline_vars=baselinevars,
         smote_cutoff=args.smote_threshold,
         freq_threshold=args.freq_threshold,
         imputer=args.imputer)

    if args.debugbreak:
        raise Exception('breaking')

    if args.demogfile is not None:
        import pandas
        bp.demogdata=pandas.read_csv(args.demogfile,index_col=0)
        bp.data_models={}
        for i in bp.demogdata.columns:
            bp.data_models[i]='gaussian'
    else:
        bp.load_demog_data()
        bp.get_demogdata_vartypes()
        bp.remove_lowfreq_vars()
        bp.binarize_ZI_demog_vars()

    def add_varsets(bp,tags,taskname=None):
        vars=list(bp.behavdata.columns)
        for tag in tags:
            varsets=tags[tag]
            indvars=[]
            for v in vars:
                for vs in varsets:
                    if taskname is None:
                        if v.find(vs)>-1:
                            indvars.append(v)
                    else:
                        if v.find(vs)==0:
                            indvars.append(v)

            if bp.verbose:
                print(tag,indvars)
            bp.add_varset(tag,indvars)

    # create apriori variable subsets
    bp.load_behav_data('task')
    task_tags={'discounting':['discount'],
                'stopping':['stop_signal','nogo'],
                'intelligence':['raven','cognitive_reflection'],
                'drift':['hddm_drift'],
                'thresh':['hddm_thresh'],
                'nondecision':['hddm_non_decision']}
    add_varsets(bp,task_tags)
    task_name_tags={'stroop':['stroop'],
                'dot_pattern_expectancy':['dot_pattern_expectancy'],
                'attention_network_task':['attention_network_task'],
                'threebytwo':['threebytwo'],
                'stop_signal':['stop_signal'],
                'motor_selective_stop_signal':['motor_selective_stop_signal'],
                'kirby':['kirby'],
                'discount_titrate':['discount_titrate'],
                'tower_of_london':['tower_of_london'],
                'columbia_card_task_hot':['columbia_card_task_hot']}
    add_varsets(bp,task_name_tags,taskname=True)

    bp.load_behav_data('survey')
    survey_tags={'impulsivity':['upps_impulsivity_survey','dickman_survey',
                                'bis11_survey','self_regulation_survey',
                                'brief_self_control_survey'],
                'big5':['ten_item_personality_survey'],
                'risktaking':['sensation_seeking_survey','dospert'],
                'grit':['grit_scale_survey'],
                'emotion_regulation':['erq_survey'],
                'bisbas':['bis_bas_survey']
                }
    add_varsets(bp,survey_tags)

    bp.load_behav_data(args.dataset)
    if args.icc_threshold>0:
        bp.filter_by_icc(args.icc_threshold)
    bp.get_joint_datasets()


    if not args.singlevar:
        vars_to_test=list(bp.demogdata.columns)
    else:
        vars_to_test=args.singlevar

    for v in vars_to_test:
        bp.lambda_optim=None
        print('RUNNING:',v,bp.data_models[v],args.dataset)
        try:
            bp.scores[v],bp.importances[v]=bp.run_crossvalidation(v,nlambda=100)
            bp.scores_insample[v],_=bp.run_lm(v,nlambda=100)
            # fit model with no regularization
            if bp.data_models[v]=='binary':
                bp.lambda_optim=[0]
            else:
                bp.lambda_optim=[0,0]
            bp.scores_insample_unbiased[v],_=bp.run_lm(v,nlambda=100)
        except:
            e = sys.exc_info()
            print('error on',v,':',e)
            bp.errors[v]=traceback.format_tb(e[2])

    if args.singlevar:
        bp.write_data(vars_to_test,listvar=True)
    else:
        bp.write_data(vars_to_test)
