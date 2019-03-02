# ****************************************************************************
# Helper functions for prediction
# ****************************************************************************
import glob
import numpy as np
import os
import pandas as pd
import pickle
from selfregulation.prediction.behavpredict import BehavPredict

def run_prediction(predictors, targets, output_base='', 
                   outfile='prediction', save=True,
                   verbose=False, classifier='lasso',
                   shuffle=False, n_jobs=2, imputer="SoftImpute",
                   smote_cutoff=.3, freq_threshold=.1,
                   binarize=True):
    
    output_dir=os.path.join(output_base,'prediction_outputs')
    
    bp = BehavPredict(behavdata=predictors,
                      targetdata=targets,
                      classifier=classifier,
                      output_dir=output_dir,
                      outfile=outfile,
                      shuffle=shuffle,
                      n_jobs=n_jobs,
                      imputer=imputer,
                      smote_cutoff=smote_cutoff,
                      freq_threshold=freq_threshold)
    if binarize:
        bp.binarize_ZI_demog_vars()
    vars_to_test=[v for v in bp.targetdata.columns if not v in bp.skip_vars]
    for v in vars_to_test:
        # run regression into non-null number is found. Should only be run once!
        # but occasionally a nan is returned for some reason
        cv_scores = insample_scores = [np.nan, np.nan]
        cv_scores = bp.run_crossvalidation(v,nlambda=100)
        insample_scores, importances, clf = bp.run_prediction(v)
        if verbose:
            print('Predicting %s' % v)
            if 'R2' in cv_scores[0].keys():
                if pd.isnull(cv_scores[0]['R2']):
                    print('No predictor variance in CV model!')
                if pd.isnull(insample_scores[0]['R2']):
                    print('No predictor variance in insample model!')
        bp.scores[v], bp.importances[v], bp.clfs[v] = cv_scores, importances, clf
        bp.scores_insample[v] = insample_scores
    if save == True:
        bp.write_data(vars_to_test)
    return bp.get_output(vars_to_test)

def print_prediction_performance(results, EFA=True):
    for classifier in ['ridge', 'lasso', 'rf', 'svm']:
        print(classifier)
        out = results.load_prediction_object(classifier=classifier,
                                             EFA=EFA)['data']
        keys = ['Binge Drinking', 'Problem Drinking', 
                'Drug Use', 'Lifetime Smoking', 'Daily Smoking', 
                'Mental Health', 'Obesity', 'Income / Life Milestones']
        for key in keys:
            val = out[key]
            print(key)
            s = ('R2 =\n  %.2f (%.2f)\nMAE =\n  %.2f (%.2f)\n\n' % 
                  (val['scores_cv'][0]['R2'],
                   val['scores_insample'][0]['R2'],
                   val['scores_cv'][0]['MAE'],
                   val['scores_insample'][0]['MAE']))
            print(s.replace('0.', '.'))
        print('*'*40)

def run_group_prediction(all_results, shuffle=False, classifier='lasso',
                       include_raw_demographics=False, rotate='oblimin',
                       verbose=False, save=True):
    if verbose:
        print('*'*79)
        print('Running Prediction, shuffle: %s, classifier: %s' % (shuffle, classifier))
        print('*'*79)
    
    names = [r.ID.split('_')[0] for r in all_results.values()]
    name = '_'.join(sorted(names)[::-1])
    factor_scores = pd.concat([r.EFA.get_scores(rotate=rotate) 
                                for r in all_results.values()], axis=1)
    tmp_results = list(all_results.values())[0]
    output_dir = os.path.dirname(tmp_results.get_output_dir())
    demographics = tmp_results.DA
    demographic_factors = demographics.reorder_factors(demographics.get_scores())

    
    targets = [('demo_factors', demographic_factors)]
    if include_raw_demographics:
        targets.append(('demo_raw', tmp_results.demographics))
    out = {}
    if shuffle is False:
        shuffle_flag = '_'
    else:
        shuffle_flag = '_shuffled_'
    for target_name, target in targets:
        predictors = ('EFA_%s_%s' % (name, rotate), factor_scores)
        # predicting using best EFA
        if verbose: print('**Predicting using %s**' % predictors[0])
        prediction = run_prediction(predictors[1], 
                        target, 
                        output_dir,
                        outfile='%s_%s%sprediction' % (predictors[0], target_name, shuffle_flag), 
                        shuffle=shuffle,
                        classifier=classifier, 
                        verbose=verbose, 
                        save=save)
        out[target_name] = prediction
    return out
    
def print_group_prediction(prediction_loc):
    for classifier in ['ridge', 'lasso', 'rf', 'svm']:
        files = glob.glob(os.path.join(prediction_loc, '*task_survey*%s*' % classifier))
        files = [f for f in files if 'shuffle' not in f]
        files.sort(key=os.path.getmtime)
        out = pickle.load(open(files[-1], 'rb'))['data']
        print(classifier)
        keys = ['Binge Drinking', 'Problem Drinking',
                'Drug Use', 'Lifetime Smoking', 'Daily Smoking', 
                'Mental Health', 'Obesity', 'Income / Life Milestones']
        for key in keys:
            val = out[key]
            print(key)
            s = ('R2 =\n  %.2f (%.2f)\nMAE =\n  %.2f (%.2f)\n\n' % 
                  (val['scores_cv'][0]['R2'],
                   val['scores_insample'][0]['R2'],
                   val['scores_cv'][0]['MAE'],
                   val['scores_insample'][0]['MAE']))
            print(s.replace('0.', '.'))
        print('*'*40)