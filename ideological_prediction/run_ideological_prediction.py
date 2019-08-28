# remove sklearn deprecation warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# imports
import argparse
# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-skip_factors', action='store_true')
parser.add_argument('-run_raw', action='store_false')
parser.add_argument('-classifier', default='ridge')
parser.add_argument('-raw_classifier', default='lasso')
parser.add_argument('-EFA_rotation', default='oblimin')
parser.add_argument('-shuffle_repeats', type=int, default=1)
args = parser.parse_args()
    
from fancyimpute import SoftImpute
import json
from os import makedirs, path
import numpy as np
import pandas as pd
import pickle

from dimensional_structure.prediction_utils import run_prediction
from dimensional_structure.utils import residualize_baseline
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_behav_data, get_recent_dataset, get_info, get_demographics


# parse args
run_factors = not args.skip_factors
run_raw = not args.run_raw
classifier = args.classifier
raw_classifier = args.raw_classifier
shuffle_reps = args.shuffle_repeats
EFA_rotation = args.EFA_rotation

results_dir = path.join(get_info('results_directory'), 'ideology_prediction')
makedirs(results_dir, exist_ok=True)

# load data
dataset = get_recent_dataset()
results = load_results(dataset)
ideo_data = get_behav_data(dataset, file='ideology.csv')

# get demographics 
ideo_demographics = get_behav_data(dataset, file='ideology_demographics.csv')
# fill in ideo demographics from demographics if needed
demographics = get_demographics()
# fill gender
missing_gender = ideo_demographics[ideo_demographics['Gender'].isnull()].index
ideo_demographics.loc[missing_gender, 'Gender'] = demographics.loc[missing_gender, 'Sex']
# Age can be off by a year potentially by the time the ideological data was collected
missing_age = ideo_demographics[ideo_demographics['Age'].isnull()].index
ideo_demographics.loc[missing_age, 'Age'] = demographics.loc[missing_age, 'Age']

# reduce dataset to where we have full demographics
ideo_demographics = ideo_demographics[ideo_demographics.isnull().sum(1)==0]
ideo_data = ideo_data.loc[ideo_demographics.index]




# define targets
targets = {'ideo_factors': ideo_data.filter(regex='Factor'),
           'ideo_orientations': ideo_data.drop(ideo_data.filter(regex='Factor|SECS').columns, axis=1).drop(['Conservatism','Intellectual Humility'], axis=1),
           'ideo_policies': ideo_data.filter(regex='SECS')}
for key, target in targets.items():
    imputed = pd.DataFrame(SoftImpute().fit_transform(target),
                            index=target.index,
                            columns=target.columns)
    
    # residualize
    imputed = residualize_baseline(imputed.join(ideo_demographics), 
                                                baseline_vars=ideo_demographics.columns)
    targets[key] = imputed+1E-5

# ****************************************************************
# run prediction
# ****************************************************************
predictors = {}
predictions = {}
shuffled_predictions = {}
# do prediction with ontological factors
if run_factors:
    # define predictors
    survey_scores = results['survey'].EFA.get_scores(rotate=EFA_rotation)
    task_scores = results['task'].EFA.get_scores(rotate=EFA_rotation)
    predictors.update({'survey_%s' % EFA_rotation: survey_scores,
                      'task_%s' % EFA_rotation: task_scores,
                      'demographics': results['task'].DA.get_scores(),
                      'full_ontology_%s' % EFA_rotation: pd.concat([survey_scores, task_scores], axis=1)})
    for predictor_key, scores in predictors.items():
        for target_key, target in targets.items():
            print('*'*79)
            print('Running Prediction: %s predicting %s' % (predictor_key, target_key))
            print('*'*79)
            predictions[(predictor_key, target_key)] = \
                        run_prediction(scores, 
                                       target, 
                                       results_dir,
                                       outfile='%s_%s_prediction' % (predictor_key, target_key), 
                                       shuffle=False,
                                       classifier=classifier, 
                                       verbose=True,
                                       save=True,
                                       binarize=False)['data']
            print('\nRunning Shuffle Prediction\n')
            shuffled_predictions[(predictor_key, target_key)] = \
                                run_prediction(scores, 
                                               target, 
                                               results_dir,
                                               outfile='%s_%s_prediction' % (predictor_key, target_key), 
                                               shuffle=shuffle_reps,
                                               classifier=classifier, 
                                               verbose=True,
                                               save=True,
                                               binarize=False)['data']

if run_raw:
    # predictions with raw measures
    predictor_key = 'raw_measures'
    DV_scores = get_behav_data(file='meaningful_variables_imputed.csv')
    predictors['raw_measures'] = DV_scores
    for target_key, target in targets.items():
        print('*'*79)
        print('Running Prediction: raw measures predicting %s' % target_key)
        print('*'*79)
        predictions[(predictor_key, target_key)] = \
                    run_prediction(DV_scores, 
                                   target, 
                                   results_dir,
                                   outfile='%s_%s_prediction' % (predictor_key, target_key), 
                                   shuffle=False,
                                   classifier=raw_classifier, 
                                   verbose=True,
                                   save=True,
                                   binarize=False)['data']
        print('\nRunning Shuffle Prediction\n')
        shuffled_predictions[(predictor_key, target_key)] = \
                            run_prediction(DV_scores, 
                                           target, 
                                           results_dir,
                                           outfile='%s_%s_prediction' % (predictor_key, target_key), 
                                           shuffle=shuffle_reps,
                                           classifier=raw_classifier, 
                                           verbose=True,
                                           save=True,
                                           binarize=False)['data']                           
  
# ****************************************************************
# save
# ****************************************************************
filename = path.join(results_dir, 'ideo_predictions.pkl')
if path.exists(filename):
    data = pickle.load(open(filename, 'rb'))
else:
    data = {'all_predictions': {},
            'all_shuffled_predictions': {},
            'predictors': {},
            'targets': {}}
# update data
data['all_predictions'].update(predictions)
data['all_shuffled_predictions'].update(shuffled_predictions)
data['predictors'].update(predictors)
data['targets'].update(targets)
                        
# save all results
pickle.dump(data,  open(filename, 'wb'))

# save results in easier-to-use format
simplified = {}
simplified_importances = {}
predictor_importances = {}
for p in data['predictors'].keys():
    simplified[p] = {}
    simplified
    predictor_importances[p] = {}
    for t in data['targets'].keys():
        # get scores
        tmp = data['all_predictions'][(p,t)]
        tmp_scores = {'CV_'+k:tmp[k]['scores_cv'][0]['R2'] for k in tmp.keys()}
        tmp_scores.update({'insample_'+k:tmp[k]['scores_insample'][0]['R2'] for k in tmp.keys()})
        simplified[p].update(tmp_scores)
        # get 95% shuffled performance
        shuffled_scores = {}
        for k in tmp.keys():
            shuffled_score = data['all_shuffled_predictions'][(p,t)][k]['scores_cv']
            shuffled_95th = np.percentile( [i['R2'] for i in shuffled_score], 95)
            shuffled_scores['shuffled95_' + k] = shuffled_95th
        simplified[p].update(shuffled_scores)
        # get importances
        for k in tmp.keys():
            importances = tmp[k]['importances'][0]
            predvars = tmp[k]['predvars']
            non_zero = np.where(importances)[0]
            zipped = list(zip([predvars[i] for i in non_zero],
                                           importances[non_zero]))
            predictor_importances[p][k] = sorted(zipped, 
                                                 key = lambda x: abs(x[1]), 
                                                 reverse=True)
simplified['Target_Cat'] = {}
for t,vals in data['targets'].items():
    simplified['Target_Cat'].update({'CV_' + c:t for c in vals.columns})
    simplified['Target_Cat'].update({'insample_' + c:t for c in vals.columns})
    simplified['Target_Cat'].update({'shuffled95_' + c:t for c in vals.columns})

simplified=pd.DataFrame(simplified)
simplified.insert(0,'prediction_type', [i[0] for i in simplified.index.str.split('_')])
simplified.to_csv(path.join(results_dir, 
                            'predictions_R2.csv'))
json.dump(predictor_importances, 
          open(path.join(results_dir, 
                         'predictor_importances.json'), 'w'))






