import numpy as np
import pandas as pd
from selfregulation.utils.result_utils import load_results
from dimensional_structure.prediction_utils import run_prediction

results = load_results('Complete_02-03-2018')['task']
final_df = pd.DataFrame()
for classifier in ['ridge', 'lasso']:
    full = results.load_prediction_object(classifier=classifier, EFA=False)
    # get no discount data
    no_discount_predictors = results.data
    no_discount_predictors.drop(results.data.filter(regex='holt|kirby|discount').columns, 
                                axis=1,
                                inplace=True)
    wD = full['data']
    nD = run_prediction(no_discount_predictors, results.DA.get_scores(),
                        classifier=classifier, save=False)
    
    # insample
    wD_insample = sorted([(k,i['scores_insample'][0]*100) for k,i in wD.items()], key=lambda x:x[0])
    nD_insample = sorted([(k,i[0]*100) for k,i in nD.scores_insample.items()], key = lambda x: x[0])
    insample_df = pd.DataFrame(wD_insample, columns=['target','wD'])
    insample_df.insert(2, 'nD', [i[1] for i in nD_insample])
    insample_df = insample_df.assign(type='insample')
    insample_df = insample_df.assign(classifier=classifier)
        
    # CV
    wD_CV = sorted([(k,i['scores_cv'][0]*100) for k,i in wD.items()], key=lambda x:x[0])
    nD_CV = sorted([(k,i[0]*100) for k,i in nD.scores.items()], key = lambda x: x[0])
    CV_df = pd.DataFrame(wD_CV, columns=['target','wD'])
    CV_df.insert(2, 'nD', [i[1] for i in nD_CV])
    CV_df = CV_df.assign(type='CV')
    CV_df = CV_df.assign(classifier=classifier)
    
    df = pd.concat([insample_df, CV_df], axis=0)
    df.insert(3, 'diff', np.round(df.wD-df.nD,4))
    final_df = pd.concat([final_df, df])