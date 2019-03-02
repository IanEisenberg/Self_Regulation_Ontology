"""
perform classification on demographic data
after binarizing continuous variables
"""

import sys,os
import random
import pickle
import importlib

import numpy

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,ShuffleSplit

from selfregulation.utils.get_balanced_folds import BalancedKFold
import selfregulation.prediction.behavpredict_V1 as behavpredict
importlib.reload(behavpredict)

if __name__=='__main__':
    # variables to be binarized for classification - dictionary
    # with threshold for each variable

    bp=behavpredict.BehavPredict(verbose=False,use_smote=False)
    bp.load_demog_data(binarize=False)
    bp.load_behav_data('task')
    bp.get_joint_datasets()
    taskdata=bp.behavdata.copy()
    bp.load_behav_data('survey')
    bp.get_joint_datasets()
    surveydata=bp.behavdata.copy()
    demogdata=bp.demogdata.copy()
    assert all(demogdata.index == surveydata.index)
    assert all(demogdata.index == taskdata.index)
    outdir='../Data/Derived_Data/%s'%bp.dataset
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    demogdata.to_csv('%s/demogdata_for_prediction.csv'%outdir)
    surveydata.to_csv('%s/surveydata_for_prediction.csv'%outdir)
    taskdata.to_csv('%s/taskdata_for_prediction.csv'%outdir)
