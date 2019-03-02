
# coding: utf-8

# This notebook assesses the ability to predict demographic outcomes from survey data.

# In[1]:

import os,glob,sys,pickle
import importlib

import warnings
import numpy,pandas
from sklearn.svm import LinearSVC,SVC,OneClassSVM
from sklearn.linear_model import LinearRegression,LogisticRegressionCV,RandomizedLogisticRegression,ElasticNet,ElasticNetCV,Ridge,RidgeCV
from sklearn.preprocessing import scale,StandardScaler,FunctionTransformer
from sklearn.decomposition import FactorAnalysis,PCA
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV,StratifiedShuffleSplit,cross_val_score,StratifiedKFold
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import fancyimpute
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import scipy.stats

def print_confusion_matrix(y_true,y_pred,labels=[0,1]):
    cm=confusion_matrix(y_true,y_pred)
    print('Confusion matrix')
    print('\t\tPredicted')
    print('\t\t0\t1')
    print('Actual\t0\t%d\t%d'%(cm[0,0],cm[0,1]))
    print('\t1\t%d\t%d'%(cm[1,0],cm[1,1]))

# this is kludgey but it works
sys.path.append('../utils')
import utils
importlib.reload(utils)

#warnings.filterwarnings("ignore") # only turn this on in production mode
                                  # to keep log files from overflowing

nruns=10
dataset='Discovery_9-26-16'
basedir=utils.get_info('base_directory')
derived_dir=os.path.join(basedir,'Data/Derived_Data/%s'%dataset)

if not os.path.exists('surveypred'):
    os.mkdir('surveypred')

try:
     binary_vars=[sys.argv[1]]
except:
     print('running for all variables')
     binary_vars=["Sex","ArrestedChargedLifeCount","DivorceCount","GamblingProblem","ChildrenNumber",
            "CreditCardDebt","RentOwn","RetirementAccount","TrafficTicketsLastYearCount","Obese",
             "TrafficAccidentsLifeCount","CaffienatedSodaCansPerDay","Nervous",
             'Hopeless', 'RestlessFidgety', 'Depressed',
             'EverythingIsEffort', 'Worthless','CigsPerDay','LifetimeSmoke100Cigs',
             'CannabisPast6Months']
     binary_vars=["Sex"]


if len(sys.argv)>2:
    shuffle=True
    shufflenum=int(sys.argv[2])
else:
    shuffle=False

verbose=False

# for testing
#shuffle=True
#shufflenum=1

if shuffle:
    shuffletag='_shuffle%04d'%shufflenum
else:
    shuffletag=''

def softimpute(X, y=None):
    return(fancyimpute.SoftImpute(verbose=False).complete(X))

imputer = FunctionTransformer(softimpute, validate=False)

# for some items, we want to use somethign other than the minimum as the
# cutoff:
item_thresholds={'Nervous':1,
                'Hopeless':1,
                'RestlessFidgety':1,
                'Depressed':1,
                'EverythingIsEffort':1,
                'Worthless':1}

def get_demog_data(binary_vars=binary_vars,ordinal_vars=[],item_thresholds={},binarize=True):
    demogdata=pandas.read_csv(os.path.join(derived_dir,
        'surveydata/demographics.tsv'),index_col=0,delimiter='\t')
    healthdata=pandas.read_csv(os.path.join(derived_dir,
        'surveydata/health_ordinal.tsv'),index_col=0,delimiter='\t')
    alcdrugs=pandas.read_csv(os.path.join(derived_dir,
        'surveydata/alcohol_drugs_ordinal.tsv'),index_col=0,delimiter='\t')
    assert all(demogdata.index==healthdata.index)
    assert all(demogdata.index==alcdrugs.index)
    demogdata=demogdata.merge(healthdata,left_index=True,right_index=True)
    demogdata=demogdata.merge(alcdrugs,left_index=True,right_index=True)
    # remove a couple of outliers - this is only for cases when we include BMI/obesity
    if 'BMI' in ordinal_vars or 'Obese' in binary_vars:
        demogdata=demogdata.query('WeightPounds>50')
        demogdata=demogdata.query('HeightInches>36')
        demogdata=demogdata.query('CaffienatedSodaCansPerDay>-1')
        demogdata=demogdata.assign(BMI=demogdata['WeightPounds']*0.45 / (demogdata['HeightInches']*0.025)**2)
        demogdata=demogdata.assign(Obese=(demogdata['BMI']>30).astype('int'))

    if binarize:
        demogdata=demogdata[binary_vars]
        demogdata=demogdata.loc[demogdata.isnull().sum(1)==0]

        for i in range(len(binary_vars)):
            v=binary_vars[i]
            if v in item_thresholds:
                threshold=item_thresholds[v]
            else:
                threshold=demogdata[v].min()
            demogdata.loc[demogdata[v]>threshold,v]=1
            assert demogdata[v].isnull().sum()==0
    return demogdata


def get_joint_dataset(d1,d2):
    d1_index=set(d1.index)
    d2_index=set(d2.index)
    inter=list(d1_index.intersection(d2_index))
    return d1.ix[inter],d2.ix[inter]
    return inter

behavdata_orig=utils.get_behav_data('Discovery_9-26-16',use_EZ=True)

demogdata,behavdata=get_joint_dataset(get_demog_data(),behavdata_orig)
assert list(demogdata.index)==list(behavdata.index)
behavvars=list(behavdata.columns)
if verbose:
    print('%d joint subjects found'%demogdata.shape[0])
    print('%d task variables found'%len(behavvars))
    print('Demographic variables to test:')
    print(list(demogdata.columns))


# First get binary variables and test classification based on survey data.
# Only include variables that have at least 10% of the infrequent category.
# Some of these were not collected as binary variables, but we binarize by
# calling anything above the minimum value a positive outcome.

bvardata=numpy.array(demogdata)
sdata_orig=numpy.array(behavdata).copy()

# just impute randomly here, since we don't have a way to work our
# imputation into the CV loop

#nanvals=numpy.where(numpy.isnan(sdata_orig))
sdata=sdata_orig
results=pandas.DataFrame(columns=['variable','fa_ctr','trainf1','testf1'])

clf_params={}

ctr=0

accuracy={}
imp={}
null_accuracy={}
cutoff={}
importances={}

def inner_cv_loop(Xtrain,Ytrain,clf,parameters=[],
                    verbose=False):
    """
    use GridSearchCV to find best classifier for training set
    """
    if parameters:
        gs=GridSearchCV(clf,parameters,scoring='roc_auc',
                        cv=StratifiedShuffleSplit(n_splits=5,test_size=0.2))
    else:
        gs=GridSearchCV(clf,scoring='roc_auc',
                        cv=StratifiedShuffleSplit(n_splits=5,test_size=0.2))

    gs.fit(Xtrain,Ytrain)
    if verbose:
        print('best:',gs.best_score_,gs.best_estimator_)

    return gs.best_estimator_,gs.best_score_

def outer_cv_loop(Xdata,Ydata,clf,parameters=[],
                    n_splits=10,test_size=0.25):

    pred=numpy.zeros(len(Ydata))
    importances=[]
    kf=StratifiedShuffleSplit(n_splits=n_splits,test_size=test_size)
    rocscores=[]
    for train,test in kf.split(Xdata,Ydata):
        if numpy.var(Ydata[test])==0:
           print('zero variance',varname)
           rocscores.append(numpy.nan)
           continue
        Ytrain=Ydata[train]
        Xtrain=fancyimpute.SoftImpute(verbose=False).complete(Xdata[train,:])
        Xtest=fancyimpute.SoftImpute(verbose=False).complete(Xdata[test,:])
        if numpy.abs(numpy.mean(Ytrain)-0.5)>0.2:
           smt = SMOTETomek()
           Xtrain,Ytrain=smt.fit_sample(Xtrain.copy(),Ydata[train])
        # filter out bad folds
        clf.fit(Xtrain,Ytrain)
        pred=clf.predict(Xtest)
        if numpy.var(pred)>0:
           rocscores.append(roc_auc_score(Ydata[test],pred))
        else:
           rocscores.append(numpy.nan)
        importances.append(clf.feature_importances_)
    return rocscores,importances

# set up classifier
forest = ExtraTreesClassifier(n_estimators=250,n_jobs=-1)
estimators = [('imputer',imputer),('clf',forest)]
pipeline=Pipeline(steps=estimators)

for varname in binary_vars:
    accuracy[varname]=[]
    importances[varname]=[]
    y=numpy.array(demogdata[varname].copy())
    if numpy.var(y)==0:
        print('skipping %s: no variance'%varname)
        print('')
        continue

    for i in range(nruns):
        if shuffle:
            numpy.random.shuffle(y)
            print('y shuffled')
        roc_scores,imp=outer_cv_loop(sdata,y,forest)
        importances[varname].append(imp)

        accuracy[varname].append(roc_scores)

    print('OUTPUT:',varname,shuffle,numpy.mean(accuracy[varname]),
            numpy.min(accuracy[varname]),numpy.max(accuracy[varname]))


    # Print the feature ranking
    if verbose:
        print("Feature ranking:")
        for f in range(5):
            print("%s. feature %d (%f)" % (behavvars[indices[f]], indices[f],
                                importances[varname][indices[f]]))
        print('')

    import pickle
    pickle.dump((accuracy,importances),
        open('behavpred/pipeline_%s_performance%s.pkl'%(varname,shuffletag),'wb'))

if verbose:
    (accuracy,importances,null_accuracy,cutoff)=pickle.load(
            open('behavpred/pipeline_%s_performance%s.pkl'%(varname,shuffletag),'rb'))
    for varname in binary_vars:
        if not varname in importances:
            continue
        indices = numpy.argsort(importances[varname])[::-1]
        print('')
        print('Variable:',varname)
        print('Prediction accuracy (ROC AUC):',numpy.mean(accuracy[varname][:10]))
        pval=1-scipy.stats.percentileofscore(null_accuracy[varname],
                    numpy.mean(accuracy[varname][:10]))/100.
        print('null mean %f, pval: %f'%(numpy.mean(null_accuracy[varname]),pval))
        if pval<0.1:
            # Print the feature ranking
            print("Feature ranking:")
            for f in range(5):
                print("%s. feature %d (%f)" % (behavvars[indices[f]], indices[f],
                            importances[varname][indices[f]]))
            print('')
