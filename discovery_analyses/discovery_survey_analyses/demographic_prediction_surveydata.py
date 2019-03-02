
# coding: utf-8

# This notebook assesses the ability to predict demographic outcomes from survey data.

# In[1]:

import os,glob,sys,pickle
import warnings
import numpy,pandas
from sklearn.svm import LinearSVC,SVC,OneClassSVM
from sklearn.linear_model import LinearRegression,LogisticRegressionCV,RandomizedLogisticRegression,ElasticNet,ElasticNetCV,Ridge,RidgeCV
from sklearn.preprocessing import scale
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from sklearn.decomposition import FactorAnalysis
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

def print_confusion_matrix(y_true,y_pred,labels=[0,1]):
    cm=confusion_matrix(y_true,y_pred)
    print('Confusion matrix')
    print('\t\tPredicted')
    print('\t\t0\t1')
    print('Actual\t0\t%d\t%d'%(cm[0,0],cm[0,1]))
    print('\t1\t%d\t%d'%(cm[1,0],cm[1,1]))

# this is kludgey but it works
sys.path.append('../utils')
from utils import get_info,get_survey_data

warnings.filterwarnings("ignore") # only turn this on in production mode
                                  # to keep log files from overflowing

dataset='Discovery_9-26-16'
basedir=get_info('base_directory')
derived_dir=os.path.join(basedir,'Data/Derived_Data/%s'%dataset)

if not os.path.exists('surveypred'):
    os.mkdir('surveypred')

binary_vars=["Sex","ArrestedChargedLifeCount","DivorceCount","GamblingProblem","ChildrenNumber",
            "CreditCardDebt","RentOwn","RetirementAccount","TrafficTicketsLastYearCount","Obese",
             "TrafficAccidentsLifeCount","CaffienatedSodaCansPerDay","Nervous",
             'Hopeless', 'RestlessFidgety', 'Depressed',
             'EverythingIsEffort', 'Worthless','CigsPerDay','LifetimeSmoke100Cigs',
             'CannabisPast6Months']
try:
    binary_vars=[sys.argv[1]]
except:
    print('specify variable as command line argument')
    binary_vars=['Sex'] #hsys.exit(1)

if len(sys.argv)>2:
    shuffle=True
    shuffletag='_shuffle%04d'%int(sys.argv[2])
else:
    shuffle=False
    shuffletag=''


nfeatures=5 # number of features to show
nfolds=4
verbose=False
simple_params=True

# for some items, we want to use somethign other than the minimum as the
# cutoff:
item_thresholds={'Nervous':1,
                'Hopeless':1,
                'RestlessFidgety':1,
                'Depressed':1,
                'EverythingIsEffort':1,
                'Worthless':1}

def get_demog_data(binary_vars=binary_vars,ordinal_vars=[],item_thresholds={},binarize=True):
    demogdata=pandas.read_csv(os.path.join(derived_dir,'surveydata/demographics.tsv'),index_col=0,delimiter='\t')
    healthdata=pandas.read_csv(os.path.join(derived_dir,'surveydata/health_ordinal.tsv'),index_col=0,delimiter='\t')
    alcdrugs=pandas.read_csv(os.path.join(derived_dir,'surveydata/alcohol_drugs_ordinal.tsv'),index_col=0,delimiter='\t')
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

surveydata_orig,surveykeys=get_survey_data('Discovery_9-26-16')

demogdata,surveydata=get_joint_dataset(get_demog_data(),surveydata_orig)
assert list(demogdata.index)==list(surveydata.index)
print('%d joint subjects found'%demogdata.shape[0])
surveyvars=list(surveydata.columns)
print('%d survey items found'%len(surveyvars))
print('Demographic variables to test:')
print(list(demogdata.columns))

# First get binary variables and test classification based on survey data.
# Only include variables that have at least 10% of the infrequent category.
# Some of these were not collected as binary variables, but we binarize by
# calling anything above the minimum value a positive outcome.

bvardata=numpy.array(demogdata)
sdata=numpy.array(surveydata).copy() #scale(numpy.array(surveydata))

results=pandas.DataFrame(columns=['variable','fa_ctr','trainf1','testf1'])

clf_params={}

ctr=0

classifier='svm'


def inner_cv_loop(Xtrain,Ytrain,clf,parameters,
                    oversample=None,fa_dims=20,
                    verbose=False):
    """
    use GridSearchCV to find best classifier for training set
    """

    rocscore={}
    best_est={}
    facanal={}
    for fa_d in [0,fa_dims]:
        clfname='fa' if fa_d>0 else "nofa"
        if fa_d>0:
            facanal[clfname]=FactorAnalysis(fa_d)
            Xtrain=facanal[clfname].fit_transform(Xtrain)
        else:
            facanal[clfname]=None

        if verbose:
            print(clfname)
        gs=GridSearchCV(clf,parameters,scoring='roc_auc')
        gs.fit(Xtrain,Ytrain)
        rocscore[clfname]=gs.best_score_
        best_est[clfname]=gs.best_estimator_

    bestscore=numpy.max([rocscore[i] for i in rocscore.keys()])
    bestclf=[i for i in rocscore.keys() if rocscore[i]==bestscore][0]
    if verbose:
        print('best:',bestclf,bestscore,best_est[bestclf],facanal[bestclf])
    return best_est[bestclf],bestscore,facanal[bestclf]

varname=binary_vars[0]
print(varname)

y=numpy.array(demogdata[varname].copy())
assert numpy.var(y)>0

if shuffle:
    numpy.random.shuffle(y)
    print('y shuffled')

# set up classifier params for GridSearchCV
if simple_params:
    print('WARNING: using simple parameters - change for production')
    parameters = {'kernel':('linear','rbf'),
       'C':[1., 100.],
       'gamma':1/numpy.array([100,500])}
else:
    parameters = {'kernel':('linear','rbf','poly'),
        'C':[0.5,1.,5, 10.,25.,50.,75.,100.],
        'degree':[2,3],'gamma':1/numpy.array([5,10,100,250,500,750,1000])}
clf=SVC(probability=True) #LogisticRegressionCV(solver='liblinear',penalty='l1')  #LinearSVC()

def main_cv_loop(Xdata,Ydata,clf,parameters,
                n_folds=4,oversample_thresh=0.1,verbose=False):

    # use stratified K-fold CV to get roughly equal folds
    #kf=StratifiedKFold(n_splits=nfolds)
    kf=StratifiedShuffleSplit(n_splits=4,test_size=0.2)
    # use oversampling if the difference in prevalence is greater than 20%
    if numpy.abs(numpy.mean(y)-0.5)>oversample_thresh:
        oversample='smote'
    else:
        oversample='none'

    # variables to store outputs
    pred=numpy.zeros(len(y))  # predicted values
    kernel=[]
    C=[]
    fa_ctr=0

    for train,test in kf.split(Xdata,Ydata):
        Xtrain=Xdata[train,:]
        Xtest=Xdata[test,:]
        Ytrain=Ydata[train]
        if numpy.abs(numpy.mean(Ytrain)-0.5)>0.2:
            if verbose:
                print('oversampling using SMOTETomek')
            sm = SMOTETomek()
            Xtrain, Ytrain = sm.fit_sample(Xtrain, Ytrain)

        best_estimator_,bestroc,fa=inner_cv_loop(Xtrain,Ytrain,clf,
                    parameters,verbose=True)
        if not fa is None:
            if verbose:
                print('transforming using fa')
                print(fa)
            tmp=fa.transform(Xtest)
            Xtest=tmp
            fa_ctr+=1
        pred.flat[test]=best_estimator_.predict_proba(Xtest)
        kernel.append(best_estimator_.kernel)
        C.append(best_estimator_.C)
    return roc_auc_score(y,pred,average='weighted'),y,pred

all_results=[]

for i in range(10):
    results,y_out,pred=main_cv_loop(sdata,y,clf,parameters,verbose=True)
    all_results.append(results)
    if shuffle:
        assert not all(numpy.array(demogdata[varname])==y_out)
    if numpy.var(pred)==0:
        print('%s: WARNING: no variance in predicted classes'%varname)
    #else:
    #    print(numpy.sum(pred==0),numpy.sum(pred==1))
with open('surveypred/surveypredict_cvresults_%s%s.csv'%(varname,shuffletag),'w') as f:
    for i in range(len(all_results)):
        f.write('%f\n'%all_results[i])

#     clf_params[binary_vars[i]]=(kernel,C)
#     ctr+=1
#     if verbose:
#         print('Training accuracy (f-score): %f'%numpy.mean(trainpredroc))
#         if numpy.var(pred)==0:
#             print('WARNING: no variance in classifier output, degenerate model fit')
#         print('Predictive accuracy')
#         print(classification_report(y,pred,labels=predlabels))
#         print_confusion_matrix(y,pred)
#         if False:
#             print("Features sorted by their absolute correlation with outcome (top %d):"%nfeatures)
#             featcorr=numpy.array([numpy.corrcoef(sdata[:,x],y)[0,1] for x in range(sdata.shape[1])])
#             idx=numpy.argsort(numpy.abs(featcorr))[::-1]
#             for i in range(nfeatures):
#                 print('%f: %s'%(featcorr[idx[i]],surveykeys[surveyvars[idx[i]]]))
#
# results.to_csv('surveypred/surveypredict_cvresults_%s%s.csv'%(varname,shuffletag))
#
# if not shuffle:
#     pickle.dump(clf_params,open('surveypred/clf_params_surveypredict_%s.pkl'%varname,'wb'))
