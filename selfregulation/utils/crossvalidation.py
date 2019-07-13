"""
crossvalidation functions
"""
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

def main_cv_loop(Xdata,Ydata,clf,parameters,
                n_folds=4,oversample_thresh=0.1,verbose=False):

    # use stratified K-fold CV to get roughly equal folds
    #kf=StratifiedKFold(n_splits=nfolds)
    kf=StratifiedShuffleSplit(n_splits=4,test_size=0.2)
    # use oversampling if the difference in prevalence is greater than 20%
    if numpy.abs(numpy.mean(Ydata)-0.5)>oversample_thresh:
        oversample='smote'
    else:
        oversample='none'

    # variables to store outputs
    pred=numpy.zeros(len(Ydata))  # predicted values
    pred_proba=numpy.zeros(len(Ydata))  # predicted values
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
        pred_proba.flat[test]=best_estimator_.predict_proba(Xtest)
        pred.flat[test]=best_estimator_.predict(Xtest)
        kernel.append(best_estimator_.kernel)
        C.append(best_estimator_.C)
    return roc_auc_score(Ydata,pred,average='weighted'),Ydata,pred,pred_proba
