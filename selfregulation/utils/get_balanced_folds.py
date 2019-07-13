

from statsmodels.regression.linear_model import OLS
from sklearn.model_selection import KFold
import numpy as N


class BalancedKFold:
    """
    This function uses anova across CV folds to find
    a set of folds that are balanced in their distriutions
    of the X value - see Kohavi, 1995
    - we don't actually need X but we take it for consistency
    """
    def __init__(self,nfolds=5,pthresh=0.8,verbose=False):
        self.nfolds=nfolds
        self.pthresh=pthresh
        self.verbose=verbose

    def split(self,X,Y,max_splits=1000):
        """
        - we don't actually need X but we take it for consistency
        """

        nsubs=len(Y)

        # cycle through until we find a split that is good enough

        runctr=0
        best_pval=0.
        while 1:
            runctr+=1
            cv=KFold(n_splits=self.nfolds,shuffle=True)

            idx=N.zeros((nsubs,self.nfolds)) # this is the design matrix
            folds=[]
            ctr=0
            for train,test in cv.split(Y):
                idx[test,ctr]=1
                folds.append([train,test])
                ctr+=1

            lm_y=OLS(Y-N.mean(Y),idx).fit()

            if lm_y.f_pvalue>best_pval:
                best_pval=lm_y.f_pvalue
                best_folds=folds

            if lm_y.f_pvalue>self.pthresh:
                if self.verbose:
                    print(lm_y.summary())
                return iter(folds)

            if runctr>max_splits:
                print('no sufficient split found, returning best (p=%f)'%best_pval)
                return iter(best_folds)

if __name__=="__main__":
    Y=N.random.randn(100,1)
    bf=BalancedKFold(4,verbose=True)
    s=bf.split(Y,Y)
