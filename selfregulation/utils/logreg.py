"""
sklearn-like wrapper for statsmodels logistic regression
"""

from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
class LogReg:
    def __init__(self):
        self.coef_=None

    def fit(self,X,y):
        X=add_constant(X)
        self.lr=Logit(y,X)
        self.l_fitted=self.lr.fit()
        self.coef_=self.l_fitted.params[:-1]

    def predict(self,X):
        if self.coef_ is None:
            print('you must first fit the model')
            return
        X=add_constant(X)
        return(self.lr.predict(self.l_fitted.params,X))
