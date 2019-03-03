# prediction_utils



import numpy, os, pandas
from selfregulation.utils.utils import get_info

#import pandas.rpy.common as com
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
R = robjects.r
base = importr('base')
stats = importr('stats')
mpath=importr('mpath')

# helper R functions for prediction
def get_demographic_model_type(demographics, verbose=False):
    base = get_info('base_directory')
    R.source(os.path.join(base, 'selfregulation', 'utils', 'utils.R'))
    
    get_vartypes = robjects.globalenv['get_vartypes']
    out=get_vartypes(demographics, verbose)
    model_types = pandas.DataFrame(numpy.reshape(numpy.matrix(out),(-1,2), 'F'))
    model_types.iloc[:, 0] = demographics.columns
    return model_types

# create a class that implements prediction using functions from R
class RModel:
    def __init__(self,modeltype,verbose=True,ncores=2,nlambda=100,
                lambda_preset=None):
        self.modeltype=modeltype
        assert self.modeltype in ['NB', 'ZINB', 'ZIpoisson', 'poisson']
        self.verbose=verbose
        self.model=None
        self.coef_=None
        self.ncores=ncores
        self.nlambda=nlambda
        self.lambda_preset=lambda_preset
        if self.lambda_preset is not None:
            print('using preset lambdas:',self.lambda_preset)
        self.lambda_optim=[]

    def fit(self,X,Y):
        self._fit_glmreg(X,Y)

    def _fit_glmreg(self,X,Y):
        if not isinstance(X, pandas.DataFrame):
            X=pandas.DataFrame(X,columns=['V%d'%i for i in range(X.shape[1])])
        X=X-X.mean(0)
        #X['intercept']=numpy.zeros(X.shape[0])
        if not isinstance(Y, pandas.DataFrame):
            Y=pandas.DataFrame(Y,columns=['X0'])

        if self.verbose:
            print('fitting using %s regression via mpath'%self.modeltype)
        data=X.copy()
        #data['y']=Y
        if self.modeltype=='poisson':
            robjects.globalenv['df']=pandas2ri.py2ri(data)
            robjects.r('df=data.matrix(df)')
            robjects.globalenv['y']=pandas2ri.py2ri(Y)
            robjects.r('y=data.matrix(y)')
            if self.lambda_preset is not None:
                robjects.r('fit=glmreg(df,y,family="poisson",lambda=%f)'%self.lambda_preset[0])
                fit=robjects.r('fit')
                self.model=fit
                robjects.r('coef_=coef(fit)')
                self.coef_=numpy.array(robjects.r('coef_'))[1:]

            else:
                self.model=mpath.cv_glmreg(base.as_symbol('df'),base.as_symbol('y'),
                                        family = 'poisson')
                fit=self.model[self.model.names.index('fit')]
                self.lambda_which=numpy.array(self.model[self.model.names.index('lambda.which')])[0]
                self.coef_=numpy.array(fit[fit.names.index('beta')])[:,self.lambda_which-1]
                self.lambda_optim=numpy.array(self.model[self.model.names.index('lambda.optim')])

        elif self.modeltype=='ZINB' or self.modeltype=='ZIpoisson' :
            #data['y']=Y.copy()
            robjects.globalenv['df']=pandas2ri.py2ri(data)
            robjects.globalenv['y']=pandas2ri.py2ri(Y)
            robjects.r('df$y=y$X0')
            if self.modeltype=='ZINB':
                family='negbin'
            else:
                family='poisson'
            # this is a kludge because I couldn't get it to work using the
            # standard interface to cv_zipath
            if self.lambda_preset is not None:
                robjects.r('fit=zipath(y~.|.,df,family="%s",penalty="enet",lambda.count=%f,lambda.zero=%f)'%(family,
                                self.lambda_preset[0],self.lambda_preset[1]))
                fit=robjects.r('fit')
                self.model=fit
                robjects.r('coef_=coef(fit,model="count")')
                self.coef_=numpy.array(robjects.r('coef_'))[1:]
            else:
                # use CV
                robjects.r('fit=cv.zipath(y~.|.,df,family="%s",penalty="enet",plot.it=FALSE,nlambda=%d,n.cores=%d)'%(family,self.nlambda,self.ncores))

                self.model=robjects.r('fit')
                fit=self.model[self.model.names.index('fit')]
            #self.lambdas=numpy.array(self.model[self.model.names.index('lambda')])
            #if self.verbose:
            #    print('model:',self.model)
                self.lambda_optim.append(numpy.array(self.model[self.model.names.index('lambda.optim')]))
            # just get the count coefficients
                robjects.r('coef_=coef(fit$fit,which=fit$lambda.which,model="count")')
            # drop the intercept term
            self.coef_=numpy.array(robjects.r('coef_'))[1:]

        #self.model=stats.lm('y~.', data = base.as_symbol('df')) #, family = "poisson")
    def predict(self,newX):
        if self.model is None:
            print('model must first be fitted')
            return None
        if not isinstance(newX, pandas.DataFrame):
            newX=pandas.DataFrame(newX,columns=['V%d'%i for i in range(newX.shape[1])])

        if self.modeltype=='poisson':
            robjects.globalenv['newX']=pandas2ri.py2ri(newX)
            robjects.r('newX=data.matrix(newX)')
            if self.lambda_preset is not None:
                # heuristic for whether we are using zipath()
                robjects.r('pred=predict(fit,newX)')
                pred=robjects.r('pred').squeeze()
            else:
                pred=mpath.predict_glmreg(self.model[self.model.names.index('fit')],
                                base.as_symbol('newX'),
                                which=self.lambda_which)
        elif self.modeltype=='ZINB' or self.modeltype=='ZIpoisson' :
            robjects.globalenv['newX']=pandas2ri.py2ri(newX)
            #robjects.r('newX=data.matrix(newX)')
            if self.lambda_preset is not None:
                # heuristic for whether we are using zipath()
                robjects.r('pred=predict(fit,newX)')
            else:
                robjects.r('pred=predict(fit$fit,newX,which=fit$lambda.which)')
            pred=robjects.r('pred').squeeze()


        return numpy.array(pred)

if __name__=='__main__':
    # run some tests
    # generate some data, here we just use gaussian
    X=pandas.DataFrame(numpy.random.randn(100,4),columns=['V%d'%i for i in range(4)])
    Y=X.dot([1,-1,1,-1])+numpy.random.randn(100)
    Yz=numpy.floor(Y-numpy.median(Y))
    Yz[Yz<0]=0
    Y=pandas.DataFrame(numpy.floor(Yz))

    for modeltype in [ 'poisson' ,'ZINB', 'ZIpoisson']:
        rm=RModel(modeltype)
        rm.fit(X,Y)
        if not rm.model is None:
            print(R.summary(rm.model))
            pred=rm.predict(X)
            print('corr(pred,actual):',numpy.corrcoef(numpy.array(pred).T,Y.T))
            print(rm.coef_)
