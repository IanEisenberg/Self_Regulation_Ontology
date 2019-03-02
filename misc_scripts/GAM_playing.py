#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:00:23 2018

@author: ian
"""
import matplotlib.pyplot as plt
import numpy as np
from os import path
import pickle
from pygam import s, f, l, LinearGAM
import seaborn as sns
from sklearn.metrics import mean_absolute_error

from selfregulation.utils.get_balanced_folds import BalancedKFold
from selfregulation.utils.plot_utils import format_num, save_figure
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_recent_dataset

# reference: https://pygam.readthedocs.io/en/latest/notebooks/tour_of_pygam.html#Terms-and-Interactions
# ********************************************************
# helper functions
# ********************************************************

def get_avg_score(scores, score_type='R2'):
    return np.mean([i[score_type] for i in scores])

def get_importances(X, y, Xtest, ytest):
    importances = {}
    for predictor, vals in X.iteritems():
        gam = LinearGAM(s(0), fit_intercept=False)
        gam.fit(vals, y)
        gam.gridsearch(vals, y)
        pred = gam.predict(Xtest[predictor])
        # define importances as the R2 for that factor alone
        R2 = np.corrcoef(ytest,pred)[0,1]**2
        importances[predictor] = R2
    return importances
    
def run_GAM(X, Y, get_importance=False, n_splines=20, folds=10):
    # set up GAM
    formula = s(0, n_splines)
    for i in range(1, X.shape[1]):
        formula = formula + s(i, n_splines)
    gam = LinearGAM(formula)
    gam.fit(X, X.iloc[:,0])
    
    # run full model
    GAM_results = {}
    for name, y in Y.iteritems():
        print("\nFitting for %s\n" % name)
        CV = BalancedKFold(folds)
        importances = {k:[] for k in X.columns}
        pred=np.zeros(y.shape[0])
        for train,test in CV.split(X,y):
            Xtrain = X.iloc[train,:]
            ytrain = y.iloc[train]
            Xtest = X.iloc[test,:]
            ytest = y.iloc[test]
            gam = LinearGAM(formula)
            gam.gridsearch(Xtrain, ytrain)

            # out of fold
            p = gam.predict(Xtest)
            if len(p.shape)>1:
                p=p[:,0]
            pred[test]=p

            if get_importance:    
                # get importances, defined as the predictive ability of each variable on its own
                importance_out = get_importances(Xtrain, ytrain, Xtest, ytest)
                for k,v in importance_out.items():
                    importances[k].append(v)
                    
        cv_scores = [{'r': np.corrcoef(y,pred)[0,1],
                      'R2': np.corrcoef(y,pred)[0,1]**2,
                      'MAE': mean_absolute_error(y,pred)}]
        
        
        # insample
        gam.gridsearch(X, y)
        in_pred = gam.predict(X)
        in_scores = [{'r': np.corrcoef(y,in_pred)[0,1],
                          'R2': np.corrcoef(y,in_pred)[0,1]**2,
                          'MAE': mean_absolute_error(y,in_pred)}]
        GAM_results[name] = {'scores_cv': cv_scores,
                             'scores_insample': in_scores,
                             'pred_vars': X.columns,
                             'importances': importances,
                             'model': gam}
    return GAM_results

def plot_term(gam, i, ax=None, color='k', size=10):
    if ax is None:
        f,ax = plt.subplots(1,1, figsize=(size,size*.7))
    XX = gam.generate_X_grid(i)
    pdep, confi = gam.partial_dependence(i, X=XX, width=.95)
    ax.plot(XX[:, i], pdep, c=color, lw=size/2)
    ax.plot(XX[:, i], confi[:, 0], c='grey', ls='--', lw=size/3)
    ax.plot(XX[:, i], confi[:, 1], c='grey', ls='--', lw=size/3)
    ax.set_xlabel('x', fontsize=size*2)
    ax.set_ylabel(gam.terms[i], fontsize=size*2)
            
def plot_GAM(gams, X, Y, size=4, dpi=300, ext='png', filename=None):
    cols = X.shape[1]
    rows = Y.shape[1]
    colors = sns.color_palette(n_colors=rows)
    plt.rcParams['figure.figsize'] = (cols*size, rows*size)
    fig, mat_axs = plt.subplots(rows, cols)
    titles = X.columns
    for j, (name, out) in enumerate(gams.items()):
        axs = mat_axs[j]
        gam = out['model']
        R2 = get_avg_score(out['scores_cv'])
        p_vals = gam.statistics_['p_values']
        for i, ax in enumerate(axs):
            plot_term(gam, i, ax, colors[j], size=size)
            ax.set_xlabel('')
            ax.text(.5, .95, 'p< %s' % format_num(p_vals[i]), va='center', 
                    fontsize=size*3, transform=ax.transAxes)
            if j%2==0:
                ax.set_title(titles[i], fontsize=size*4)
            if i==0:
                ax.set_ylabel(name + ' (%s)' % format_num(R2), 
                              fontsize=size*4)
            else:
                ax.set_ylabel('')
                
    plt.subplots_adjust(hspace=.4)
    if filename is not None:
        save_figure(fig, '%s.%s' % (filename,ext),
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()

def check_gam(gam, X, y, size=5):
    f, axes = plt.subplots(2,2, figsize=(size*2, size*2))
    predictions = gam.predict(X)
    residuals = gam.deviance_residuals(X, y) # same as y-predictions
    
    axes[0][1].plot(residuals, predictions, 'o')
    axes[0][1].set_title('Resids vs. linear pred.', fontweight='bold')
    
    axes[1][0].hist(residuals, linewidth=size/5, edgecolor='w', bins=20)
    axes[1][0].set_title('Histogram of Residuals', fontweight='bold')
    
    axes[1][1].plot(y, predictions, 'o')
    axes[1][1].set_title('Response vs. Fitted Values', fontweight='bold')
    
    

# ********************************************************
# Load Data
# ********************************************************
results = load_results(get_recent_dataset())
Y = results['task'].DA.get_scores()

# ********************************************************
# Fitting
# ********************************************************
output_dir = path.dirname(results['task'].get_output_dir())
output_file = path.join(output_dir, 'GAM_out.pkl')
if path.exists(output_file):
    GAM_results = pickle.load(open(output_file, 'rb'))
else:
    GAM_results = {}
    GAM_results['task'] = run_GAM(results['task'].EFA.get_scores(), Y)
    GAM_results['survey'] = run_GAM(results['survey'].EFA.get_scores(), Y)
    pickle.dump(GAM_results, open(output_file, 'wb'))

# ********************************************************
# Inspect
# ********************************************************
gams = GAM_results['task']
X = results['task'].EFA.get_scores()
ridge_prediction = results['task'].load_prediction_object(classifier='ridge')['data']

for k,v in gams.items():
    ridge_r2cv = ridge_prediction[k]['scores_cv'][0]['R2']
    ridge_r2in = ridge_prediction[k]['scores_insample'][0]['R2']
    print('*'*79)
    print(k)
    print('GAM CV', get_avg_score(v['scores_cv']))
    print('GAM Insample', get_avg_score(v['scores_insample']))
    print('*')
    print('Ridge CV', format_num(ridge_r2cv, 3))
    print('Ridge insample', format_num(ridge_r2in, 3))
    print('*'*79)
    

# plot full matrix
plot_dir = path.dirname(results['task'].get_plot_dir())
plot_GAM(GAM_results['task'], 
         results['task'].EFA.get_scores(), 
         Y, 
         filename=path.join(plot_dir, 'task_GAM'))

plot_GAM(GAM_results['survey'], 
         results['survey'].EFA.get_scores(), 
         Y, 
         filename=path.join(plot_dir, 'survey_GAM'))