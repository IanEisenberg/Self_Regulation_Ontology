
from expanalysis.experiments.psychological_models import MoE_Model
import numpy as np
import pandas as pd
from selfregulation.utils.utils import get_behav_data

data = get_behav_data(file='Individual_Measures/hierarchical_rule.csv.gz')
workers = data.worker_id.unique()
data = data.query("worker_id == '%s'" % workers[0])

        

from scipy.optimize import minimize
def eval_MoE(fit_args, passed_args):
    args = {
        'kappa': fit_args[0],
        'zeta': fit_args[1],
        'xi': fit_args[2],
        'beta2' : fit_args[3],
        'beta3': fit_args[4],
        'alphaC': fit_args[5],
        'alphaO': fit_args[6],
        'alphaS': fit_args[7],
        'beta_hierarchy': fit_args[8],
        'data': data,
        }
    
    MoE_model = MoE_Model(**args)
    likelihoods = []
    for i, trial in data.iterrows():
        if trial.key_press != -1:
            action_probs = MoE_model.get_action_probs(trial)
            likelihood = action_probs[trial.key_press]
            likelihoods.append(likelihood)
            # update model
            MoE_model.update_confidence(trial)
            MoE_model.update_experts(trial)
    print(-np.sum(np.log(likelihoods)))
    return -np.sum(np.log(likelihoods))
        
def fit_MoE(args):
    assert set(['kappa', 'zeta', 'xi']) <= set(args.keys())
    fit_args = [args['kappa'], args['zeta'], 
                args['xi'], args['beta2'], args['beta3'], args['alphaC'], 
                args['alphaO'], args['alphaS'], args['beta_hierarchy']]
    
    
    passed_args = {'data': args['data']}
    out = minimize(eval_MoE, fit_args, passed_args, method='Nelder-Mead')
    return out



kappa = 1
zeta = .1
xi = .2
alphaC = 1
alphaS = 1
alphaO = 1
beta2 = 100
beta3 = 100
beta_hierarchy=20
                 
args = {
        'kappa': kappa,
        'zeta': zeta,
        'xi': xi,
        'beta2' : beta2,
        'beta3': beta3,
        'alphaC': alphaC,
        'alphaO': alphaO,
        'alphaS': alphaS,
        'beta_hierarchy': beta_hierarchy,
        'data': data,
        }

#out = fit_MoE(args)


model = MoE_Model(**args.copy())

model_data = data.copy()
new_data = []
trial_by_trial_confidence = []
verbose=False
for i, trial in data.iloc[0:100].iterrows():
    if trial.key_press != -1:
        action_probs = model.get_action_probs(trial)
        #action = np.random.choice(a=list(action_probs.keys()), 
        #                          p = list(action_probs.values()))
        maxx = max(action_probs.values())
        action= [k for k,v in action_probs.items() if v==maxx][0]
        correct = trial.correct_response == action
        # update trial
        trial.correct = correct
        trial.key_press = action
        new_data.append(trial)
        if verbose:
            print('*'*79)
            print('Trial:', i)
            print('OSC:', list(trial[['orientation', 'stim', 'border']]))
            print([e.get_action_probs(trial) for e in model.experts])
            print('correct choice', trial.correct_response)
            print('top action', action)
        # update model
        model.update_confidence(trial)
        model.update_experts(trial)
        trial_by_trial_confidence.append(model.get_all_confidences(trial))
        if verbose:
            print(trial_by_trial_confidence[-1]['hierarchy'])
            input()
        
new_data = pd.DataFrame(new_data)
trial_by_trial_confidence = pd.DataFrame(trial_by_trial_confidence)
# plot model confidences over learning
trial_by_trial_confidence.plot(figsize=(12,8))    

# test divergence between hierarchical and flat experts after training
correspond = []
h = model.experts[0]
f = model.experts[1]
for i, trial in data.sample(100).iterrows():
    h_actions = h.get_action_probs(trial)
    f_actions = f.get_action_probs(trial)
    maxh = max(h_actions.values())
    maxf = max(f_actions.values())
    h_choice = [k for k,v in h_actions.items() if v==maxh]
    f_choice = [k for k,v in f_actions.items() if v==maxf]
    correspond.append(h_choice==f_choice)






