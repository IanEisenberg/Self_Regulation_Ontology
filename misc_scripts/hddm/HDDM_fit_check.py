from expanalysis.experiments.ddm_utils import get_HDDM_fun
from glob import glob
import hddm
from kabuki.analyze import gelman_rubin
import matplotlib.pyplot as plt
from multiprocessing import Pool 
import numpy as np
from os import path, remove, rename
import pandas as pd
import pickle
from selfregulation.utils.utils import get_behav_data


samples=3000
hddm_fun_dict = get_HDDM_fun(None, samples)
hddm_fun_dict.pop('twobytwo')
gelman_vals = {}

def run_model(task, data):
    # compute DVs (create models)
    group_dvs = hddm_fun_dict[task](data)
    # load models
    base_files = glob('%s*_base.model' % task)
    #m_base = hddm.load(base_files[0])
    m_base = pickle.load(open(base_files[0], 'rb'))
    #Try: m_base = pickle.load(open(base_files[0], 'rb')) if you get a unicode error
    m_condition = None
    condition_files = glob('%s*_condition.model' % task)
    if len(condition_files)>0:
        #m_condition = hddm.load(condition_files[0])
        m_condition = pickle.load(open(condition_files[0], 'rb'))
    return (m_base, m_condition)

def assess_convergence(task, reps=5):
    # load data
    data = get_behav_data(file='Individual_Measures/%s.csv.gz' % task)
    data = data.query('worker_id in %s' % list(data.worker_id.unique()[0:20]))
    outputs = []
    for _ in range(reps):
        output = run_model(task, data)
        outputs.append(output)
    return {task: outputs}
        
# **************************************************************
# Assess Convergence using Gelman Rubin criteria
# Runs the model multiple times on the same data
# ************************************************************
# create group maps
pool = Pool()
#mp_results = pool.map(assess_convergence, hddm_fun_dict.keys())
mp_results = pool.map(assess_convergence, ['stroop'])
pool.close() 
pool.join()

results = {}
for d in mp_results:
    results.update(d)

# save plots of traces
# make sure the output directory exists
for k,v in results.items():
    gelman_vals[k+'_base'] = gelman_rubin([i[0] for i in v])
    # plot posteriors
    v[0][0].plot_posteriors(['a', 't', 'v'], save=True)
    plots = glob('*png')
    for p in plots:
        rename(p, path.join('hddm_output', 'Plots', '%s_base_%s' % (k,p)))
    
    if v[0][1] is not None:
        gelman_vals[k+'_condition'] = gelman_rubin([i[1] for i in v])
        
        v[0][1].plot_posteriors(['a', 't', 'v'], save=True)
        plots = glob('*png')
        for p in plots:
            rename(p, path.join('hddm_output', 'Plots', '%s_condition_%s' % (k,p)))

# save gelman vals
pickle.dump(gelman_vals, open(path.join('hddm_output', 'gelman_vals.pkl'), 'wb'))



# *******************************************
# Overview of different ways we can assess HDDM convergence
# Useful functions:
    # PYMC: https://healthyalgorithms.com/2010/10/19/mcmc-in-python-how-to-stick-a-statistical-model-on-a-system-dynamics-model-in-pymc/
    # m.gen_stats: get stats matrix
    # m.print_stats: print stats matrix
    # m.dic_info: fit indices
# *******************************************
task = 'stroop'
N = 20 # number of workers
full_data = get_behav_data(file='Individual_Measures/%s.csv.gz' % task)

# let's only look at a few workers
data = full_data.query('worker_id in %s' % list(full_data.worker_id.unique()[0:N]))
outputs = run_model(task, data)
m=outputs[0]
# after the model is made is generates a .db file, a data file and a 'model' file
# the data file is a transformation of the data we put in - with different scales for
# rt, different columns, and some rows remove
data_in = pd.read_csv('%s_data.csv' % task)

# Example of extracting stats
# get stats for individual drift rates
stats= m.gen_stats().filter(regex='a_subj.', axis=0)
# the mc error reflects the variance around the estimate
stats['mc err']

# first thing to do is look at the traces - make sure the model converged
m.plot_posteriors(['a', 't', 'v']) 
# these traces are stored in the mc object...
mc = m.mc
# in the mc object are nodes corresponding to each variables
mc_nodes = mc.nodes
# you can get the trace if you know the name of the variable
trace = list(mc.trace('a'))

# ok, back to hddm stuff

# generate post predictive. This will run the model {samples} times
# using the parameters for each subject
ppc = hddm.utils.post_pred_gen(m, samples=10)
# we can see the responses for each subject/sample in the returned dataframe
print(ppc.head())

# we can compare the simulated model to data using post_pred_stats
# This gives the parameter estimates for many variables (rows) and prints the observed value
ppc_compare = hddm.utils.post_pred_stats(data,ppc)
print(ppc_compare)

# by setting "call_compare" to False we just get the summary stats for each subject
ppc_summary = hddm.utils.post_pred_stats(data, ppc, call_compare=False)
print(ppc_summary)

# we can now compare that summary to the real data
# accuracy
sim_acc = ppc_summary.groupby(level=0).accuracy.mean()
sim_acc.index = [int(i.split('.')[1]) for i in sim_acc.index]
sim_acc.sort_index(inplace=True)
data_acc = data_in.groupby('subj_idx').response.mean()
corr_acc = np.corrcoef(data_acc, sim_acc)[0,1].round(4)
# upper and lower bound rt
sim_rt = ppc_summary.groupby(level=0).mean()[['mean_ub', 'mean_lb']]
sim_rt.index = [int(i.split('.')[1]) for i in sim_rt.index]
sim_rt.sort_index(inplace=True)
data_rt = data_in.query('response==1').groupby('subj_idx').rt.median()
corr_rt = np.corrcoef(data_rt, sim_rt.mean_ub)[0,1].round(4)
# Plotting!
plt.scatter(data_acc, sim_acc, label='Accuracy'); 
plt.text(.9,.9, 'Acc r = %s' % corr_acc)

plt.scatter(data_rt, sim_rt.mean_ub, label='Upper Bound RT'); 
plt.xlabel('Data'); plt.ylabel('Model')
plt.text(.6,.8, 'RT r = %s' % corr_rt)
plt.legend()
# we can also plot the poster predictive
m.plot_posterior_predictive(figsize=(12,80), num_subjs=20)
plt.suptitle('Posterior Predictives')

# clean up
for f in glob('*.model'):
    rename(f, path.join('hddm_output', f))
    
for f in glob('*_traces.db'):
    rename(f, path.join('hddm_output', f))
    
for f in glob('*.csv'):
    remove(f)
    
# ************************************************************************
# Example of getting log likelihood (recreating plot_posterior_predictive)
# *************************************************************************

from kabuki.analyze import _parents_to_random_posterior_sample
from kabuki.analyze import _plot_posterior_pdf_node
from scipy.stats import entropy

observeds = m.get_observeds()
plot_func = _plot_posterior_pdf_node
required_method = 'pdf'
figsize = (12,12)
num_subjs=4
columns=2
value_range = np.linspace(-5, 5, 100)
kwargs = {'value_range': value_range}

for tag, nodes in observeds.groupby('tag'):
    fig = plt.figure(figsize=figsize)
    fig.suptitle(tag, fontsize=12)
    fig.subplots_adjust(top=0.9, hspace=.4, wspace=.3)

    # Plot individual subjects (if present)
    i = 0
    for subj_i, (node_name, bottom_node) in enumerate(nodes.iterrows()):
        i += 1
        if not hasattr(bottom_node['node'], required_method):
            continue # skip nodes that do not define the required_method

        nrows = num_subjs or len(nodes)/columns     
        ax = fig.add_subplot(np.ceil(nrows), columns, subj_i+1)
        if 'subj_idx' in bottom_node:
            ax.set_title(str(bottom_node['subj_idx']))

        plot_func(bottom_node['node'], ax, **kwargs)

# how do you get the likelihood? 
samples=10
node = bottom_node['node']
like = np.empty((samples, len(value_range)), dtype=np.float32)
for sample in range(samples):
    _parents_to_random_posterior_sample(node)
    # Generate likelihood for parents parameters
    like[sample,:] = node.pdf(value_range)

y = like.mean(axis=0)
plt.plot(value_range,y)
plt.hist(node.value.values, normed=True, color='r')


def get_likelihood(model, samples=10):
    value_range = np.linspace(-5,5,100)
    observeds = m.get_observeds()
    like = np.empty((samples, len(value_range)), dtype=np.float32)
    KLs = {}
    for subj_i, (node_name, bottom_node) in enumerate(observeds.iterrows()):
        node = bottom_node['node']
        for sample in range(samples):
            _parents_to_random_posterior_sample(node)
            # Generate likelihood for parents parameters
            like[sample,:] = node.pdf(value_range)
        y = like.mean(axis=0)
        data_bins = np.histogram(node.value, value_range, density=True)[0]
        plt.plot(value_range[:-1], data_bins)
        plt.plot(value_range[:-1], y[:-1])
        KL_divergence = entropy(y[1:]+1E-10, data_bins+1E-10)
        KLs[subj_i] = KL_divergence
    return KLs

KLs = get_likelihood(m)