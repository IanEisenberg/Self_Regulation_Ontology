"""
assess the ability to reconstruct data from a subset of variables
choose variables instead of tasks
"""


import os,glob,sys,itertools,time,pickle
import numpy,pandas
import json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import FactorAnalysis

from genetic_algorithm import get_initial_population_vars,get_population_fitness_vars
from genetic_algorithm import select_parents_vars,crossover_vars,immigrate_vars
from search_objectives import get_subset_corr_vars,get_reconstruction_error_vars

# this is kludgey but it works
sys.path.append('../utils')
from utils import get_info,get_behav_data
from r_to_py_utils import missForest


if len(sys.argv)>1:
   clf=sys.argv[1]
   print('using ',clf)
else:
    clf='linear'

if len(sys.argv)>2:
   nsplits=int(sys.argv[2])
   print('nsplits=',nsplits)
else:
    nsplits=8

target='all' # or 'all' or 'task'
objective_weights=[0.5,0.5] # weights for reconstruction and correlation
num_cores=2
#num_cores = multiprocessing.cpu_count()
print('using %d cores'%num_cores)
assert target in ['all','task']

dataset=get_info('dataset')
print('using dataset:',dataset)
basedir=get_info('base_directory')
derived_dir=os.path.join(basedir,'Data/Derived_Data/%s'%dataset)


data=get_behav_data('Discovery_11-07-2016', file = 'taskdata_imputed.csv')
for c in data.columns:
    if c.find('survey')>-1:
        data.drop(c)
taskvars=list(data.columns)

if target=='all':
    print('targeting all variables')
    alldata=get_behav_data(dataset)
    surveydata=pandas.DataFrame()
    for k in alldata.columns:
        if k.find('survey')>-1:
            surveydata[k]=alldata[k]

    assert all(data.index == surveydata.index)

    alldata = surveydata.merge(data,'inner',right_index=True,left_index=True)

    taskvaridx=[i for i in range(len(alldata.columns)) if alldata.columns[i] in taskvars]

    data=missForest(alldata)[0]

else:
    print('targeting task variables')
    taskvaridx=[i for i in range(data.shape[1])]


# set up genetic algorithm
start_time = time.time()
nvars=10
ngen=2500
initpopsize=500
nselect=50
nimmigrants=500
nbabies=4
mutation_rate=1/nvars
ccmax={}
bestp_saved={}
bestctr=0
clf='kridge'
population=get_initial_population_vars(initpopsize,nvars,data,taskvaridx)

for generation in range(ngen):
    population,cc,maxcc=select_parents_vars(population,data,nselect,clf,
                                            obj_weight=objective_weights)
    ccmax[generation]=[numpy.max(cc)]+maxcc
    bestp=population[numpy.where(cc==numpy.max(cc))[0][0]]
    bestp.sort()
    bestp_saved[generation]=bestp
    if generation>1 and bestp_saved[generation]==bestp_saved[generation-1]:
        bestctr+=1
    else:
        bestctr=0
    #for i in bestp:
    #    print(i,data.columns[i])
    print(bestp,bestctr)
    if bestctr>10:
        break
    population=crossover_vars(population,data,taskvaridx,nbabies=nbabies)
    population=immigrate_vars(population,nimmigrants,nvars,data,taskvaridx)
    print(generation,ccmax[generation])


bestp.sort()
print('best set')
for i in bestp:
    print(i,data.columns[i])

print('')
print('best items across population')
popfreq=numpy.zeros(len(data.columns))
for i in range(len(data.columns)):
    popfreq[i]=numpy.sum(numpy.array(population)==i)
idx=numpy.argsort(popfreq)[::-1]
for i,index in enumerate(idx):
    if popfreq[index]>0:
        print(data.columns[index],popfreq[index])

print('Time elapsed (secs):', time.time()-start_time)



pickle.dump((bestp_saved,ccmax,population),open("ga_results_%s_%s.pkl"%(clf,target),'wb'))

#numpy.save('reconstruction_%s_cc.npy'%clf,cc)
#numpy.save('reconstruction_%s_sse.npy'%clf,sse)

# first run
# 0 adaptive_n_back
# 1 angling_risk_task_always_sunny
# 2 attention_network_task
# 4 choice_reaction_time
# 5 cognitive_reflection_survey
# 18 kirby
# 29 spatial_span
# 35 two_stage_decision
