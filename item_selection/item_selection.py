"""
assess the ability to reconstruct data from a subset of variables
choose variables instead of tasks
"""


import os,glob,sys,itertools,time,pickle
import numpy,pandas
import binascii
from joblib import Parallel, delayed
import multiprocessing

from gasearch import GASearchParams,GASearch

# this is kludgey but it works
sys.path.append('../utils')
from utils import get_info,get_behav_data,get_demographics
#from r_to_py_utils import missForest

__TEST_DECIMATE__=False
# set up variables

if len(sys.argv)>1:
    reconweight=float(sys.argv[1])
    objective_weights=[reconweight,1-reconweight]
else:
    objective_weights=[0.5,0.5]
print('using objective weights:',objective_weights)
gasearch=GASearch(objective_weights=objective_weights,targets=['task'],
                usepca=True,
                lasso_alpha=0.05)
gasearch.get_taskdata()
if __TEST_DECIMATE__:
    tasks_to_keep=gasearch.decimate_task_data()

gasearch.load_targetdata()
gasearch.get_tasktimes()

#gasearch.impute_targetdata()
print('using targets:',gasearch.params.targets)
print('%d variables in taskdata'%gasearch.taskdata.shape[1])
print('%d variables in targetdata'%gasearch.targetdata.shape[1])

# get initial population
gasearch.params.start_time=time.time()
gasearch.get_initial_population_tasks()

# perform selection for maximum of ngen generations

for generation in range(gasearch.params.ngen):
    roundstart=time.time()

    maxcc=gasearch.select_parents_tasks()

    gasearch.ccmax[generation]=[numpy.max(gasearch.cc)]+maxcc

    # store the best scoring set of tasks and count to see if we have
    # exceeded convergence criterion
    assert len(gasearch.population)==len(gasearch.cc_sorted)
    bestp=gasearch.population[numpy.where(gasearch.cc_sorted==numpy.max(gasearch.cc_sorted))[0][0]]
    bestp.sort()
    gasearch.bestp_saved[generation]=bestp
    if generation>1 and gasearch.bestp_saved[generation]==gasearch.bestp_saved[generation-1]:
        bestctr+=1
    else:
        bestctr=0
    print('best set:',bestp,bestctr)
    if bestctr>gasearch.params.convergence_threshold:
        break

    # crossover and immigrate
    gasearch.crossover_tasks()
    gasearch.immigrate_tasks()

    print('gen ',generation,'(Z,recon,subcorr):',gasearch.ccmax[generation])
    print('Time elapsed (secs):', time.time()-roundstart)

# print best outcome
bestp.sort()
print('best set: task (time)')
totaltasktime=0
for i in bestp:
    print(i,gasearch.tasks[i],'(%f)'%gasearch.params.tasktime[i])
    totaltasktime+=gasearch.params.tasktime[i]
print('total task time:',totaltasktime)
if __TEST_DECIMATE__:
    try:
        assert bestp==tasks_to_keep
        print('DECIMATION TEST PASSED!')
    except AssertionError:
        print('DECIMATION TEST FAILED!!!!!!!')
        print(bestp,'estimated')
        print(tasks_to_keep,'true')

print('Time elapsed (secs):', time.time()-gasearch.params.start_time)

gasearch.params.hash=binascii.hexlify(os.urandom(4)).decode('utf-8')

# add a random hash to the file name so that we can run it multiple times
# and it will save to different files
if not __TEST_DECIMATE__:
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    pickle.dump(gasearch,
      open("outputs/ga_results_tasks_%s_%s_%s_%s.pkl"%(gasearch.params.clf,'-'.join(gasearch.params.targets),
        '-'.join(['%s'%i for i in gasearch.params.objective_weights]),gasearch.params.hash),'wb'))
