"""
functions for genetic algorithm search
"""
import os,sys
import numpy,pandas
from sklearn.preprocessing import scale
import fancyimpute

sys.path.append('../utils')
from utils import get_info,get_behav_data,get_demographics

from search_objectives import get_reconstruction_error_vars,get_subset_corr_vars
from search_objectives import get_reconstruction_error,get_subset_corr

__USE_MULTIPROC__=True

if __USE_MULTIPROC__:
    from joblib import Parallel, delayed
    import multiprocessing
    if 'NUMCORES' in os.environ:
        num_cores=int(os.environ['NUMCORES'])
    else:
        num_cores = multiprocessing.cpu_count()
    if num_cores==0:
        __USE_MULTIPROC__=False
    else:
        print('multiproc: using %d cores'%num_cores)



def get_taskdata(dataset,file= 'taskdata_imputed.csv',
                drop_tasks=['cognitive_reflection_survey','writing_task']):
    taskdata=get_behav_data(dataset, file )
    for c in taskdata.columns:
        taskname=c.split('.')[0]
        if taskname in drop_tasks:
            print('dropping',c)
            del taskdata[c]
    print('taskdata: %d variables'%taskdata.shape[1])
    taskvars=list(taskdata.columns)
    tasknames=[i.split('.')[0] for i in taskdata.columns]
    tasks=list(set(tasknames))
    tasks.sort()
    return taskdata,taskvars,tasks


def load_targetdata(dataset,targets,taskdata):
    targetdata=None
    if 'task' in targets:
        targetdata=get_behav_data(dataset, file = 'taskdata_imputed.csv')
        assert all(taskdata.index == targetdata.index)
        print('target: task, %d variables'%taskdata.shape[1])
        print('%d missing values'%numpy.sum(numpy.isnan(taskdata.values)))

    if 'survey' in targets:
        alldata=get_behav_data(dataset)
        surveydata=pandas.DataFrame()
        for k in alldata.columns:
            if k.find('survey')>-1:
                surveydata[k]=alldata[k]
        print('target: survey, %d variables'%surveydata.shape[1])
        print('%d missing values'%numpy.sum(numpy.isnan(surveydata.values)))
        if not targetdata is None:
            assert all(taskdata.index == surveydata.index)
            targetdata = surveydata.merge(targetdata,'inner',right_index=True,left_index=True)
        else:
            targetdata=surveydata

    if 'demog' in targets:
        demogvars=['BMI','Age','Sex','RetirementAccount','ChildrenNumber',
                    'CreditCardDebt','TrafficTicketsLastYearCount',
                    'TrafficAccidentsLifeCount','ArrestedChargedLifeCount',
                    'LifetimeSmoke100Cigs','AlcoholHowManyDrinksDay',
                    'CannabisPast6Months','Nervous',
                    'Hopeless', 'RestlessFidgety', 'Depressed',
                    'EverythingIsEffort','Worthless']
        demogdata=get_demographics(dataset,var_subset=demogvars)
        print('target: demog, %d variables'%demogdata.shape[1])
        print('%d missing values'%numpy.sum(numpy.isnan(demogdata.values)))
        if not targetdata is None:
            assert all(taskdata.index == demogdata.index)
            targetdata = demogdata.merge(targetdata,'inner',right_index=True,left_index=True)
        else:
            targetdata=demogdata
    return targetdata

def impute_targetdata(targetdata):
    """
    there are very few missing values, so just use a fast but dumb imputation here
    """
    if numpy.sum(numpy.isnan(targetdata.values))>0:
        targetdata_imp=fancyimpute.SimpleFill().complete(targetdata.values)
        targetdata=pandas.DataFrame(targetdata_imp,index=targetdata.index,columns=targetdata.columns)
    return targetdata


def get_initial_population_tasks(params):
    poplist=[]
    idx=[i for i in range(params.ntasks)]
    for i in range(params.initpopsize):
        numpy.random.shuffle(idx)
        poplist.append(idx[:params.nvars])
    return poplist

def get_population_fitness_tasks(pop,taskdata,targetdata,params): #nsplits,clf,obj_weight):
    # first get cc for each item in population
    cc_recon=numpy.zeros(len(pop))
    predacc_insample=numpy.zeros(len(pop))
    if params.objective_weights[0]>0:
        if __USE_MULTIPROC__:
            cc_recon=Parallel(n_jobs=num_cores)(delayed(get_reconstruction_error)(ct,taskdata,targetdata,params) for ct in pop)
        else:
            cc_recon=[get_reconstruction_error(ct,taskdata,targetdata,params) for ct in pop]
    else:
        cc_recon=[0]
    if params.objective_weights[1]>0:
        if __USE_MULTIPROC__:
            cc_subsim=Parallel(n_jobs=num_cores)(delayed(get_subset_corr)(ct,taskdata,targetdata) for ct in pop)
        else:
            cc_subsim=[get_subset_corr(ct,taskdata,targetdata) for ct in pop]
    else:
        cc_subsim=[0]
    maxcc=[numpy.max(cc_recon),numpy.max(cc_subsim)]
    cc_recon=scale(cc_recon)
    cc_subsim=scale(cc_subsim)
    try:
        print('corr recon-subsim:',numpy.corrcoef(cc_recon,cc_subsim)[0,1])
    except:
        pass
    cc=cc_recon*params.objective_weights[0] + cc_subsim*params.objective_weights[1]
    return cc,maxcc


def select_parents_tasks(pop,taskdata,targetdata,params):
    cc,maxcc=get_population_fitness_tasks(pop,taskdata,targetdata,params)
    idx=numpy.argsort(cc)[::-1]
    pop_sorted=[pop[i] for i in idx[:params.nselect]]
    cc_sorted=[cc[i] for i in idx[:params.nselect]]
    return(pop_sorted,cc_sorted,maxcc)


def crossover_tasks(pop,params):
    if params.mutation_rate is None:
        params.mutation_rate=1/len(pop[0])
    # assortative mating - best parents mate
    families=numpy.kron(numpy.arange(numpy.floor(len(pop)/2)),[1,1])
    numpy.random.shuffle(families)
    for f in numpy.unique(families):
        famidx=[i for i in range(len(families)) if families[i]==f]
        if len(famidx)!=2:
            continue
        try:
            subpop=[pop[i] for i in famidx]
        except:
            print('oops...')
            print(len(pop))
            print(famidx)
            raise Exception('breaking')
        parents=list(numpy.unique(numpy.hstack((subpop[0],subpop[1]))))
        if len(set(parents))<len(subpop[1]):
            continue
        for b in range(params.nbabies):
            numpy.random.shuffle(parents)
            baby=parents[:len(subpop[1])]
            nmutations=numpy.floor(len(baby)*numpy.random.rand()*params.mutation_rate).astype('int')
            alts=[i for i in range(params.ntasks) if not i in baby]
            numpy.random.shuffle(alts)
            for m in range(nmutations):
                mutpos=numpy.random.randint(len(baby))
                baby[mutpos]=alts[m]
            pop.append(baby)
    return pop

def immigrate_tasks(pop,params):
    immigrants=[]
    idx=[i for i in range(params.ntasks)]
    for i in range(params.nimmigrants):
        numpy.random.shuffle(idx)
        immigrants.append(idx[:params.nvars])
    return pop+immigrants

# functions for variable rather than task selection

def get_initial_population_vars(popsize,nvars,data,taskvaridx):
    poplist=[]
    for i in range(popsize):
        numpy.random.shuffle(taskvaridx)
        poplist.append(taskvaridx[:nvars])
    return(poplist)

def select_parents_vars(pop,data,nsel,clf,nsplits=4,obj_weight=[0.5,0.5]):
    cc,maxcc=get_population_fitness_vars(pop,data,nsplits,clf,obj_weight)
    idx=numpy.argsort(cc)[::-1]
    pop_sorted=[pop[i] for i in idx[:nsel]]
    cc_sorted=[cc[i] for i in idx[:nsel]]
    return(pop_sorted,cc_sorted,maxcc)

def get_population_fitness_vars(pop,data,nsplits,clf,obj_weight):
    # first get cc for each item in population
    if obj_weight[0]>0:
        cc_recon=[get_reconstruction_error_vars(cv,data,nsplits,clf) for cv in pop]
    else:
        cc_recon=[0]
    if obj_weight[1]>0:
        cc_subsim=[get_subset_corr_vars(cv,data) for cv in pop]
    else:
        cc_subsim=[0]
    maxcc=[numpy.max(cc_recon),numpy.max(cc_subsim)]
    cc_recon=scale(cc_recon)
    cc_subsim=scale(cc_subsim)
    cc=cc_recon*obj_weight[0] + cc_subsim*obj_weight[1]
    return cc,maxcc

def crossover_vars(pop,data,taskvaridx,nbabies=2,
                mutation_rate=None):
    if mutation_rate is None:
        mutation_rate=1/len(pop[0])
    # assortative mating - best parents mate
    families=numpy.kron(numpy.arange(numpy.floor(len(pop)/2)),[1,1])
    numpy.random.shuffle(families)
    for f in numpy.unique(families):
        famidx=[i for i in range(len(families)) if families[i]==f]
        if len(famidx)!=2:
            continue
        try:
            subpop=[pop[i] for i in famidx]
        except:
            print('oops...')
            print(len(pop))
            print(famidx)
            raise Exception('breaking')
        parents=list(numpy.unique(numpy.hstack((subpop[0],subpop[1]))))
        if len(set(parents))<len(subpop[1]):
            continue
        for b in range(nbabies):
            numpy.random.shuffle(parents)
            baby=parents[:len(subpop[1])]
            if numpy.random.randn()<mutation_rate:
                alts=[i for i in taskvaridx if not i in baby]
                numpy.random.shuffle(alts)
                mutpos=numpy.random.randint(len(baby))
                baby[mutpos]=alts[0]
            pop.append(baby)
    return pop


def immigrate_vars(pop,n,nvars,data,taskvaridx):
    immigrants=[]
    for i in range(n):
        numpy.random.shuffle(taskvaridx)
        immigrants.append(taskvaridx[:nvars])
    return pop+immigrants
