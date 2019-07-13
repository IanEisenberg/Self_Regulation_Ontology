"""
some util functions
"""
from glob import glob
import os
import pandas,numpy
import re
from sklearn.metrics import confusion_matrix
import pkg_resources
from collections import OrderedDict


# Regex filtering helper functions
def not_regex(txt):
        return '^((?!%s).)*$' % txt

def filter_behav_data(data, filter_regex):
    """ filters dataframe using regex
    Args:
        filter_regex: regex expression to filter data columns on. Can also supply
            "survey(s)" or "task(s)" to return measures associated with those
    """
    # main variables used to reduce task data down to its "main" variables
    main_vars = ['adaptive_n_back.mean_load',
             'angling_risk_task_always_sunny\..*_adjusted_clicks',
             'angling_risk_task_always_sunny\.release_adjusted_clicks',
             'attention_network_task\.alerting',
             'attention_network_task\.orienting',
             'attention_network_task\.conflict',
             'bickel_titrator\.hyp_discount_rate_medium',
             'choice_reaction_time',
             'cognitive_reflection_survey\.correct_proportion',
             'columbia_card_task_cold',
             'columbia_card_task_hot'
             'dietary_decision',
             'digit_span',
             'directed_forgetting\.proactive_interference_hddm_drift',
             'discount_titrate\.percent_patient',
             'dot_pattern_expectancy\.AY-BY',
             'dot_pattern_expectancy\.BX-BY',
             'go_nogo',
             'hierarchical_rule\.score',
             'holt_laury_survey\.risk_aversion',
             'information_sampling_task\..*P_correct',
             'keep_track\.score',
             'kirby\.hyp_discount_rate_medium',
             'local_global_letter\.conflict',
             'local_global_letter\.global',
             'local_global_letter\.switch',
             'motor_selective_stop_signal\.SSRT',
             'probabilistic_selection',
             'psychological_refractory_period_two_choices',
             'ravens\.score',
             'recent_probes\.proactive_interference',
             'shape_matching\.stimulus_interference',
             'shift_task\.model',
             'simon\.simon',
             'simple_reaction_time',
             'spatial_span',
             'stim_selective_stop_signal\.SSRT',
             '^stop_signal\.SSRT_high',
             'stroop\.stroop',
             'threebytwo\.cue_switch',
             'threebytwp\.task_switch',
             'tower_of_london\.planning_time',
             'two_stage_decision\.model'
             ]
    # filter columns if filter_regex is set
    if filter_regex.rstrip('s').lower() == 'survey':
        regex = not_regex(not_regex('survey')+'|cognitive_reflection|holt')
    elif filter_regex.rstrip('s').lower() == 'task':
        regex = not_regex('survey')+'|cognitive_reflection|holt'
    elif filter_regex.lower() == 'main':
        regex = '|'.join(main_vars)
    else:
        regex = filter_regex
    return data.filter(regex=regex)

def get_var_category(var):
    ''' Return "task" or "survey" classification for variable

    var: variable name passed as a string
    '''
    m = re.match(not_regex('survey')+'|cognitive_reflection|holt', var)
    if m is None:
        return 'survey'
    else:
        return 'task'

# Data get methods
def sorting(L):
    date = L.split('_')[-1]
    month,day,year = date.split('-')
    return year, month, day

def get_recent_dataset():
    basedir=get_info('base_directory')
    files = glob(os.path.join(basedir,'Data/Complete*'))
    files.sort(key=sorting)
    dataset = files[-1].split(os.sep)[-1]
    return dataset

def get_behav_data(dataset=None, file=None, filter_regex=None,
                flip_valence=False, verbose=False, full_dataset=None):
    '''Retrieves a file from a data release.

    By default extracts meaningful_variables from the most recent Complete dataset.

    Args:
        dataset: optional, string indicating discovery, validation, or complete dataset of interest
        file: optional, string indicating the file of interest
        filter_regex: regex expression to filter data columns on. Can also supply
            "survey(s)" or "task(s)" to return measures associated with those
        flip_valence: bool, default false. If true use DV_valence.csv to flip variables based on their subjective valence
    '''
    if full_dataset is not None:
        print("Full dataset is deprecrated and no longer functional")

    basedir=get_info('base_directory')
    if dataset == None:
        dataset = get_recent_dataset()
    datadir = os.path.join(basedir,'Data',dataset)
    if file == None:
        file = 'meaningful_variables.csv'
    if verbose:
        print('Getting dataset: %s...:\n' 'file: %s \n ' % (datadir, file))
    datafile=os.path.join(datadir,file)
    if os.path.exists(datafile):
        data=pandas.read_csv(datafile,index_col=0)
    else:
        data = pandas.DataFrame()
        print('Error: %s not found in %s' % (file, datadir))
        return None

    def valence_flip(data, flip_list):
        for c in data.columns:
            try:
                data.loc[:,c] = data.loc[:,c] * flip_list.loc[c]
            except TypeError:
                continue
    if flip_valence==True:
        print('Flipping variables based on valence')
        flip_df = os.path.join(datadir, 'DV_valence.csv')
        valence_flip(data, flip_df)
    if filter_regex is not None:
        data = filter_behav_data(data, filter_regex=filter_regex)
    return data.sort_index()

def get_retest_data(dataset):
    retest_data = get_behav_data(dataset, file='bootstrap_merged.csv.gz')
    if retest_data is None:
        return
    retest_data = retest_data[~retest_data.index.isnull()]
    retest_data = retest_data.loc[:, ['dv','icc3.k', 'spearman', 'pearson']]
    for column in retest_data.columns[1:]:
        retest_data[column] = pandas.to_numeric(retest_data[column])
    retest_data = retest_data.groupby('dv').mean()    
    retest_data.rename(index={'dot_pattern_expectancy.BX.BY_hddm_drift': 'dot_pattern_expectancy.BX-BY_hddm_drift',
                        'dot_pattern_expectancy.AY.BY_hddm_drift': 'dot_pattern_expectancy.AY-BY_hddm_drift'},
                        inplace=True)
    return retest_data
    
def get_info(item,infile=None):
    """
    get info from settings file
    """
    config=pkg_resources.resource_string('selfregulation',
                        'data/Self_Regulation_Settings.txt')
    config=str(config,'utf-8').strip()
    infodict={}
    for l in config.split('\n'):
        if l.find('#')==0:
            continue
        l_s=l.rstrip('\n').split(':')
        if len(l_s)>1:
                infodict[l_s[0]]=l_s[1]
    if (item == 'dataset') and (not 'dataset' in infodict):
        files = glob(os.path.join(infodict['base_directory'],'Data/Complete*'))
        files.sort(key=sorting)
        datadir = files[-1]
        return os.path.basename(datadir)
    try:
        assert item in infodict
    except:
        raise Exception('infodict does not include requested item: %s' % item)
    return infodict[item]

def get_item_metadata(survey, dataset=None,verbose=False):
    data = get_behav_data(dataset=dataset, file=os.path.join('Individual_Measures',
                                                             '%s.csv.gz' % survey))

    metadata = []
    for i in data.question_num.unique():
        item = data[data['question_num'] == i].iloc[0].to_dict()
        # drop unnecessary variables
        for drop in ['battery_name', 'finishtime', 'required', 'response',
                     'response_text', 'worker_id','experiment_exp_id']:
            try:
                item.pop(drop)
            except KeyError:
                continue
        if type(item['options']) != list:
            if verbose:
                print(item['options'])
            item['options'] = eval(item['options'])
        # turn options into an ordered dict, indexed by option number
        item['responseOptions']=OrderedDict()
        for o in item['options']:
            option_num=int(o['id'].split('_')[-1])
            o.pop('id')
            o['valueOrig']=option_num
            try:
                v=int(o['value'])
            except ValueError:
                v=o['value']
            if v in item['responseOptions']:
                item['responseOptions'][v]['valueOrig']=[item['responseOptions'][v]['valueOrig'],o['valueOrig']]
                item['responseOptions'][v]['text']=[item['responseOptions'][v]['text'],o['text']]
            else:
                item['responseOptions'][v]=o.copy()
                item['responseOptions'][v].pop('value')
        # scoring
        values = [int(i['value']) for i in item['options']]
        sorted_values = list(range(1,len(values)+1))
        cc=numpy.corrcoef(values,sorted_values)[0,1]
        if cc>0.5:
            item['scoring'] = 'Forward'
        elif cc<0.5:
            item['scoring'] = 'Reverse'
        else:
            item['scoring'] = 'other'
        # convert from numpy.int64 since it's not json serializable
        item['question_num']=int(item['question_num'])
        item_s=item['id'].replace('_options','').split('_')
        item['expFactoryName']='_'.join(item_s[:-1])+'.'+item_s[-1]
        item.pop('id')
        item.pop('options')
        metadata.append(item)
    return metadata

def get_demographics(dataset=None, cleanup=True, num_response_thresh=10,
                     drop_categorical=True, verbose=False):
    """ Preprocess and return demographic data
    
    Args:
        dataset: optional, which data release to draw from. The most recent one
            will be used if not specified
        cleanup: bool, indicated whether to remove data that is impossible
            (e.g. really low weights or heights)
        num_response_thresh: int, number of NaN responses allowed before removing
            variable
        drop_categorical: bool, whether to drop categorical variables
    
    """
    categorical_vars = ['HispanicLatino','Race',
                        'DiseaseDiagnoses', 'DiseaseDiagnosesOther',
                        'MotivationForParticipation', 'MotivationOther',
                        'NeurologicalDiagnoses',
                        'NeurologicalDiagnosesDescribe',
                        'OtherDebtSources',
                        'OtherDrugs', 'OtherRace', 'OtherTobaccoProducts',
                        'PsychDiagnoses',
                        'PsychDiagnosesOther']
        
    demogdata=get_behav_data(dataset,'demographic_health.csv')
    if cleanup:
        q=demogdata.query('WeightPounds<50')
        for i in q.index:
            demogdata.loc[i,'WeightPounds']=numpy.nan
        if verbose and len(q)>0:
            print('replacing bad WeightPounds value for', list(q.index))
        q=demogdata.query('HeightInches<36')
        for i in q.index:
            demogdata.loc[i,'HeightInches']=numpy.nan
        if verbose and len(q)>0:
            print('replacing bad HeightInches value for', list(q.index))
        q=demogdata.query('CaffienatedSodaCansPerDay<0')
        for i in q.index:
            demogdata.loc[i,'CaffienatedSodaCansPerDay']=numpy.nan
        q=demogdata.query('CaffieneOtherSourcesDayMG>2000')
        for i in q.index:
            demogdata.loc[i,'CaffieneOtherSourcesDayMG']=numpy.nan
        if verbose and len(q)>0:
            print('replacing bad CaffienatedSodaCansPerDay value for', list(q.index))

    demogdata=demogdata.assign(BMI=demogdata['WeightPounds']*0.45 / (demogdata['HeightInches']*0.025)**2)
    if drop_categorical:
       demogdata.drop(categorical_vars, axis=1, inplace=True)
       if verbose: 
           print('dropping categorical variables')
    demogdata=demogdata.assign(Obese=(demogdata['BMI']>30).astype('int'))
        
    # only keep variables with fewer NaNs then num_response_thresh
    if num_response_thresh is not None:
        good_vars = demogdata.isnull().sum() <= num_response_thresh
        demogdata = demogdata.loc[:,good_vars]
    return demogdata
                  
def get_single_dataset(dataset,survey):
    basedir=get_info('base_directory')
    infile=os.path.join(basedir,'data/Derived_Data/%s/surveydata/%s.tsv'%(dataset,survey))
    print(infile)
    assert os.path.exists(infile)
    if survey.find('ordinal')>-1:
        survey=survey.replace('_ordinal','')
    mdfile=os.path.join(basedir,'data/Derived_Data/%s/metadata/%s.json'%(dataset,survey))
    print(mdfile)
    assert os.path.exists(mdfile)
    data=pandas.read_csv(infile,index_col=0,sep='\t')
    metadata=load_metadata(survey,os.path.join(basedir,
        'data/Derived_Data/%s/metadata'%dataset))
    return data,metadata


def get_survey_data(dataset):
    basedir=get_info('base_directory')
    infile=os.path.join(basedir,'Data/Derived_Data/%s/surveydata.csv'%dataset)
    surveydata=pandas.read_csv(infile,index_col=0)
    keyfile=os.path.join(basedir,'Data/Derived_Data/%s/surveyitem_key.txt'%dataset)
    with open(keyfile) as f:
        keylines=[i.strip().split('\t') for i in f.readlines()]
    surveykey={}
    for k in keylines:
        surveykey[k[0]]=k[2]
    return surveydata,surveykey

def print_confusion_matrix(y_true,y_pred,labels=[0,1]):
    cm=confusion_matrix(y_true,y_pred)
    print('Confusion matrix')
    print('\t\tPredicted')
    print('\t\t0\t1')
    print('Actual\t0\t%d\t%d'%(cm[0,0],cm[0,1]))
    print('\t1\t%d\t%d'%(cm[1,0],cm[1,1]))