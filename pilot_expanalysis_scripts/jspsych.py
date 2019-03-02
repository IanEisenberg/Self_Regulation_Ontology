'''
analysis/experiments/jspsych.py: part of expfactory package
jspsych functions
'''
import numpy
from expanalysis.experiments.utils import get_data, lookup_val, select_worker
from expanalysis.experiments.processing import extract_experiment

def calc_time_taken(data):
    '''Selects a worker (or workers) from results object and sorts based on experiment and time of experiment completion
    '''
    instruction_lengths = []
    exp_lengths = []
    for i,row in data.iterrows():
        if row['experiment_template'] == 'jspsych':
            exp_data = get_data(row)
            #ensure there is a time elapsed variable
            assert 'time_elapsed' in list(exp_data[-1].keys()), \
                '"time_elapsed" not found for at least one dataset in these results'
            #sum time taken on instruction trials
            instruction_length = numpy.sum([trial['time_elapsed'] for trial in exp_data if lookup_val(trial.get('trial_id')) == 'instruction'])        
            #Set the length of the experiment to the time elapsed on the last 
            #jsPsych trial
            experiment_length = exp_data[-1]['time_elapsed']
            instruction_lengths.append(instruction_length/1000.0)
            exp_lengths.append(experiment_length/1000.0)
        else:
            instruction_lengths.append(numpy.nan)
            exp_lengths.append(numpy.nan)
    data.loc[:,'total_time'] = exp_lengths
    data.loc[:,'instruct_time'] = instruction_lengths
    data.loc[:,'ontask_time'] = data['total_time'] - data['instruct_time']
        

def get_average_variable(results, var):
    '''Prints time taken for each experiment in minutes
    '''
    averages = {}
    for exp in results.get_experiments():
        data = extract_experiment(results,exp)
        try:
            average = data[var].mean()
        except TypeError:
            print("Cannot average %s" % (var))
        averages[exp] = average
    return averages
    
    
def get_post_task_responses(data):
    question_responses = [numpy.nan] * len(data)
    for i,row in data.iterrows():
        row_data = get_data(row)
        if row['experiment_template'] == 'jspsych':
            if row_data[-2].get('trial_id') =='post task questions' and \
                'responses' in list(row_data[-2].keys()):
                question_responses[i]= (row_data[-2]['responses'])
    data.loc[:,'post_task_responses'] = question_responses

    


    
    
    