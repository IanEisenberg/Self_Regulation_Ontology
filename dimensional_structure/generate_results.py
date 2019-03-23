# Running script to create results objects for subsets of data and plot

# imports
import argparse
# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', default=None)
parser.add_argument('-no_analysis', action='store_false')
parser.add_argument('-no_prediction', action='store_false')
parser.add_argument('-no_plot', action='store_false')
parser.add_argument('-no_group_analysis', action='store_false')
parser.add_argument('-no_group_plot', action='store_false')
parser.add_argument('-run_change', action='store_true')
parser.add_argument('-bootstrap', action='store_true')
parser.add_argument('-boot_iter', type=int, default=1000)
parser.add_argument('-shuffle_repeats', type=int, default=1)
parser.add_argument('-subsets', nargs='+', default=['task', 'survey'])
parser.add_argument('-classifiers', nargs='+', default=['lasso', 'ridge',  'svm', 'rf'])
parser.add_argument('-plot_backend', default=None)
parser.add_argument('-dpi', type=int, default=300)
parser.add_argument('-size', type=int, default=4.6)
parser.add_argument('-ext', default='pdf')
parser.add_argument('-quiet', action='store_false')
args = parser.parse_args()

dataset = args.dataset
run_analysis = args.no_analysis
run_prediction = args.no_prediction
run_change = args.run_change
run_plot = args.no_plot
group_analysis = args.no_group_analysis
group_plot = args.no_group_plot
bootstrap = args.bootstrap
boot_iter = args.boot_iter
shuffle_repeats = args.shuffle_repeats
classifiers = args.classifiers
selected_subsets = args.subsets
verbose = args.quiet

# import matplotlib and set backend
import matplotlib
if args.plot_backend:
    matplotlib.use('Agg')
    
# imports
from glob import glob
import numpy as np
from os import makedirs, path, remove
import pickle
import random
from shutil import copyfile, copytree, rmtree
import subprocess
import time

from dimensional_structure.results import Results
from dimensional_structure.cross_results_plots import (plot_corr_heatmap, 
                                                       plot_glasso_edge_strength,
                                                       plot_cross_within_prediction,
                                                       plot_cross_relationship,
                                                       plot_BIC,
                                                       plot_cross_silhouette,
                                                       plot_cross_communality)
from dimensional_structure.cross_results_utils import run_cross_prediction
from dimensional_structure.DA_plots import plot_DA
from dimensional_structure.EFA_plots import plot_EFA
from dimensional_structure.EFA_test_retest import (calc_EFA_retest,
                                                   plot_EFA_change, 
                                                   plot_EFA_retest, 
                                                   plot_cross_EFA_retest)
from dimensional_structure.HCA_plots import plot_HCA
from dimensional_structure.prediction_plots import (plot_results_prediction,
                                                    plot_prediction, 
                                                    plot_prediction_scatter,
                                                    plot_prediction_comparison,
                                                    plot_factor_fingerprint)
from dimensional_structure.prediction_utils import run_group_prediction
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_info, get_recent_dataset


if verbose:
    print('Running Analysis? %s, Prediction? %s, Plotting? %s, Bootstrap? %s, Selected Subsets: %s' 
        % (['No', 'Yes'][run_analysis],  
            ['No', 'Yes'][run_prediction], 
            ['No', 'Yes'][run_plot], 
            ['No', 'Yes'][bootstrap],
            ', '.join(selected_subsets)))

# get dataset of interest
basedir=get_info('base_directory')
if dataset == None:
    dataset = get_recent_dataset()
dataset = path.join(basedir,'Data',dataset)
datafile = dataset.split(path.sep)[-1]

# label subsets
demographic_factor_names = ['Drug Use',
                            'Mental Health',
                            'Problem Drinking',
                            'Daily Smoking',
                            'Binge Drinking',
                            'Lifetime Smoking',
                            'Obesity',
                            'Income / Life Milestones']


                                      
subsets = [{'name': 'task', 
            'regex': 'task',
            'oblimin_cluster_names': ['Conflict Processing',
                                      'Information Processing',
                                      'Shifting',
                                      'Speeded Information Processing',
                                      'Inhibition-Related Threshold',
                                      'Caution',
                                      'Perc/Resp',
                                      'Inhibition-Related Perc/Resp',
                                      'NA1',
                                      'Discounting',
                                      'NA2',
                                      'Cold/Model-Based',
                                      'Hot/Model-Free',
                                      'NA3',
                                      'NA4'],
            'oblimin_factor_names': ['Speeded IP', 'Strategic IP', 
                                     'Perc / Resp','Caution', 
                                     'Discounting']
                                     ,
            'varimax_cluster_names': None,
            'varimax_factor_names': ['Speeded IP', 'Strategic IP', 
                                     'Perc / Resp',  'Caution', 
                                     'Discounting'],
            'predict': True},
            {'name': 'survey',
             'regex': 'survey',
             'oblimin_cluster_names': ['Financial Risk-Taking',
                                       'Eating',
                                       'Behavioral Approach',
                                       'Behavioral Inhibition',
                                       'Mindfulness',
                                       'Impulsivity',
                                       'Goal-Direcedness',
                                       'Ethical/Health Risk-Taking',
                                       'Risk Perception',
                                       'Sensation Seeking',
                                       'Sociability',
                                       'Reward Sensitivity'],
             'oblimin_factor_names':  ['Sensation Seeking', 'Emotional Control',  
                                   'Mindfulness', 'Impulsivity',
                                   'Reward Sensitivity', 'Goal-Directedness', 
                                   'Risk Perception', 'Eating Control', 
                                   'Ethical Risk-Taking', 'Social Risk-Taking',
                                   'Financial Risk-Taking', 'Agreeableness'],
            'varimax_cluster_names': None,
            'varimax_factor_names': None,
             'predict': True},
             {'name': 'main_subset', 
            'regex': 'main',
            'oblimin_cluster_names': [],
            'oblimin_factor_names': [],
            'predict': False},
             {'name': 'all', 
              'regex': '.',
              'oblimin_cluster_names': [],
              'oblimin_factor_names': [],
              'predict': False}]
results = None
all_results = None
ID = str(random.getrandbits(16)) 
# create/run results for each subset
for subset in subsets:
    name = subset['name']
    if verbose:
        print('*'*79)
        print('SUBSET: %s' % name.upper())
        print('*'*79)
    if selected_subsets is not None and name not in selected_subsets:
        continue
    if run_analysis == True:
        print('*'*79)
        print('Analyzing Subset: %s' % name)
        # ****************************************************************************
        # Laad Data
        # ****************************************************************************
        # run dimensional analysis
        start = time.time()
        results = Results(datafile=datafile, 
                          dist_metric='abscorrelation',
                          name=subset['name'],
                          filter_regex=subset['regex'],
                          boot_iter=boot_iter,
                          ID=ID,
                          residualize_vars=['Age', 'Sex'])
        results.run_demographic_analysis(verbose=verbose, bootstrap=bootstrap)
        for rotate in ['oblimin', 'varimax']:
            results.run_EFA_analysis(rotate=rotate, 
                                     verbose=verbose, 
                                     bootstrap=bootstrap)
            results.run_clustering_analysis(rotate=rotate, 
                                            verbose=verbose, 
                                            run_graphs=False)
            c = results.EFA.get_c()
            # name factors and clusters
            factor_names = subset.get('%s_factor_names' % rotate, None)
            cluster_names = subset.get('%s_cluster_names' % rotate, None)
            if factor_names:
                results.EFA.name_factors(factor_names, rotate=rotate)
            if cluster_names:
                results.HCA.name_clusters(cluster_names, inp='EFA%s_%s' % (c, rotate))
        ID = results.ID.split('_')[1]
        results.DA.name_factors(demographic_factor_names)
        if verbose: print('Saving Subset: %s' % name)
        id_file = results.save_results()
        # ***************************** saving ****************************************
        # copy latest results and prediction to higher directory
        copyfile(id_file, path.join(path.dirname(results.get_output_dir()), 
                                    '%s_results.pkl' % name))

    if run_prediction == True:   
        if verbose:
            print('*'*79)
            print('Running prediction: %s' % name)
        if results is None or name not in results.ID:
            results = load_results(datafile, name=name)[name]
        # run behavioral prediction using the factor results determined by BIC
        for classifier in classifiers:
            for rotate in ['oblimin', 'varimax']:
                results.run_prediction(classifier=classifier, rotate=rotate, verbose=verbose)
                results.run_prediction(classifier=classifier, rotate=rotate, shuffle=shuffle_repeats, verbose=verbose) # shuffled
                # predict demographic changes
                if run_change:
                    results.run_change_prediction(classifier=classifier, rotate=rotate, verbose=verbose)
                    results.run_change_prediction(classifier=classifier, rotate=rotate, shuffle=shuffle_repeats, verbose=verbose) # shuffled
        # ***************************** saving ****************************************
        prediction_dir = path.join(results.get_output_dir(), 'prediction_outputs')
        new_dir = path.join(path.dirname(results.get_output_dir()), 'prediction_outputs')
        for classifier in classifiers:
            for change_flag in [False, True]:
                for subset in ['varimax', 'oblimin', 'raw']:
                    prediction_files = glob(path.join(prediction_dir, '*%s*' % classifier))
                    # filter by change
                    prediction_files = filter(lambda x: ('change' in x) == change_flag, prediction_files)
                    # filter by rorate
                    prediction_files = filter(lambda x: subset in x, prediction_files)
                    # sort by creation time and get last two files
                    prediction_files = sorted(prediction_files, key = path.getmtime)[-4:]
                    for filey in prediction_files:
                        filename = '_'.join(path.basename(filey).split('_')[:-1])
                        copyfile(filey, path.join(new_dir, '%s_%s.pkl' % (name, filename)))

    # ****************************************************************************
    # Plotting
    # ****************************************************************************
    dpi = args.dpi
    ext = args.ext
    size = args.size
    if run_plot==True:
        if verbose:
            print('*'*79)
            print('Plotting Subset: %s' % name)
        if results is None or name not in results.ID:
            results = load_results(datafile, name=name)[name]
        plot_dir = results.get_plot_dir()
        DA_plot_dir = path.join(plot_dir, 'DA')
        EFA_plot_dir = path.join(plot_dir, 'EFA')
        HCA_plot_dir = path.join(plot_dir, 'HCA')
        prediction_plot_dir = path.join(plot_dir, 'prediction')
        makedirs(DA_plot_dir, exist_ok = True)
        makedirs(EFA_plot_dir, exist_ok = True)
        makedirs(HCA_plot_dir, exist_ok = True)
        
        # set up kws for plotting functions
        tasks = np.unique([i.split('.')[0] for i in results.data.columns])
        if name == 'task':
            plot_task_kws= {'task_sublists': {'tasks': [t for t in tasks if 'survey' not in t]}}
        elif name == 'survey':
            plot_task_kws= {'task_sublists': {'surveys': [t for t in tasks if 'survey' in t]}}
        else:
            plot_task_kws={}
         
            # Plot EFA
        if verbose: print("** Plotting DA **")
        plot_DA(results, DA_plot_dir, verbose=verbose, size=size, dpi=dpi, ext=ext)
        
        for rotate in ['oblimin', 'varimax']:
            # Plot EFA
            if verbose: print("** Plotting EFA %s **" % rotate)
            plot_EFA(results, EFA_plot_dir, rotate=rotate,
                     verbose=verbose, size=size, dpi=dpi, 
                     ext=ext, plot_task_kws=plot_task_kws)
            
            # Plot EFA retest
            combined, *the_rest = calc_EFA_retest(results, rotate=rotate)
            plot_EFA_retest(combined=combined, 
                            plot_dir=path.join(EFA_plot_dir, rotate), 
                            size=size, dpi=dpi, ext=ext)
            plot_EFA_change(combined=combined, 
                            plot_dir=path.join(EFA_plot_dir, rotate),
                            size=size, dpi=dpi, ext=ext)
            # Plot HCA
            if verbose: print("** Plotting HCA %s **" % rotate)
            drop_list = {('task', 'oblimin'): ([1,5,8,9,12,15],[2,4,6,14]) ,
                         ('survey', 'oblimin'): ([0,2,4,6,8,10], None)}
            drop1, drop2 = drop_list.get((name, rotate), (None, None))
            plot_HCA(results, HCA_plot_dir, rotate=rotate,
                     drop_list = drop1, double_drop_list=drop2,
                     size=size, dpi=dpi, ext=ext)
        # Plot prediction
        if results.get_prediction_files() is not None:
            target_order = results.DA.reorder_factors(results.DA.get_loading()).columns
            change_target_order = [i + ' Change' for i in target_order]
            for classifier in classifiers:
                for EFA in [True, False]:
                    if EFA:
                        for rotate in ['oblimin', 'varimax']:
                            rotate_plot_dir = path.join(prediction_plot_dir, rotate)
                            print("** Plotting Prediction, classifier: %s, EFA: %s **" % (classifier, EFA))
                            plot_results_prediction(results, 
                                            target_order=target_order, 
                                            EFA=EFA, 
                                            rotate=rotate,
                                            classifier=classifier, 
                                            plot_dir=rotate_plot_dir,
                                            dpi=dpi,
                                            ext=ext,
                                            size=10)
                            plot_prediction_scatter(results, target_order=target_order, 
                                                    EFA=EFA, 
                                                    rotate=rotate,
                                                    classifier=classifier, 
                                                    plot_dir=rotate_plot_dir,
                                                    dpi=dpi,
                                                    ext=ext,
                                                    size=size)
                            """
                            print("** Plotting Change Prediction, classifier: %s, EFA: %s **" % (classifier, EFA))
                            try:
                                plot_prediction(results, target_order=change_target_order, 
                                                EFA=EFA, change=True,
                                                classifier=classifier,
                                                rotate=rotate,
                                                plot_dir=rotate_plot_dir,
                                                dpi=dpi,
                                                ext=ext,
                                                size=size)
                                plot_prediction_scatter(results, 
                                                        target_order=change_target_order, 
                                                        EFA=EFA, 
                                                        rotate=rotate,
                                                        change=True,
                                                        classifier=classifier, 
                                                        plot_dir=rotate_plot_dir,
                                                        dpi=dpi,
                                                        ext=ext,
                                                        size=size)
                            except AssertionError:
                                print('No shuffled data was found for %s change predictions, EFA: %s' % (name, EFA))
                            """
                            plot_factor_fingerprint(results, change=False, rotate=rotate,
                                                    classifier=classifier,
                                                    size=size, ext=ext, dpi=dpi, 
                                                    plot_dir=rotate_plot_dir)
                            plot_factor_fingerprint(results, change=True, rotate=rotate,
                                                    size=size, ext=ext, dpi=dpi, 
                                                    classifier=classifier,
                                                    plot_dir=rotate_plot_dir)
                    else:
                        plot_results_prediction(results, 
                                            target_order=target_order, 
                                            EFA=False, 
                                            rotate=rotate,
                                            classifier=classifier, 
                                            plot_dir=prediction_plot_dir,
                                            dpi=dpi,
                                            ext=ext,
                                            size=size)
                        plot_prediction_scatter(results, target_order=target_order, 
                                                EFA=False, 
                                                rotate=rotate,
                                                classifier=classifier, 
                                                plot_dir=prediction_plot_dir,
                                                dpi=dpi,
                                                ext=ext,
                                                size=size)
            plot_prediction_comparison(results, change=False, size=size, ext=ext,
                                       dpi=dpi, plot_dir=prediction_plot_dir)
            #plot_prediction_comparison(results, change=True, size=size, ext=ext, 
            #                           dpi=dpi, plot_dir=prediction_plot_dir)

        
        # copy latest results and prediction to higher directory
        generic_dir = '_'.join(plot_dir.split('_')[0:-1])
        if path.exists(generic_dir):
            rmtree(generic_dir)
        copytree(plot_dir, generic_dir)

# ****************************************************************************
# group analysis (across subsets)
# ****************************************************************************
        
if group_analysis == True:
    if verbose:
            print('*'*79)
            print('Running group analysis')
    all_results = load_results(datafile)
    for classifier in classifiers:
        ontology_prediction = run_group_prediction(all_results, 
                                                   shuffle=False, 
                                                   classifier=classifier,
                                                   include_raw_demographics=False,
                                                   rotate='oblimin',
                                                   verbose=False)
    for classifier in classifiers:
        ontology_prediction_shuffled = run_group_prediction(all_results, 
                                                            shuffle=shuffle_repeats, 
                                                            classifier=classifier,
                                                            include_raw_demographics=False, 
                                                            rotate='oblimin',
                                                            verbose=False)
    run_cross_prediction(all_results)
    a=subprocess.Popen('python gephi_graph_plot.py', shell=True)

if group_plot == True:
    if verbose:
        print('*'*79)
        print('*'*79)
        print("** Group Plots **")
    all_results = load_results(datafile)
    output_loc = path.dirname(all_results['task'].get_output_dir())
    prediction_loc = path.join(output_loc, 'prediction_outputs')
    plot_file = path.dirname(all_results['task'].get_plot_dir())
    # plotting
    # plot full prediction
    for classifier in classifiers:
        ontology_prediction_files = glob(path.join(prediction_loc, "EFA_task_survey*%s*" % classifier))
        ontology_prediction_files = sorted(ontology_prediction_files, key=path.getmtime)[-2:]
        ontology_prediction = open([i for i in ontology_prediction_files if 'shuffle' not in i][0], 'rb')
        ontology_prediction_shuffled = open([i for i in ontology_prediction_files if 'shuffle' in i][0], 'rb')
        ontology_prediction = pickle.load(ontology_prediction)
        ontology_prediction_shuffled = pickle.load(ontology_prediction_shuffled)
        survey_prediction = all_results['survey'].load_prediction_object(classifier=classifier)
        filename = path.join(plot_file, 'ontology_prediction.%s' % ext)
        target_order = all_results['task'].DA.reorder_factors(all_results['task'].DA.get_loading()).columns
        plot_prediction(ontology_prediction['data'],
                        ontology_prediction_shuffled['data'],
                        target_order=target_order,
                        size=size, dpi=dpi,
                        filename=filename)
        filename = path.join(plot_file, 'ontology_vs_survey_prediction_%s.%s' % (classifier, ext))
        plot_prediction(ontology_prediction['data'],
                        survey_prediction['data'],
                        comparison_label='Survey Prediction',
                        target_order=target_order,
                        size=size, dpi=dpi,
                        filename=filename)
    # plot other cross results
    plot_corr_heatmap(all_results, size=size*1/2, ext=ext, dpi=dpi, plot_dir=plot_file)
    plot_BIC(all_results, size=size, ext=ext, dpi=dpi, plot_dir=plot_file)
    # rotation dependent
    for rotate in ['oblimin', 'varimax']:
        plot_cross_silhouette(all_results, rotate=rotate, size=size, ext=ext, 
                              dpi=dpi, plot_dir=plot_file)
        plot_cross_communality(all_results,rotate=rotate,  size=size, ext=ext, 
                               dpi=dpi, plot_dir=plot_file)
        plot_cross_EFA_retest(all_results, rotate=rotate, size=size, ext=ext, 
                              dpi=dpi, plot_dir=plot_file)
        plot_cross_EFA_retest(all_results, rotate=rotate, annot_heatmap=True, 
                              size=size, ext=ext, dpi=dpi, plot_dir=plot_file)
    # plot analysis overview
    a=subprocess.Popen('python analysis_overview_plot.py -dataset %s -dpi %s -ext %s -size %s' \
                           % (datafile, dpi, ext, 4.6),
                           shell=True)
    
    # plot cross prediction
    prediction_loc = path.join(output_loc, 'cross_prediction.pkl')
    if path.exists(prediction_loc):
        plot_cross_within_prediction(prediction_loc, 
                                     size=size*1/4, ext=ext, dpi=dpi, plot_dir=plot_file)
    # plot graph
    graph_loc = path.join(output_loc,'graph_results', 'weighted_graph.pkl')
    if path.exists(graph_loc):
        plot_glasso_edge_strength(all_results,
                              graph_loc, 
                              size=size*1/4, ext=ext, dpi=dpi, plot_dir=plot_file)
    else:
        print('No graph data found!')
    if path.exists(prediction_loc) and path.exists(graph_loc):
        plot_cross_relationship(all_results, graph_loc, prediction_loc,
                        size=4.6, ext='pdf', plot_dir=plot_file)
# ****************************************************************************
# move plots to paper directory
# ****************************************************************************
if run_plot or group_plot:
    if verbose:
            print('*'*79)
            print('Moving plots to paper directory')
    if all_results is not None:
        plot_file = path.dirname(all_results['task'].get_plot_dir())
    else:
        plot_file = path.dirname(results.get_plot_dir())
    
    rotate = 'oblimin'
    exhaustive_lookup = {
            'analysis_overview': 'Fig01_Analysis_Overview',
            'survey/HCA/%s/dendrogram_EFA12_%s' % (rotate, rotate): 'Fig03_Survey_Dendrogram',
            'task/HCA/%s/dendrogram_EFA5_%s' % (rotate, rotate): 'Fig04_Task_Dendrogram',
            'survey/prediction/EFA_ridge_prediction_bar': 'Fig05_Survey_prediction',
            'task/prediction/EFA_ridge_prediction_bar': 'Fig06_Task_prediction',
            # test-retest
            'cross_relationship': 'FigS02_cross_relationship',
            'BIC_curves': 'FigS03_BIC_curves',
            'survey/EFA/factor_correlations_EFA12': 'FigS04_Survey_2nd-order',
            'task/EFA/factor_correlations_EFA5': 'FigS05_Task_2nd-order',
            '%s/communality_adjustment' % rotate: 'FigS06_communality',
            '%s/EFA_test_retest' % rotate: 'FigS07_EFA_retest',
            'survey/EFA/factor_heatmap_EFA12': 'FigS08_Survey_EFA',
            'task/EFA/factor_heatmap_EFA5': 'FigS09_Task_EFA',
            'survey/HCA/dendrogram_data': 'FigS10_Survey_Raw_Dendrogram',
            'task/HCA/dendrogram_data': 'FigS11_Task_Raw_Dendrogram',
            '%s/silhouette_analysis' % rotate: 'FigS12_Survey_Silhouette',
            # survey clusters
            # task clusters
            'task/DA/factor_heatmap_DA9': 'FigS15_Outcome_EFA',
            'task/DA/factor_correlations_DA9': 'FigS16_Outcome_2nd-order',
            'survey/prediction/IDM_lasso_prediction_bar': 'FigS17_Survey_IDM_prediction',
            'task/prediction/IDM_lasso_prediction_bar': 'FigS18_Task_IDM_prediction',
            'survey/prediction/EFA_ridge_factor_fingerprint': 'FigS19_Survey_Factor_Fingerprints'
            }
    
    shortened_lookup = {
            'analysis_overview': 'Fig01_Analysis_Overview',
            '%s/EFA_test_retest' % rotate: 'Fig03_EFA_retest',
            'survey/HCA/%s/dendrogram_EFA12_%s' % (rotate, rotate): 'Fig04_Survey_Dendrogram',
            'task/HCA/%s/dendrogram_EFA5_%s' % (rotate, rotate): 'Fig05_Task_Dendrogram',
            'survey/prediction/%s/EFA_ridge_prediction_bar' % rotate: 'Fig06_Survey_prediction',
            'task/prediction/%s/EFA_ridge_prediction_bar' % rotate: 'Fig07_Task_prediction',
            'survey/prediction/%s/EFA_ridge_factor_fingerprint' % rotate: 'Fig08_Survey_Factor_Fingerprints',
            # test-retest
            'cross_relationship': 'FigS02_cross_relationship',
            'BIC_curves': 'FigS03_BIC_curves',
            '%s/communality_adjustment' % rotate: 'FigS04_communality',
            '%s/silhouette_analysis' % rotate: 'FigS05_HCA_Silhouettes',
            'ontology_vs_survey_prediction_ridge': 'FigS06_ontology_prediction',
            'task/DA/factor_correlations_EFA8': 'FigS07_demo_correlation'
            }
    
    paper_dir = path.join(basedir, 'Results', 'Psych_Ontology_Paper')
    figure_lookup = shortened_lookup
    for filey in figure_lookup.keys():
        remove_orig = False
        figure_num = figure_lookup[filey].split('_')[0]
        orig_file = path.join(plot_file, filey+'.'+ext)
        new_file = path.join(paper_dir, 'Plots', figure_lookup[filey]+'.'+ext)
        if ext == 'eps' and path.exists(orig_file):
            a=subprocess.Popen('epspdf %s' % (orig_file),
                             shell=True, 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE)
            a.wait()
            orig_file = orig_file.replace('.eps', '.pdf')
            new_file = new_file.replace('.eps', '.pdf')
            remove_orig = True
        if figure_num[-1] in 'abcdefg':
            copyfile(orig_file,  new_file)
        else:
            a=subprocess.Popen('cpdf -scale-to-fit "4.6in PH mul 4.6in div PW" %s -o %s' % (orig_file, new_file),
                             shell=True, 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE)
            out, err = a.communicate()
            if 'cpdf: not found' in str(err):
                try:
                    copyfile(orig_file, 
                             new_file)
                except FileNotFoundError:
                    print('%s not found' % filey)
            elif 'No such file or directory' in str(err):
                print('%s not found' % filey)
        if remove_orig:
            remove(orig_file)
