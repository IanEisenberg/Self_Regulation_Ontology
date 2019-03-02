from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dimensional_structure.utils import get_factor_groups
from selfregulation.utils.plot_utils import format_variable_names

def plot_factor_df(EFA, rotate='oblimin'):
    c = EFA.get_c()
    loadings = EFA.get_loading(c, rotate=rotate)
    loadings = EFA.reorder_factors(loadings, rotate=rotate)           
    grouping = get_factor_groups(loadings)
    flattened_factor_order = []
    for sublist in [i[1] for i in grouping]:
        flattened_factor_order += sublist
    loadings = loadings.loc[flattened_factor_order]
    loadings.index = format_variable_names(loadings.index)
    loadings.columns = loadings.columns.map(lambda x: str(x).ljust(15))

    # visualization functions
    def magnify():
        return [dict(selector="tr:hover",
                    props=[("border-top", "2pt solid black"),
                           ("border-bottom", "2pt solid black")]),
                dict(selector="th:hover",
                     props=[("font-size", "10pt")]),
                dict(selector="td",
                     props=[('padding', "0em 0em")]),
               # dict(selector="th:hover",
               #      props=[("font-size", "12pt")]),
                dict(selector="tr:hover td:hover",
                     props=[('max-width', '200px'),
                            ('font-size', '16pt')])
    ]
    
    cm =sns.diverging_palette(220,15,n=200)
    def color_cell(val):
        color = to_hex(cm[int(val*100)+100])
        return 'background-color: %s' % color

    
    styler = loadings.style
    styler \
        .applymap(color_cell) \
        .set_properties(**{'max-width': '100px', 'font-size': '0pt', 'border-color': 'white'})\
        .set_precision(2)\
        .set_table_styles(magnify())
    return styler

def plot_EFA_robustness(EFA_robustness):
    EFA_robustness = pd.DataFrame(EFA_robustness).T
    EFA_robustness.index = [' '.join(i.split('_')) for i in EFA_robustness.index]
    min_robustness = EFA_robustness.min().min()
    def color(val):
        return 'color: white' if val <.9 else 'color: black'

    def cell_color(val):
        if val>.9:
            return 'background-color: None'
        else:
            cm =sns.color_palette('Reds', n_colors=100)[::-1]
            color = to_hex(cm[int((val-min_robustness)*50)])
            return 'background-color: %s' % color

    return EFA_robustness.style \
        .applymap(color) \
        .applymap(cell_color) \
        .set_properties(**{'font-size': '12pt', 'border-color': 'white'}) \
            .set_precision(3)

# helper plotting function
def plot_bootstrap_results(boot_stats):
    mean_loading = boot_stats['means']
    std_loading = boot_stats['sds']
    coef_of_variation = std_loading/mean_loading
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
    ax1.plot(mean_loading.values.flatten(), coef_of_variation.values.flatten(), 'o')
    ax1.set_xlabel('Mean Loading')
    ax1.set_ylabel('Coefficient of Variation')
    ax1.set_ylim([-1,1])
    ax1.grid()

    ax2.plot(mean_loading.values.flatten(), std_loading.values.flatten(),'o')
    ax2.set_xlabel('Mean Loading')
    ax2.set_ylabel('Standard Deviation of Loading')

    plt.subplots_adjust(wspace=.5)

def plot_HCA_AMI(consensus):
    simulation_AMI_values = consensus.AMI_scores
    mean_simulation = np.mean(simulation_AMI_values)
    consensus_AMI = consensus.compare_clusters()
    f = plt.figure(figsize=(12,8))
    f = sns.kdeplot(simulation_AMI_values, shade=True, color='grey')
    ymin, ymax = f.get_ylim()
    f.set_xlim(0,1)
    plt.axvline(consensus_AMI, ymin, ymax, linestyle='--')
    plt.xlabel('Adjusted mutual Information (AMI)')
    plt.title('AMI vs. Original Clustering')
    plt.text(consensus_AMI, ymax*.75, 
             'Consensus\nCluster', ha='center', fontsize=25,
            color='red')
    plt.text(mean_simulation, ymax*.05, 
             'Distribution\nover\nSimulations', ha='center', fontsize=25,
            color='grey')
    
def display_closest_DVs(consensus, n_closest=10):
    nth = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
        8: "eigth",
        9: "ninth",
        10: "tenth",
    }
    df = consensus.get_consensus_cluster()['distance_df']
    df.index = format_variable_names(df.index)
    df.columns = format_variable_names(df.columns)

    sorted_df = pd.DataFrame(data=np.zeros((len(df),n_closest)), index=df.index)
    sorted_df.columns = [nth[i+1] for i in sorted_df.columns]
    for name, row in sorted_df.iterrows():
        closest = 1-df.loc[name].drop(name).sort_values()[:n_closest]
        closest = ['%s: %s%%' % (i,int(b*100)) for i,b in closest.iteritems()]
        sorted_df.loc[name] = closest
        
    def magnify():
        return [dict(selector="tr:hover",
                    props=[("border-top", "2pt solid black"),
                           ("border-bottom", "2pt solid black")]),
                dict(selector="th:hover",
                     props=[("font-size", "10pt")]),
                dict(selector="td",
                     props=[('padding', "0em 0em")]),
               # dict(selector="th:hover",
               #      props=[("font-size", "12pt")]),
                dict(selector="tr:hover td:hover",
                     props=[('max-width', '200px'),
                            ('font-weight', 'bold'),
                            ('color', 'black'),
                           ('font-size', '9pt')])
    ]

    cm =sns.diverging_palette(220,15,n=161)
    def color_cell(val):
        val = val[val.rindex(': ')+2:val.rindex('%')]
        color = to_hex(cm[int(val)+30])
        return 'background-color: %s' % color


    styler = sorted_df.style
    styler \
        .applymap(color_cell) \
        .set_properties(**{'max-width': '100px','font-size': '10pt', 'border-color': 'white'})\
        .set_precision(2)\
        .set_table_styles(magnify())
    return styler

def display_cluster_DVs(consensus, results):
    nth = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
        8: "eigth",
        9: "ninth",
        10: "tenth",
        11: "11th",
        12: "12th",
        13: "13th",
        14: "14th",
        15: "15th",
        16: "16th",
        17: "17th",
        18: "18th",
        19: "19th",
        20: "20th",
    }
    c = results.EFA.get_c()
    cluster_DVs = results.HCA.get_cluster_DVs(inp='EFA%s_oblimin' % c)
    df = consensus.get_consensus_cluster()['distance_df']
    sorted_df = pd.DataFrame(data=np.zeros((len(df),20)), index=df.index)
    for name, row in sorted_df.iterrows():
        neighbors = [v for v in cluster_DVs.values() if name in format_variable_names(v)][0]
        neighbors = format_variable_names(neighbors)
        closest = 1-df.loc[name, neighbors].drop(name).sort_values()
        closest = ['%s: %s%%' % (i,int(b*100)) for i,b in closest.iteritems()]
        closest += ['' for _ in range(20-len(closest))]
        sorted_df.loc[name] = closest

    def magnify():
        return [dict(selector="tr:hover",
                    props=[("border-top", "2pt solid black"),
                           ("border-bottom", "2pt solid black")]),
                dict(selector="th:hover",
                     props=[("font-size", "10pt")]),
                dict(selector="td",
                     props=[('padding', "0em 0em")]),
               # dict(selector="th:hover",
               #      props=[("font-size", "12pt")]),
                dict(selector="tr:hover td:hover",
                     props=[('max-width', '200px'),
                            ('font-weight', 'bold'),
                            ('color', 'black'),
                           ('font-size', '9pt')])
    ]

    cm =sns.diverging_palette(220,15,n=161)
    def color_cell(val):
        if val =='':
            return 'background-color: None'
        num = val[val.rindex(': ')+2:val.rindex('%')]
        color = to_hex(cm[int(num)+30])
        return 'background-color: %s' % color


    styler = sorted_df.style
    styler \
        .applymap(color_cell) \
        .set_properties(**{'max-width': '100px',  'font-size': '10pt', 'border-color': 'white'})\
        .set_precision(2)\
        .set_table_styles(magnify())
    return styler