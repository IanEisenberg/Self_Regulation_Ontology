import argparse
import glob
from os import path, remove
import seaborn as sns
import svgutils.transform as sg

from dimensional_structure.cross_results_plots import plot_corr_hist
from dimensional_structure.HCA_plots import plot_dendrogram, plot_subbranches
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_recent_dataset

# load data
# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', default=None)
args = parser.parse_args()

datafile = args.dataset
if datafile == None:
    datafile = get_recent_dataset()
results = load_results(datafile)
plot_file = path.dirname(results['task'].plot_dir)

# make histogram plot
colors = sns.color_palette('Blues_d',3)[0:2] + sns.color_palette('Reds_d',2)[:1]
f = plot_corr_hist(results,colors, reps=2)
f.savefig(path.join(plot_file, 'within-across_correlations.png'), 
                    bbox_inches='tight', 
                    dpi=300)
# Create and save dendorgram plots temporarily
scores = {}
for title, subset in [('Behavioral Tasks', 'task'), ('Self-Report Surveys', 'survey')]:
    r = results[subset]
    c = r.EFA.get_metric_cs()['c_metric-BIC']
    inp = 'EFA%s' % c
    plot_dendrogram(r, c, inp=inp, titles=[title],
                     figsize=(12,10), ext='svg',  plot_dir='/tmp/')
    # get scores
    scores[subset] = r.EFA.get_scores(c)
dendrograms = glob.glob('/tmp/dendrogram*.svg')

# get example branch
cluster_figs = plot_subbranches(results['task'], 
                               c=5, 
                               inp='EFA5', 
                               cluster_range=range(2,4), 
                               figsize=(4,6),
                               ext='svg')
cluster_figs += plot_subbranches(results['task'], 
                                   c=5, 
                                   inp='EFA5', 
                                   cluster_range=range(2,4), 
                                   figsize=(4,6),
                                   ext='svg')

# ***************************************************************************
# Ontology Figure
# ***************************************************************************


#plot_clusterings(results['all'], show_clusters=False, figsize=(10,10),
#                 plot_dir='/tmp/', ext='svg')


# combine them into one SVG file
# load matpotlib-generated figures
fig1 = sg.fromfile(dendrograms[0])
fig2 = sg.fromfile(dendrograms[1])
fig3 = sg.from_mpl(f, {})
fig4 = sg.from_mpl(cluster_figs[0], {})
fig5 = sg.from_mpl(cluster_figs[1], {})

#create new SVG figure
# set height and width based on constituent plots
size1 = [float(i[:-2]) for i in fig1.get_size()]
size2 = [float(i[:-2]) for i in fig2.get_size()]

width1 = max([size1[0], size2[0]]) 
width2 = float(fig3.get_size()[0])
width3 = float(fig4.get_size()[0])
wpad = (width1 + width2)*.02
width = width1 + width2 + wpad

height1 = size1[1]
height2 = size2[1]
hpad = (height1+height2)*.02
height = height1 + height2 + hpad


# create svg fig
fig = sg.SVGFigure(width, height)
fig.root.set("viewbox", "0 0 %s %s" % (width, height))

# get the plot objects
plot1 = fig1.getroot()
plot2 = fig2.getroot()
plot2.moveto(0, height1+hpad)
plot3 = fig3.getroot()
plot3.moveto(width1+wpad, 0)
plot4 = fig4.getroot()
plot4.moveto(width1+wpad*2, height1*.6+hpad)
plot5 = fig5.getroot()
plot5.moveto(width1+wpad*3+width3*1.2, height1*.6+hpad)
#plot4 = fig4.getroot()
#plot4.moveto(int(fig3.get_size()[0]) + wpad, height1+hpad)

# add text labels
txt1 = sg.TextElement(25,30, "A", size=30, weight="bold")
txt2 = sg.TextElement(25, 30+hpad+height1, "B", size=30, weight="bold")
txt3 = sg.TextElement(width1+wpad+25, 30, "C", size=30, weight="bold")
txt4 = sg.TextElement(width1+wpad+25, 
                      30+hpad+height1*.5, "D", size=30, weight="bold")

# append plots and labels to figure
fig.append([plot1, plot2, plot3, plot4, plot5])
fig.append([txt1, txt2, txt3, txt4])

# save generated SVG files
fig.save(path.join(plot_file, 'Ontology_Figure.svg'))

for file in dendrograms:
    remove(file)