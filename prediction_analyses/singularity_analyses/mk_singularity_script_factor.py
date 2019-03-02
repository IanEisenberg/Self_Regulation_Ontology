import os,sys
container_tag='2017-11-03-6260a014e642'
machine='wrangler'
machine='stampede2'
machine='ls5'
classifier='lasso'
njobs=1
if len(sys.argv)>1:
   shuffle=True
   print('shuffling')
else:
   shuffle=False

container='/work/01329/poldrack/stampede2/singularity_images/selfregulation-%s.img'%container_tag
outdir='/scratch/01329/poldrack/SRO/%s_factor'%classifier
if not os.path.exists(outdir):
    os.mkdir(outdir)

if classifier=='rf':
    clf='-c rf'
else:
    clf=''
datasets=['task','survey']
datasets=datasets+['impulsivity','big5','risktaking','grit','emotion_regulation','bisbas'] 
datasets=datasets+['stroop',
 'dot_pattern_expectancy',
 'attention_network_task',
 'threebytwo',
 'stop_signal',
 'motor_selective_stop_signal',
 'kirby',
 'discount_titrate',
 'tower_of_london',
 'columbia_card_task_hot']
shuf_flag=''
nruns=120
with open('../variables_filtered.txt') as f:
    variables=[i.strip() for i in f.readlines()]
cmds=[]
for i in range(nruns):
  for t in datasets:
      cmds.append("singularity run -e %s /workdir/Self_Regulation_Ontology/prediction_analyses/behav_prediction.py -d %s -r %s -j %d %s %s --demogfile /workdir/Self_Regulation_Ontology/Data/Derived_Data/Complete_10-08-2017/factor_scores.csv --no_baseline_vars \n"%(container,t,outdir,njobs,shuf_flag,clf)) 
import random
#random.shuffle(cmds)
with open('%s/%s_tasks_factor.sh'%(machine,classifier),'w') as f:
  for c in cmds:
    f.write(c)
