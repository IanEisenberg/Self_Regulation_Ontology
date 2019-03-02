import os,glob
import pickle
clf='lasso'
minsize=10

indir='/work/01329/poldrack/stampede2/code/Self_Regulation_Ontology/prediction_analyses/singularity_analyses/ls5/results/prediction_outputs'
indir='/data/01329/poldrack/SRO/lasso/prediction_outputs'
files=glob.glob(os.path.join(indir,'pred*pkl'))
indir='/data/01329/poldrack/SRO/lasso_septask/prediction_outputs'
files=files+glob.glob(os.path.join(indir,'pred*pkl'))
indir='/data/01329/poldrack/SRO/lasso_septask/prediction_outputs_single'
files=files+glob.glob(os.path.join(indir,'pred*pkl'))

files.sort()
datasets={}
for f in files:
    l_s=os.path.basename(f).replace('.pkl','').split('_')  
    clfpos=[i for i,x in enumerate(l_s) if x == clf]
    dsname='_'.join(l_s[1:clfpos[0]])
    shufflepos=[i for i,x in enumerate(l_s) if x == 'shuffle']
    if len(shufflepos)>0:
        dsname=dsname+'_shuffle'
    #print(f,dsname,l_s)
    if not dsname in datasets:
        datasets[dsname]=[]
    datasets[dsname].append(f)
counter={}
completed={}
incomplete={}
allkeys=[]
allsets=['baseline','task','survey','baseline_shuffle','discounting','intelligence','stopping','impulsivity','big5','risktaking','grit','emotion','bisbas','thresh','drift','nondecision']
allsets = allsets + ['stroop',
 'dot_pattern_expectancy',
 'attention_network_task',
 'threebytwo',
 'stop_signal',
 'motor_selective_stop_signal',
 'kirby',
 'discount_titrate',
 'tower_of_london',
 'columbia_card_task_hot']
data={}
for t in allsets:
    if not t in datasets:
        print('nothing for',t)
        datasets[t]={}
    if not t in data:
        data[t]={}
    if not t in counter:
        counter[t]={}
        incomplete[t]={}
        completed[t]=[]
    print('')
    print(t,len(datasets[t]))
    for v in datasets[t]:

        d=pickle.load(open(v,'rb'))
        for k in d['data'].keys():
            if not k in data[t]:
                data[t][k]=[d['data'][k]]
            else:
                data[t][k].append(d['data'][k])
            if not k in counter[t]:
                allkeys.append(k)
                counter[t][k]=1
            else:
                counter[t][k]+=1
pickle.dump(data,open('%s_data.pkl'%clf,'wb'))
skip_vars=['RetirementPercentStocks',
         'HowOftenFailedActivitiesDrinking',
         'HowOftenGuiltRemorseDrinking',
         'AlcoholHowOften6Drinks']
allkeys=list(set(allkeys))        
for t in allsets:
    print(t)
    for k in allkeys:
        if k in skip_vars:
            continue
        if not k in counter[t]:
             incomplete[t][k]=100 
             continue
        if counter[t][k]>=100:
               completed[t].append(k)  
        else:
               incomplete[t][k]=100-counter[t][k]
               print(k,incomplete[t][k])	
    #print(t,len(completed[t]),'completed',len(incomplete[t]),'incomplete')
pickle.dump(incomplete,open('incomplete.pkl','wb'))

