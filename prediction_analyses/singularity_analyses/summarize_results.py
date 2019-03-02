import os,glob,pickle
import numpy,pandas
import scipy.stats
import selfregulation.prediction.behavpredict as behavpredict

basedir='/scratch/01329/poldrack/SRO'
#basedir='/Users/poldrack/Downloads'

files=glob.glob(os.path.join(basedir,'prediction_outputs/*pkl'))
files=[i for i in files if i.find('_rf_')>-1]
files.sort()
acc={}
features={}
accuracy=pandas.DataFrame()
if os.path.exists('rf_data.pkl'):
  acc,features=pickle.load(open('rf_data.pkl','rb'))
else:
  for f in files:
    d=pickle.load(open(f,'rb'))
    l_s=os.path.basename(f).replace('.pkl','').split('_')
    if l_s[3]=='shuffle':
        l_s[3]=l_s[4]
        l_s[1]=l_s[1]+'_shuffle'
    #data=
    if not l_s[1] in acc:
        acc[l_s[1]]={}
        features[l_s[1]]={}

    if not l_s[3] in acc[l_s[1]]:
        acc[l_s[1]][l_s[3]]=[d[0]]
        features[l_s[1]][l_s[3]]=[d[1]]
    else:
        acc[l_s[1]][l_s[3]].append(d[0])
        features[l_s[1]][l_s[3]].append(d[1])
  print('saving data to pickle')
  pickle.dump((acc,features),open('rf_data.pkl','wb'))

skip_vars=['RetirementPercentStocks',
'HowOftenFailedActivitiesCannabis',
'HowOftenHazardousCannabis',
'SpouseParentsComplainDrugUse',
'NeglectedFamilyDrugUse',
'MedicalProblemsDueToDrugUse']

bp=behavpredict.BehavPredict(verbose=True,
     drop_na_thresh=100,n_jobs=1,
     skip_vars=skip_vars)
bp.load_demog_data()
bp.get_demogdata_vartypes()
data=[]
nfiles={}

def mk_scripts(t,n,v):
    njobs=8
    if t.find('shuffle')>-1:
        shuffle='-s'
    else:
        shuffle=''
    container='/work/01329/poldrack/stampede2/singularity_images/selfregulation-2017-10-10-646e0b351ab0.img'
    with open('%s_sing.sh'%t,'a') as f:
        for i in range(n):
           f.write("singularity run -e %s /workdir/Self_Regulation_Ontology/prediction_analyses/behav_prediction.py -d %s -r /scratch/01329/poldrack/SRO -j %d %s --singlevar %s\n"%(container,t.replace('_shuffle',''),njobs,shuffle,v))

singfiles=glob.glob('*sing.sh')
for s in singfiles:
    os.remove(s)
for t in acc:
    print('')
    bp.load_behav_data(t.replace('_shuffle',''))
    nfiles[t]={}
    for v in bp.demogdata.columns.tolist():
        if not v in acc[t]:
            nfiles[t][v]=0
            continue
        else:
            nfiles[t][v]=len(acc[t][v])
        #print(t,v,nfiles[t][v])
        meanacc=numpy.mean(acc[t][v])
        lower5pct=scipy.stats.scoreatpercentile(acc[t][v],5)
        upper5pct=scipy.stats.scoreatpercentile(acc[t][v],95)
        meanfeaturevals=numpy.mean(features[t][v],0)
        maxfeatidx=numpy.argmax(meanfeaturevals)
        minfeatidx=numpy.argmin(meanfeaturevals)
        #print(v,bp.data_models[v],meanacc,lower5pct,
        #    upper5pct,bp.behavdata.columns[maxfeatidx],bp.behavdata.columns[minfeatidx])
        data.append([t,v,bp.data_models[v],meanacc,lower5pct,upper5pct,
            bp.behavdata.columns[maxfeatidx],meanfeaturevals[0,maxfeatidx],
            bp.behavdata.columns[minfeatidx],meanfeaturevals[0,minfeatidx]])

for t in acc:
    for v in bp.demogdata.columns.tolist():
        if v in skip_vars:
            continue
        #print(t,v,100 -nfiles[t][v])
        if nfiles[t][v]<100:
            print(t,v,100-nfiles[t][v])
            mk_scripts(t,100 - nfiles[t][v],v)
