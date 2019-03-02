import os,glob,pickle
from joblib import Parallel, delayed
import time

njobs=2
basedir='/Users/poldrack/code/Self_Regulation_Ontology/results/prediction_outputs'
#basedir='/Users/poldrack/Downloads'
clf='lasso'
#clf='rf'
files=glob.glob(os.path.join(basedir,'*pkl'))
files=[i for i in files if i.find('_%s_'%clf)>-1]
files.sort()
print('found %d files'%len(files))

start = time.time()
def load_data(f):
    output={}
    d=pickle.load(open(f,'rb'))
    l_s=os.path.basename(f).replace('.pkl','').split('_')
    if l_s[3]=='shuffle':
        l_s[3]=l_s[4]
        l_s[1]=l_s[1]+'_shuffle'
    #data=
    if not l_s[1] in output:
        output[l_s[1]]={}

    if not l_s[3] in output[l_s[1]]:
        output[l_s[1]][l_s[3]]=d
    else:
        output[l_s[1]][l_s[3]].append(d)
    return output

output=Parallel(n_jobs=njobs)(delayed(load_data)(f) for f in files)
print('saving data to pickle')
pickle.dump(output,open('%s_data.pkl'%clf,'wb'))

end = time.time()
print('elapsed time:',end - start)
