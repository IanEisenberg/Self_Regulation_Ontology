import os,glob

files=glob.glob('/data/01329/poldrack/SRO/lasso/prediction_outputs/*pkl')
files.sort()
datasets={}

for f in files:
    l_s=os.path.basename(f).replace('.pkl','').split('_')  
    if l_s[2]=='shuffle':
        l_s[2]=l_s[3]
        l_s[1]=l_s[1]+'_shuffle'
    if not l_s[1] in datasets:
        datasets[l_s[1]]={}
    if not l_s[3] in datasets[l_s[1]]:
        datasets[l_s[1]][l_s[3]]=1
    else:
        datasets[l_s[1]][l_s[3]]+=1

for t in datasets:
    print('')
    print(t,len(datasets[t]))
    for v in datasets[t]:
        print(v,datasets[t][v])
