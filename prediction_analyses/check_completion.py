import os,glob

files=glob.glob('../results/prediction_outputs/*pkl')
files.sort()
datasets={}

for f in files:
    l_s=os.path.basename(f).replace('.pkl','').split('_')  
    if l_s[2]=='shuffle':
        l_s[2]=l_s[3]
        l_s[1]=l_s[1]+'_shuffle'
    if not l_s[1] in datasets:
        datasets[l_s[1]]={}
    if not l_s[2] in datasets[l_s[1]]:
        datasets[l_s[1]][l_s[2]]=1
    else:
        datasets[l_s[1]][l_s[2]]+=1

for t in datasets:
    print('')
    print(t)
    for v in datasets[t]:
        print(v,datasets[t][v])
