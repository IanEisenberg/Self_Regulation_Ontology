import os

njobs=8
shuffle=False
container='/work/01329/poldrack/stampede2/singularity_images/selfregulation-2017-10-20-ebd01093c60d.img'
machine='wrangler'
classifier='lasso'
outdir='/data/01329/poldrack/SRO/%s'%classifier
if not os.path.exists(outdir):
    os.mkdir(outdir)

if classifier=='rf':
    clf='-c rf'
else:
    clf=''

if shuffle:
    datasets=['baseline']
    shuf_flag='-s'
    nruns=1000
else:
    datasets=['baseline','task','survey','discounting','stopping','intelligence']
    shuf_flag=''
    nruns=1000

f=open('%s/%s%s_all.sh'%(machine,classifier,shuf_flag),'w')
for t in datasets:
   for i in range(nruns):
       f.write("singularity run -e %s /workdir/Self_Regulation_Ontology/prediction_analyses/behav_prediction.py -d %s -r %s -j %d %s %s\n"%(container,t,outdir,njobs,shuf_flag,clf)) 
f.close()
