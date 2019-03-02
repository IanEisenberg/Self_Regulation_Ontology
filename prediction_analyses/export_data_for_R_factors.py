"""
export data from pickle into text files that can be loaded into R
"""

import pickle,numpy
import pandas
import os
from selfregulation.utils.utils import get_info
import selfregulation.prediction.behavpredict_V1 as behavpredict

clf='lasso'
data=pickle.load(open('singularity_analyses/ls5/%s_data.pkl'%clf,'rb'))
outdir='R_exports_%s_factor'%clf
if not os.path.exists(outdir):
    os.mkdir(outdir)
if not os.path.exists('%s/features'%outdir):
    os.mkdir('%s/features'%outdir)

# Note: factors were misnamed because R doesn't spit them out
# in numeric order

factor_renaming_dict={'smoking severity':'MentalIllness',
                      'mental illness':'ProblemDrinking',
                    'smoking':'Obesity',
                      'obesity':'Smoking',
                     'alcohol':'SmokingSeverity',
                      'domestic':'Financial'}


predvars={}

acc_frames={}
feat_frames={}

maxdata=120
accvars=['scores_cv','scores_insample_unbiased']
print('creating data frames for:')
for k in data.keys():
    print(k)
    acc_frames[k]={}
    feat_frames[k]={}
    for v in data[k]:
        if not 'importances' in data[k][v][0]:
            print('no importances for',k,v)
            continue
        if not k in predvars:
            predvars[k]=data[k][v][0]['predvars']
            assert len(predvars[k])==data[k][v][0]['importances'].shape[1]
        if not v in acc_frames[k]:
            acc_frames[k][v]={}
            feat_frames[k][factor_renaming_dict[v]]=pandas.DataFrame()
            for accvar in accvars:
                acc_frames[k][v][accvar]=pandas.DataFrame()
        if len(data[k][v])>maxdata:
            data[k][v]=data[k][v][:maxdata]
        for accvar in accvars:
            if not accvar in data[k][v][0]:
                continue
            for i in range(len(data[k][v])):
                if not accvar in data[k][v][i]:
                    continue
                if len(data[k][v][i][accvar])==1:
                    df=pandas.DataFrame(data[k][v][i][accvar],
                        index=['AUROC']).T
                else:
                    df=pandas.DataFrame(data[k][v][i][accvar],
                        index=['r2','MAE']).T
                acc_frames[k][v][accvar]=acc_frames[k][v][accvar].append(df)
                if accvar==accvars[0]:
                    feats=pandas.DataFrame(data[k][v][i]['importances'],
                        columns=predvars[k])
                    feat_frames[k][factor_renaming_dict[v]]=feat_frames[k][factor_renaming_dict[v]].append(feats)

# now reformat so that frames contain all vars for each output type
output_frames={}
insample_frames={}
print('')
print('reformatting data frames for:')
for k in data.keys():
    print(k)
    output_frames[k]={}
    for v in acc_frames[k].keys():
        if not k in insample_frames:
            insample_frames[k]=pandas.DataFrame([])
        outvars=list(acc_frames[k][v]['scores_cv'].columns)
        for ov in outvars:
            if not ov in output_frames[k]:
                output_frames[k][ov]={}

            for accvar in accvars:
                if not ov in acc_frames[k][v][accvar]:
                    continue
                tmpdata=acc_frames[k][v][accvar][ov].values
                # fill any missing entires with NaN
                if not accvar in output_frames[k][ov]:
                    output_frames[k][ov][accvar]=pandas.DataFrame({'tmp':numpy.zeros(maxdata)})
                    output_frames[k][ov][accvar][v]=numpy.nan*numpy.zeros(maxdata)
                    output_frames[k][ov][accvar][v][:tmpdata.shape[0]]=tmpdata
                else:
                    output_frames[k][ov][accvar][v]=numpy.nan*numpy.zeros(maxdata)
                    output_frames[k][ov][accvar][v][:tmpdata.shape[0]]=tmpdata

    # rename columns
    for v in acc_frames[k].keys():
        outvars=list(acc_frames[k][v]['scores_cv'].columns)
        for ov in outvars:
            for accvar in accvars:
                if 'tmp' in output_frames[k][ov][accvar]:
                    del output_frames[k][ov][accvar]['tmp']
                for c in output_frames[k][ov][accvar].columns:
                    if not c in factor_renaming_dict:
                        continue
                    output_frames[k][ov][accvar][factor_renaming_dict[c]]=output_frames[k][ov][accvar][c]
                    del output_frames[k][ov][accvar][c]
print('')
print('writing data files')
for k in output_frames.keys():
    print(k)
    for ov in output_frames[k].keys():
        for accvar in accvars:
            index=False
            if accvar.find('insample')>-1:
                tmp=output_frames[k][ov][accvar].mean(0)
                index=True
            else:
                tmp=output_frames[k][ov][accvar]
            tmp.to_csv('%s/%s_%s_%s.csv'%(outdir,k,ov,accvar.replace('scores_','')),index=index)
    for v in feat_frames[k]:
        feat_frames[k][v].to_csv('%s/features/%s_%s_features.csv'%(outdir,k,v),index=False)

pickle.dump((output_frames,feat_frames),open('singularity_analyses/ls5/%s_data_collapsed.pkl'%clf,'wb'))
