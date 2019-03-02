

import numpy
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
f='sims.o572552'

lines=[i.strip().split('"')[1].split(' ') for i in open(f).readlines() if i.find('[1]')>-1]

data=[]
for l in lines:
    d=[]
    for v in l:
        if v=='NA':
            d.append(numpy.nan)
        else:
            d.append(float(v))
    data.append(d)

df=pandas.DataFrame(data,columns=['shuffle','b1','simnum','corr_ZINB_cont','corr_Lasso_bin','corr_Lasso_cont'])
del df['simnum']

df_shuffle=df.query('shuffle==1')
df_noshuf=df.query('shuffle==0')
del df_noshuf['shuffle']
df_noshuf.groupby('b1').describe()
df_shuffle.groupby('b1').describe()
df_noshuf.groupby('b1').mean().plot()
plt.xlabel('signal parameter (b1)')
plt.ylabel('correlation(pred,actual)')
