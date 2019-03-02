'''
expanalysis/plot.py: part of expfactory package
plotting functions
'''
import seaborn as sns
from matplotlib import pyplot as plt
import pandas



def plot_groups(df, groupby):
    print('hi')
    # subset summary to just mean for plotting
    try:
        df = df[groupby].join(df._get_numeric_data())
    except ValueError:
        numeric_df = df._get_numeric_data()
        drop_cols = [col for col in groupby if col in numeric_df.columns]
        numeric_df = numeric_df.drop(drop_cols, axis = 1)
        df = df[groupby].join(numeric_df)
    df = pandas.melt(df, groupby)
    if len(groupby)== 0:
        p = sns.factorplot(y = 'value', data = df, kind = 'bar', col = 'variable', sharey = False, col_wrap = 3)
    elif len(groupby) == 1:
        p = sns.factorplot(x = groupby[0], y = 'value', data = df, kind = 'bar', col = 'variable', sharey = False, col_wrap = 3)
    elif len(groupby) == 2:
        p = sns.factorplot(x = groupby[0], y = 'value', hue = groupby[1], data = df, kind = 'bar', col = 'variable', sharey = False, col_wrap = 3)
    elif len(groupby) > 2:
        p = sns.factorplot(x = groupby[0], y = 'value', hue = groupby[1], data = df, kind = 'bar', col = 'variable', row = groupby[2], sharey = False)  
        if len(groupby) > 3:
             print('Only grouping by %s, %s and %s' % (groupby[0], groupby[1], groupby[2]))
    plt.show()
    return p
    
    

    
