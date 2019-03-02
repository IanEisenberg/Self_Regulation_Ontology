#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 14:29:45 2018

@author: ian
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

loading = np.zeros([9,3])
loading[0,0]=.59
loading[1,0]=.57
loading[2,0]=.46
loading[3,1]=.46
loading[4,1]=.45
loading[5,1]=.63
loading[6,2]=.57
loading[7,2]=.33
loading[8,2]=.4
loading = pd.DataFrame(loading)
loading.index = ['Plus-Minus', 'Number-Letter', 'Local-Global', 
                 'Keep Track', 'Tone-Monitoring', 'Lettery-Memory', 
                 'Antisaccade', 'Stop-Signal', 'Stroop']
loading.columns = ['Shifting', 'Updating', 'Inhibition']

plt.figure(figsize=(8,8));
sns.heatmap(loading, 
            vmax=.6, vmin=-.6, 
            cmap=sns.diverging_palette(220,15,n=100,as_cmap=True)); 
plt.tick_params(labelsize=24); 
plt.title('Loadings', fontsize=24)