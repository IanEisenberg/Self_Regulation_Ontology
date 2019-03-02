import fancyimpute
import numpy as np
from os import path
import pandas as pd
import seaborn as sns
from sklearn import decomposition
import sys

sys.path.append('../utils')
from utils import get_behav_data
from plot_utils import dendroheatmap

# get dependent variables
DV_df = get_behav_data('Discovery_9-26-16', use_EZ = True)


# ************************************
# ************ Imputation *******************
# ************************************
DV_df_complete = fancyimpute.SoftImpute().complete(DV_df)
DV_df_complete = pd.DataFrame(DV_df_complete, index = DV_df.index, columns = DV_df.columns)

# ************************************
# ************ PCA *******************
# ************************************

pca_data = DV_df_complete.corr()
pca = decomposition.PCA()
pca.fit(pca_data)

# plot explained variance vs. components
sns.plt.plot(pca.explained_variance_ratio_.cumsum())

# plot loadings of 1st component
sns.barplot(np.arange(200),pca.components_[0])

# dimensionality reduction
pca.n_components = 2
reduced_df = pca.fit_transform(pca_data)
sns.plt.scatter(reduced_df[:,0], reduced_df[:,1])

def top_variables(pca, labels, n = 5):
    components = pca.components_
    variance = pca.explained_variance_ratio_
    output = []
    for c,v in zip(components,variance):
        order = np.argsort(np.abs(c))[::-1]
        output.append(([labels[o] for o in order[:n]],v))
    return output
    

top = top_variables(pca, pca_data.columns)