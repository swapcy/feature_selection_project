# %load q02_best_k_features/build.py
# Default imports

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression

# Write your solution here:
def percentile_k_features(df,k=20):
    y = df['SalePrice']
    X = df.loc[:,data.columns !='SalePrice']   
    
    kpsec = SelectPercentile(score_func=f_regression, percentile=k)
    percentileCols = kpsec.fit_transform(X,y)    
    getIndices = np.asarray(kpsec.get_support(indices=True))
    scores = kpsec.scores_ 
    sorted_scores = np.argsort(scores)[::-1]
    
    list_cols = []
    for ind in sorted_scores:
        if(ind in getIndices):
            list_cols.append(X.columns[ind])
        
    #print(list_cols)
    
    return list_cols

percentile_k_features(data)


