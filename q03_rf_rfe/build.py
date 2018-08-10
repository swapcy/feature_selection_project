# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd
import numpy as np
data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
# Your solution code here

def rf_rfe(df):
    X = df.iloc[:,:-1]
    y = df['SalePrice']

    model = RandomForestClassifier()
    nfeatures_to_select = (len(X.columns))//2
    rfe = RFE(model,n_features_to_select=nfeatures_to_select)
    rfe = rfe.fit(X,y)

    rfe_support  = rfe.support_
    rank = (rfe.ranking_)
    topFeatures = list(X.columns[rfe_support])

    #print((rank))
    #print(rfe_support)
        
    return topFeatures


