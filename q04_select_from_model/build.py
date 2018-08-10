# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')
# Your solution code here
def select_from_model(df):
    X = df.iloc[:,:-1]
    y = df.SalePrice

    rfc = RandomForestClassifier(random_state=9)
    rfc.fit(X,y)

    model = SelectFromModel(rfc,prefit=True)
    model.transform(X)
    cols = model.get_support()

    feature_name = list(X.columns[cols])
    return feature_name


