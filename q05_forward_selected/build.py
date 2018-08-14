# %load q05_forward_selected/build.py
# Default imports
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict

data = pd.read_csv('data/house_prices_multivariate.csv')
model = LinearRegression()
# Your solution code here
def forward_selected(df,model):

    X = df.iloc[:,:-1]
    y = df.SalePrice

    colnames = X.columns
    r2Score_list = []
    variables = []

    max_scores = 0.0
    df = pd.DataFrame()
    for i in range(len(X.columns)):
        max_r2_score = 0.0
        selected_col = ''
        for j in X.columns:
            if j not in variables:
                input_df = df.copy()
                input_df[j] = X[j]
                model.fit(input_df, y)
                y_pred = model.predict(input_df)
                r2score = r2_score(y, y_pred)
                if max_r2_score < r2score:
                    max_r2_score = r2score
                    selected_col = j
        df[selected_col] = X[selected_col]
        variables.append(selected_col)
        r2Score_list.append(max_r2_score)
    return variables, r2Score_list        


#forward_selected(data,model)    

#expected_acc = [0.61972765016619102, 0.7110122362921284, 0.74208020244393813, 0.76370229136595302,
#                0.77146549956264021, 0.77743942439428682, 0.78190516290253775,
#                0.78559309845190683, 0.78926329832950681, 0.79084962683577575]

