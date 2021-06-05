# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from sklearn.datasets import load_boston
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import roc_auc_score,  roc_curve
from matplotlib import pyplot
from sklearn.metrics import auc
import seaborn as sns
from sklearn import metrics
import datetime
from sklearn.model_selection import GridSearchCV

print("xgb.__version__ : ",xgb.__version__)
data_dir= '/home/lpatel/projects/AKI/data_592v'
#data_dir= '~/projects/AKI/test'
#data_dir='/home/lpatel/projects/AKI/data'
train_csv = os.path.join(data_dir,'train_csv.csv')
test_csv = os.path.join(data_dir,'test_csv.csv')
weight_csv = os.path.join(data_dir,'weight_csv.csv')

train = pd.read_csv(train_csv)
test = pd.read_csv(test_csv)
weight = pd.read_csv(weight_csv)
#column names are formted inconsitantly 
weight['col_fmt'] = weight.col.str.replace('-','.').str.replace(':','.')


cols = train.columns.tolist()
X_col = cols[1:-1]
y_col = cols[-1]

X_train,y_train = train[X_col],train[y_col]
X_test,  y_test = test[X_col] ,test[y_col]

print(set(X_col) -set(weight.col_fmt.tolist()) )
print(set(weight.col_fmt.tolist()) - set(X_col) )

weight1_lst =  weight.set_index(keys=['col_fmt']).reindex(X_train.columns.tolist()).weight1.tolist()
weight2_lst =  weight.set_index(keys=['col_fmt']).reindex(X_train.columns.tolist()).weight2.tolist()
weight3_lst =  weight.set_index(keys=['col_fmt']).reindex(X_train.columns.tolist()).weight3.tolist()
weight4_lst =  weight.set_index(keys=['col_fmt']).reindex(X_train.columns.tolist()).weight4.tolist()
weight5_lst =  weight.set_index(keys=['col_fmt']).reindex(X_train.columns.tolist()).weight5.tolist()



# +
def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, 
                       model, param_grid, cv=10, scoring_fit = 'roc_auc',
                       do_probabilities = True):
    
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid, 
        cv=cv, 
        n_jobs=4, 
        scoring=scoring_fit,
        verbose=2
    )
    fitted_model = gs.fit(X_train_data, y_train_data)
    
    if do_probabilities:
        pred = fitted_model.predict_proba(X_test_data)
    else:
        pred = fitted_model.predict(X_test_data)
    
    return fitted_model, pred

model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_jobs = 6
)
param_grid = {
    'max_depth': [3, 6, 9],
    'n_estimators': [500, 1000, 1500],
    'colsample_bytree': [0.05,0.5,0.75],
    'subsample': [0.5, 0.75, 0.9],
    'objective': ['binary:logistic'],

}


model, pred  = algorithm_pipeline(X_train, X_test, y_train, y_test, model, 
                                 param_grid, cv=5)

data = pd.DataFrame(model.cv_results_)
# pd.options.display.max_columns = None
# pd.options.display.max_rows = None
print(data)
t = datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
data.to_csv("~/results_parm_cv.csv_weight1_lst" + t)
print ("done")

