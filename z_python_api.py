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
from bayes_opt import BayesianOptimization
import warnings

def sendable_float_to_cpp(lst, colsample_bytree_weight_factor):
        return tuple([round(n * colsample_bytree_weight_factor) for n in lst])

print("xgb.__version__ : ",xgb.__version__)
data_dir= '/home/lpatel/projects/AKI/data_592v'

train_csv = os.path.join(data_dir,'train_csv.csv')
test_csv = os.path.join(data_dir,'test_csv.csv')
weight_csv = os.path.join(data_dir,'weight_csv.csv')

train = pd.read_csv(train_csv
        , nrows=10
        )
test = pd.read_csv(test_csv
        , nrows=10
        )
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
#weight1_lst = [1,2,4]
weight2_lst =  weight.set_index(keys=['col_fmt']).reindex(X_train.columns.tolist()).weight2.tolist()
weight3_lst =  weight.set_index(keys=['col_fmt']).reindex(X_train.columns.tolist()).weight3.tolist()
weight4_lst =  weight.set_index(keys=['col_fmt']).reindex(X_train.columns.tolist()).weight4.tolist()
weight5_lst =  weight.set_index(keys=['col_fmt']).reindex(X_train.columns.tolist()).weight5.tolist()

#colsample_bytree_weight=(4,6)
colsample_bytree_weight=tuple(weight1_lst)
colsample_bytree_weight_factor=10000
sendable_colsample_bytree_weight = sendable_float_to_cpp(colsample_bytree_weight,colsample_bytree_weight_factor)

print('py____colsample_bytree_weight', colsample_bytree_weight)
print('py____sendable_colsample_bytree_weight', sendable_colsample_bytree_weight)

model_iteration =2
xgb_model=None
for i in range(model_iteration):
        print('\n',"model_iteration:",i,'\n')
        model = xgb.XGBClassifier(n_estimators=1,booster= 'gbtree', max_depth= 2 , min_child_weight= 10, eta= 0.3, objective= 'binary:logistic', n_jobs= 20, silent= True, eval_metric= 'logloss',
                subsample= 0.8, colsample_bytree= 0.5, seed= 1001, colsample_bytree_weight=sendable_colsample_bytree_weight, colsample_bytree_weight_factor=colsample_bytree_weight_factor,
                xgb_model=None)
        model.fit(X_train, y_train)
        xgb_model='model.model'
        model.save_model(xgb_model)


print(model.get_xgb_params)
df= pd.DataFrame({'cols':X_train.columns,'feature_importances' :model.feature_importances_ }).sort_values(by='feature_importances',ascending=False)
t = datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
df.to_csv("/home/lpatel/aki/results/feature_importance_tesing.csv"+t+'_w0',index=False)

exit(0)
