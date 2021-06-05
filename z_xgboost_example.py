# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# https://xgboost.readthedocs.io/en/latest/python/python_api.html
# obj (function) – Customized objective function.
# feval (function) – Customized evaluation function

from sklearn.datasets import load_boston
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

boston = load_boston()
# -

# print(boston.keys());print(boston.data.shape);print(boston.feature_names);print(boston.DESCR);data.head()

data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
data['PRICE'] = boston.target

data.info()

X, y = data.iloc[:,:-1],data.iloc[:,-1]
data_dmatrix = xgb.DMatrix(data=X,label=y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)
print (X_train.shape)
#s = np.random.uniform(-1,0,1000)
xg_reg = xgb.XGBRegressor(
    objective ='reg:linear',
    colsample_bytree = 0.3,
    learning_rate = 0.1,
    max_depth = 5,
    alpha = 10,
    n_estimators = 2,
)

# param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
# num_round = 2
#
# bst = xgb.train(param, dtrain, num_round)
# # make prediction
# preds = bst.predict(dtest)

xg_reg.fit(X_train,y_train,
           #sample_weight = s
           )

preds = xg_reg.predict(X_test)
# -

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


