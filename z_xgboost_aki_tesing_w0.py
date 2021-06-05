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

#geting feature importance for the best round
#params = {'booster': 'gbtree', 'max_depth': 10, 'min_child_weight': 10, 'eta': 0.3, 'objective': 'binary:logistic', 'n_jobs': 20, 'silent': True, 'eval_metric': 'logloss', 'subsample': 0.8, 'colsample_bytree': 0.5, 'seed': 1001}
model = xgb.XGBClassifier(
  booster= 'gbtree', max_depth= 10, min_child_weight= 10, eta= 0.3, objective= 'binary:logistic', n_jobs= 20, silent= True, eval_metric= 'logloss', subsample= 0.8, colsample_bytree= 0.5, seed= 1001
)
model.fit(X_train, y_train)
print(model.get_xgb_params)
df= pd.DataFrame({'cols':X_train.columns,'feature_importances' :model.feature_importances_ }).sort_values(by='feature_importances',ascending=False)
t = datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
df.to_csv("/home/lpatel/aki/results/feature_importance_tesing.csv"+t+'_w0',index=False)

exit(0)

# +
# def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, 
#                        model, param_grid, cv=10, scoring_fit = 'roc_auc',
#                        do_probabilities = True):
    
#     gs = GridSearchCV(
#         estimator=model,
#         param_grid=param_grid, 
#         cv=cv, 
#         n_jobs=4, 
#         scoring=scoring_fit,
#         verbose=2
#     )
#     fitted_model = gs.fit(X_train_data, y_train_data)
    
#     if do_probabilities:
#         pred = fitted_model.predict_proba(X_test_data)
#     else:
#         pred = fitted_model.predict(X_test_data)
    
#     return fitted_model, pred

# model = xgb.XGBClassifier(
#     objective='binary:logistic',
#     n_jobs = 6
# )
# param_grid = {
#     'max_depth': [3, 6, 9],
#     'n_estimators': [500, 1000, 1500],
#     'colsample_bytree': [0.05,0.5,0.75],
#     'subsample': [0.5, 0.75, 0.9],
#     'objective': ['binary:logistic'],

# }

# # ddddddddddddddddddd


# model, pred  = algorithm_pipeline(X_train, X_test, y_train, y_test, model, 
#                                  param_grid, cv=5)

# data = pd.DataFrame(model.cv_results_)
# # pd.options.display.max_columns = None
# # pd.options.display.max_rows = None
# print(data)
# t = datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
# data.to_csv("~/results_parm_cv.csv_weight1_lst" + t)
# print ("done")


# +
AUC_LIST = []
LOG_LOSS_LIST = []
ITERbest_LIST = []
PARAM_LIST = []

dtrain = xgb.DMatrix(X_train, label = y_train)


def XGB_CV(max_depth,
          # n_estimators, 
           colsample_bytree, subsample, min_child_weight,eta):


    global AUC_LIST
    global LOG_LOSS_LIST
    global ITERbest_LIST
    global PARAM_LIST
    
    #print(n_estimators)

    paramt = {
              'booster' : 'gbtree',
              'max_depth' :  int(max_depth),
              'min_child_weight' : int(min_child_weight),
#               'n_estimators': int(n_estimators),
              'eta' : float(eta),
              'objective' : 'binary:logistic',
              'n_jobs' : 20,
              'silent' : True,
              'eval_metric': 'logloss',
              'subsample' : max(min(subsample, 1), 0),
              'colsample_bytree' : max(min(colsample_bytree, 1), 0),
              'seed' : 1001
              }
    
    PARAM_LIST.append(paramt)

    folds = 5
    cv_score = 0

    print("\n Search parameters (%d-fold validation):\n %s" % (folds, paramt), file=log_file )
    log_file.flush()

    xgbc = xgb.cv(
                    paramt,
                    dtrain,
                    #num_boost_round = int(n_estimators),
                    stratified = True,
                    nfold = folds,
                    early_stopping_rounds = 100,
                    metrics = ['auc', 'logloss'],
                    show_stdv = True
               )



    auc_score = xgbc['test-auc-mean'].iloc[-1]
    logloss_score = xgbc['test-logloss-mean'].iloc[-1]
    iterbest = len(xgbc)
    AUC_LIST.append(auc_score)
    LOG_LOSS_LIST.append(logloss_score)
    ITERbest_LIST.append(iterbest)
    

    return (auc_score*2) - 1



XGB_BO = BayesianOptimization(XGB_CV, {
                                     'max_depth': (4, 10),
#                                      'n_estimators': (1, 10),
                                     'colsample_bytree': (0.5, 0.9),
                                     'subsample': (0.5, 0.8),
                                     'min_child_weight':(1,10),
                                     'eta':(0.05,0.3)
                                    })


# +
t = datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
log_file = open('/home/lpatel/aki/results/test.log'+t, 'a')
log_file.flush()

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    XGB_BO.maximize(init_points=10, n_iter=100)

# +
df = pd.DataFrame({"auc": AUC_LIST, "log": LOG_LOSS_LIST, "round": ITERbest_LIST, "param": PARAM_LIST })
df['param'] =  df['param'].astype(str)

t = datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
df.to_csv("/home/lpatel/aki/results/cv_result_baysian.csv"+t+"_w0", sep="|")
# -

print (len(ITERbest_LIST),len(PARAM_LIST),len(LOG_LOSS_LIST),len(AUC_LIST))

#print(weight2_lst)


