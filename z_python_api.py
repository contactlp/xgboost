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
import pprint


def sendable_float_to_cpp(lst, colsample_bytree_weight_factor):
        return tuple([round(n * colsample_bytree_weight_factor) for n in lst])


def fmap(trees):
    fmap = {}
    for tree in trees:
        for line in tree.split('\n'):
            # look for the opening square bracket
            arr = line.split('[')
            # if no opening bracket (leaf node), ignore this line
            if len(arr) == 1:
                continue

            # extract feature name from string between []
            fid = arr[1].split(']')[0].split('<')[0]

            if fid not in fmap:
                # if the feature hasn't been seen yet
                fmap[fid] = 1
            else:
                fmap[fid] += 1
    return fmap


def find_view_weights(w_list):
    
    view_weight = {}
    features_per_view = {}
    
    for feature, weight in zip(X_train.columns.tolist(),w_list):
        view_weight[feature] = weight
        
        if weight not in features_per_view:
            features_per_view[weight]= [feature]
        else:
            features_per_view[weight].append(feature)
            
    return (features_per_view,view_weight)


def find_new_view_importance(last_round_feature_weight,current_w):
    
    # how many view are there ?
    features_per_view, feature_weight = find_view_weights(w[current_w])
    unused_views = set(features_per_view)
    
    # find view which has not been used
    for feature in last_round_feature_weight:
        #print(feature)
        unused_views = unused_views - set([feature_weight[feature]])
            
    # min weight of last round features
    min_feat_weight = min(last_round_feature_weight.values())
    
    # give weight if there is unused view
    views_weight = {}
    if len(unused_views)!= 0:
        for view in unused_views:
            views_weight[view] = min_feat_weight * 0.9
    
    # used view weight adding
    for feature in last_round_feature_weight:
        view = feature_weight[feature]
        if view not in views_weight:
            views_weight[view] = last_round_feature_weight[feature]
        else:
            views_weight[view] = views_weight[view] + last_round_feature_weight[feature]
    return  views_weight 
            

def normalize_dict_values(d):
    output = {}
    total = sum(d.values())
    for i in d:
        output[i] = d[i]/total
    return output


def MyCallback():
    def callback(env):
        print('\n------------------starting callback------------------')
        trees = env.model.get_dump(with_stats=True)
        feature_weight = fmap(trees)
        #pp.pprint(trees)
        
        global gain 
        gain = env.model.get_score(importance_type='gain')
        print("\n gain %s" %(gain))
        print('\n------------------ending callback------------------')
    return callback


def read_csvs(data_dir,nrows=None):
    train_csv = os.path.join(data_dir,'train_csv.csv')
    test_csv = os.path.join(data_dir,'test_csv.csv')
    weight_csv = os.path.join(data_dir,'weight_csv.csv')
    
    if nrows==None:
        train = pd.read_csv(train_csv)
        test = pd.read_csv(test_csv)
    else:
        train = pd.read_csv(train_csv, nrows=nrows)
        test = pd.read_csv(test_csv, nrows=nrows)
    weight = pd.read_csv(weight_csv)
    
    print ("train shape:%s , test shape: %s , weight shape: %s" %(train.shape,test.shape,weight.shape))
    #column names are formted inconsitantly 
    weight['col_fmt'] = weight.col.str.replace('-','.').str.replace(':','.')
    
    return (train,test,weight)


def convert_to_dmatix(train,test,weight):
    cols = train.columns.tolist()
    X_col = cols[1:-1]
    y_col = cols[-1]

    X_train,y_train = train[X_col],train[y_col]
    X_test,  y_test = test[X_col] ,test[y_col]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    
    print("missing cols X vs Weights : ",set(X_col) -set(weight.col_fmt.tolist()) )
    print("missing cols Weights vs X : ",set(weight.col_fmt.tolist()) - set(X_col) )

    return (X_train, X_test, dtrain, dtest)


def find_all_weight(weight,X_train):
    
    weight1 =  weight.set_index(keys=['col_fmt']).reindex(X_train.columns.tolist()).weight1.tolist()
    weight2 =  weight.set_index(keys=['col_fmt']).reindex(X_train.columns.tolist()).weight2.tolist()
    weight3 =  weight.set_index(keys=['col_fmt']).reindex(X_train.columns.tolist()).weight3.tolist()
    weight4 =  weight.set_index(keys=['col_fmt']).reindex(X_train.columns.tolist()).weight4.tolist()
    weight5 =  weight.set_index(keys=['col_fmt']).reindex(X_train.columns.tolist()).weight5.tolist()

    return (weight1, weight2, weight3, weight4, weight5)


def weighted_resampling_params(colsample_bytree_weight_lst,colsample_bytree_weight_factor):

    # colsample_bytree_weight needs to be tupple
    colsample_bytree_weight=tuple(colsample_bytree_weight_lst)
    # colsample_bytree_weight_factor need to int and big enough so small float can be represented as int.
    colsample_bytree_weight_factor=colsample_bytree_weight_factor

    sendable_colsample_bytree_weight = sendable_float_to_cpp(colsample_bytree_weight,colsample_bytree_weight_factor)

    #print('\n colsample_bytree_weight', colsample_bytree_weight)
    print('\n colsample_bytree_weight', "min:", min(colsample_bytree_weight),";  max :", max(colsample_bytree_weight))
    print('\n sendable_colsample_bytree_weight', "min:", min(sendable_colsample_bytree_weight),";  max :", max(sendable_colsample_bytree_weight))
    params={
        'booster' : 'gbtree',
        'max_depth' : 10 ,
        'min_child_weight' : 10,
        #'eta' : 0.01,
        'objective' : 'binary:logistic',
        'n_jobs' : 20,
        'silent' : True,
        'eval_metric' : 'logloss',
        'subsample' : 0.8,
        'colsample_bytree' : 0.5,
        'seed': 1001,
        'colsample_bytree_weight' : sendable_colsample_bytree_weight,
        'colsample_bytree_weight_factor' : colsample_bytree_weight_factor,
    }

    return params


def model_iterate(iteration,params,dtrain,dtest,MyCallback, colsample_bytree_weight_factor):
    
    xgb_model=None
    for i in range(iteration):
            print('''
            \n
            ----------------------------------------------------------------------------------
            ------------------------------- model_iteration: %s-------------------------------
            ----------------------------------------------------------------------------------
            \n
            '''%(i))

            model = xgb.train(
                 params=params
                ,dtrain=dtrain
                ,evals=[(dtrain, 'train'), (dtest, 'test')]
                ,num_boost_round=1
                ,callbacks=[MyCallback()]
                ,xgb_model=xgb_model
                )
            
            new_view_weight= find_new_view_importance(gain,current_w)
            new_view_weight_normalized = normalize_dict_values(new_view_weight)
            print ('new_view_weight_normalized: %s'%(new_view_weight_normalized))
            
            
            next_w = w[current_w].copy()
            for view in new_view_weight_normalized:
                next_w = [new_view_weight_normalized[view] if w == view else w for w in next_w]
            
            print("\n current_w first 10 : %s \n" %(w[current_w][:10]))
            print("\n next_w first 10 : %s \n" %(next_w[:10]))
            
            params = weighted_resampling_params(next_w,colsample_bytree_weight_factor)
            
            xgb_model='model.model'
            model.save_model(xgb_model)


    print ("model.get_score_gain : ", model.get_score(importance_type='gain'))
    print( "model.get_fscore     : ", model.get_fscore())
    
    return model
# -

# # Main

# +
pp = pprint.PrettyPrinter()
gain = None
print("xgb.__version__ : ",xgb.__version__)


data_dir= '/home/lpatel/projects/AKI/data_592v'
nrows = 100000
colsample_bytree_weight_factor = 10000


train,test,weight = read_csvs(data_dir,nrows=nrows)
X_train, X_test, dtrain, dtest =  convert_to_dmatix(train,test,weight)
w1, w2, w3, w4, w5 = find_all_weight(weight,X_train)
w = {
    'w1': w1,
    'w2': w2,
    'w3': w3,
    'w4': w4,
    'w5': w5
    }
# -


for current_w in w:
    print("\n current_w : %s \n" %(current_w))
    
    params = weighted_resampling_params(w[current_w],colsample_bytree_weight_factor)
    model = model_iterate(100,params,dtrain,dtest,MyCallback,colsample_bytree_weight_factor)
    
    

    t = datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    df=pd.DataFrame(model.get_score(importance_type='gain'),index=[0])
    df.to_csv("/home/lpatel/aki/results/feature_importance_python_api_%s_%s.csv" % (t,current_w),index=False)
    
    break


