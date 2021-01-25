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
from sklearn.metrics import roc_auc_score,  roc_curve, log_loss
from matplotlib import pyplot
from sklearn.metrics import auc
import seaborn as sns
from sklearn import metrics
import datetime
from sklearn.model_selection import GridSearchCV
from bayes_opt import BayesianOptimization
import warnings
import pprint
import scipy


def find_float_signfigance(float_lst, float_signfigance):

    mi = min(float_lst)
    # print(mi)
    mi = str(mi)
    mi = mi.replace('0.', '')

    leading_zeros = 0
    if 'e' not in mi:
        for ch in mi:
            if ch == '0':
                leading_zeros += 1
            else:
                break
    else:
        # print("e in float")
        mi, leading_zeros = mi.split('e-')
        leading_zeros = int(leading_zeros)
        leading_zeros -= 1

    float_signfigance = float_signfigance + leading_zeros

    return (float_signfigance)


def sendable_float_to_cpp(lst):

    colsample_bytree_weight_factor = find_float_signfigance(
        lst, float_signfigance)
    return tuple([round(n * 10**colsample_bytree_weight_factor) for n in lst])


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


def find_view_weights(w_list,X_train):

    view_weight = {}
    features_per_view = {}

    for feature, weight in zip(X_train.columns.tolist(), w_list):
        view_weight[feature] = weight

        if weight not in features_per_view:
            features_per_view[weight] = [feature]
        else:
            features_per_view[weight].append(feature)

    return (features_per_view, view_weight)


def find_new_view_importance(last_round_feature_weight, current_w,X_train):

    # how many view are there ?
    features_per_view, feature_weight = find_view_weights(w[current_w],X_train)
    unused_views = set(features_per_view)

    # find view which has not been used
    for feature in last_round_feature_weight:
        # print(feature)
        unused_views = unused_views - set([feature_weight[feature]])

    # min weight of last round features
    min_feat_weight = min(last_round_feature_weight.values())

    # give weight if there is unused view
    views_weight = {}
    if len(unused_views) != 0:
        for view in unused_views:
            views_weight[view] = min_feat_weight * 0.5

    # used view weight adding
    for feature in last_round_feature_weight:
        view = feature_weight[feature]
        if view not in views_weight:
            views_weight[view] = last_round_feature_weight[feature]
        else:
            views_weight[view] = views_weight[view] + \
                last_round_feature_weight[feature]
    return views_weight


def normalize_dict_values(d):
    '''
    input :  {0.0032258064516129: 48.354329420666666,
        0.000985221674876847: 0.3671875}
    output : {0.0032258064516129: 0.9924635454064804,
        0.000985221674876847: 0.007536454593519575}
    '''
    output = {}
    total = sum(d.values())
    for i in d:
        output[i] = d[i]/total

    # print("normalize_dict_values : %s" % (output))

    return output


def divide_views_weight_by_number_of_features(d):
    '''
    input  : {0.0032258064516129: 0.9924635454064804,
        0.000985221674876847: 0.007536454593519575}
    output : {0.0032258064516129: 0.0024444914911489666,
        0.000985221674876847: 1.8562696043151663e-05}
    '''

    # TODO: automate this part 0.000985 and 0.003225

    output2 = {}

    for view in d:
        if round(view) == round(0.000985):
            output2[view] = d[view]/186  # 406

        elif round(view) == round(0.003225):
            output2[view] = d[view]/406  # 186
        else:
            # print("\n \n issue in divide_views_weight_by_number_of_features \n \n")
            break
    # TODO: automate this part

    return output2


def normalized_and_divide_views_weight_by_number_of_features(d):

    d = normalize_dict_values(d)
    d = divide_views_weight_by_number_of_features(d)
    return d


def MyCallback():
    def callback(env):
        print('\n------------------starting callback------------------')
        trees = env.model.get_dump(with_stats=True)
        feature_weight = fmap(trees)
        # pp.pprint(trees)

        global gain
        gain = env.model.get_score(importance_type='gain')
        print("\n gain %s" % (gain))
        print('\n------------------ending callback------------------')
    return callback


def read_csvs(data_dir, nrows=None):
    weight_csv = os.path.join(data_dir, 'weight_csv.csv')
    train_csv = os.path.join(data_dir, 'train_csv.csv')
    test_csv = os.path.join(data_dir, 'test_csv.csv')



    if nrows == None:
        train = pd.read_csv(train_csv)
        test = pd.read_csv(test_csv)
    else:
        train = pd.read_csv(train_csv, nrows=nrows)
        test = pd.read_csv(test_csv, nrows=nrows)
    weight = pd.read_csv(weight_csv)

    print("train shape:%s , test shape: %s , weight shape: %s" %
          (train.shape, test.shape, weight.shape))
    # column names are formted inconsitantly
    try:
        weight['col_fmt'] = weight.col.str.replace('-', '.').str.replace(':', '.')
    except Exception as e:
        print(e)

    return (train, test, weight)


def convert_to_dmatix(train, test, weight):
    cols = train.columns.tolist()
    X_col = cols[1:-1]
    y_col = cols[-1]

    X_train, y_train = train[X_col], train[y_col]
    X_test,  y_test = test[X_col], test[y_col]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    print("missing cols X vs Weights : ", set(
        X_col) - set(weight.col_fmt.tolist()))
    print("missing cols Weights vs X : ", set(
        weight.col_fmt.tolist()) - set(X_col))

    return (X_train, X_test, dtrain, dtest, y_train,  y_test)


def find_all_weight(weight, X_train):

    weight1 = weight.set_index(keys=['col_fmt']).reindex(
        X_train.columns.tolist()).weight1.tolist()
    weight2 = weight.set_index(keys=['col_fmt']).reindex(
        X_train.columns.tolist()).weight2.tolist()
    weight3 = weight.set_index(keys=['col_fmt']).reindex(
        X_train.columns.tolist()).weight3.tolist()
    weight4 = weight.set_index(keys=['col_fmt']).reindex(
        X_train.columns.tolist()).weight4.tolist()
    weight5 = weight.set_index(keys=['col_fmt']).reindex(
        X_train.columns.tolist()).weight5.tolist()

    return (weight1, weight2, weight3, weight4, weight5)


def weighted_resampling_params(colsample_bytree_weight_lst, max_depth,
                               min_child_weight, eta, subsample, colsample_bytree):

    # colsample_bytree_weight needs to be tupple
    colsample_bytree_weight = tuple(colsample_bytree_weight_lst)

    sendable_colsample_bytree_weight = sendable_float_to_cpp(
        colsample_bytree_weight)
    colsample_bytree_weight_factor = find_float_signfigance(
        colsample_bytree_weight, float_signfigance)

    # print('\n colsample_bytree_weight', colsample_bytree_weight)
    # print('\n colsample_bytree_weight', "min:", min(
    #     colsample_bytree_weight), ";  max :", max(colsample_bytree_weight), colsample_bytree_weight)
    # print('\n sendable_colsample_bytree_weight', "min:", min(
    #     sendable_colsample_bytree_weight), ";  max :", max(sendable_colsample_bytree_weight), sendable_colsample_bytree_weight)

    params = {
        'booster': 'gbtree',
        'max_depth': int(max_depth),
        'min_child_weight': float(min_child_weight),
        'eta': float(eta),
        'objective': 'binary:logistic',
        'n_jobs': 20,
        'silent': True,
        'eval_metric': 'logloss',
        'subsample': max(min(float(subsample), 1), 0),
        'colsample_bytree': max(min(float(colsample_bytree), 1), 0),
        'seed': 1001,
        'colsample_bytree_weight': sendable_colsample_bytree_weight,
        'colsample_bytree_weight_factor': colsample_bytree_weight_factor,
    }

    return params


def model_iterate(iteration, params, dtrain, dtest, MyCallback,
                  max_depth, min_child_weight, eta, subsample, colsample_bytree,X_train
                  ):
    auc_score_list = []
    xgb_model = None

    for i in range(iteration):
        print('''
            \n
            ----------------------------------------------------------------------------------
            ------------------------------- model_iteration: %s-------------------------------
            ----------------------------------------------------------------------------------
            \n
            ''' % (i))

        model = xgb.train(
            params=params, dtrain=dtrain, evals=[(dtrain, 'train'), (dtest, 'test')], num_boost_round=1,
            callbacks=[MyCallback()], xgb_model=xgb_model
        )
        print('gain : %s' % (gain))
        new_view_weight = find_new_view_importance(gain, current_w,X_train)
        # print('new_view_weight: %s' % (new_view_weight))
        new_view_weight_normalized = normalized_and_divide_views_weight_by_number_of_features(
            new_view_weight)
        # print('new_view_weight_normalized: %s' % (new_view_weight_normalized))

        next_w = w[current_w].copy()
        for view in new_view_weight_normalized:
            next_w = [new_view_weight_normalized[view]
                      if w == view else w for w in next_w]

        # print("\n current_w first 10 : %s ; sum : %s \n " %
        #       (w[current_w][:10], sum(w[current_w])))
        # print("\n next_w first 10 : %s    ; sum : %s :   len : %s \n" %
        #       (next_w[:10], sum(next_w), len(next_w)))

        params = weighted_resampling_params(
            next_w,
            max_depth, min_child_weight, eta, subsample, colsample_bytree)

        # bxgb_model = model.save_raw()  # .decode('utf-8')  # 'model.model'
        # print(type(bxgb_model))
        # xgb_model = [b for b in bxgb_model]
        xgb_model = 'model.model'
        model.save_model(xgb_model)
        # print(xgb_model)
        # break

        score = model.predict(dtest)
        auc = roc_auc_score(y_test, score)
        auc_score_list.append(auc)

        if max(auc_score_list[-early_stopping_n:]) < early_stopping_at * max(auc_score_list):
            break

    # print("model.get_score_gain : ", model.get_score(importance_type='gain'))
    # print("model.get_fscore     : ", model.get_fscore())

    return model, i


def XGB_CV(max_depth,
           # n_estimators,
           colsample_bytree, subsample, min_child_weight, eta
           #, X_train
          ):

    global AUC_LIST
    global LOG_LOSS_LIST
    global ITERbest_LIST
    global PARAM_LIST

    params = weighted_resampling_params(
        w[current_w], max_depth,
        min_child_weight, eta, subsample, colsample_bytree
    )
    PARAM_LIST.append(params)

    model, i = model_iterate(model_iteration, params, dtrain, dtest,
                             MyCallback, max_depth,
                             min_child_weight, eta, subsample, colsample_bytree ,X_train)

    score = model.predict(dtest)
    auc_score = roc_auc_score(y_test, score)
    log_loss_score = log_loss(y_test, score)

#     t = datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
#     df = pd.DataFrame(model.get_score(importance_type='gain'), index=[0])
#     df.to_csv("/home/lpatel/aki/results/feature_importance_python_api_%s_%s.csv" %
#               (t, current_w), index=False)
    # print(n_estimators)

    AUC_LIST.append(auc_score)
    LOG_LOSS_LIST.append(log_loss_score)
    ITERbest_LIST.append(i)

    return (auc_score*2) - 1


# # Main


# +
max_depth, min_child_weight, eta, subsample, colsample_bytree = 10, 10, 0.01, 0.8, 0.5
nrows = 100000 #None  # 100000     # None will load all the data
float_signfigance = 2      # use at least 2 , larger int, better accuracy
model_iteration = 2 #500      # ideal 500
early_stopping_n = 20      # larger int, better accuracy
early_stopping_at = 0.9998  # larger float(ideal 0.9998), better accuracy



pp = pprint.PrettyPrinter()
gain = None
print("xgb.__version__ : ", xgb.__version__)



# +
###################################################################################
## data_592v
###################################################################################
#data_dir = '/home/lpatel/projects/AKI/data_592v'
'''
(auto_weight) [lpatel@paddlefish read_csvs_6182_comment_35]$ mv ../stg2up_2d_full_test.txt  test_csv.csv
(auto_weight) [lpatel@paddlefish read_csvs_6182_comment_35]$ mv ../stg2up_2d_full_train.txt train_csv.csv
(auto_weight) [lpatel@paddlefish read_csvs_6182_comment_35]$ cp ../weight_csv.csv .
'''

data_dir = '/home/lpatel/aki/inputs_6182_comment_21/preproc/read_csvs_6182_comment_35'
train, test, weight = read_csvs(data_dir, nrows=nrows)
X_train, X_test, dtrain, dtest, y_train,  y_test = convert_to_dmatix(
    train, test, weight)
w1, w2, w3, w4, w5 = find_all_weight(weight, X_train)
w = {
    'w1': w1,
    'w2': w2,
    # 'w3': w3,
    # 'w4': w4,
    # 'w5': w5
}
w
# -

test = scipy.io.mmread('/home/lpatel/aki/inputs_6182_comment_21/preproc/read_csvs_6182_comment_35/test_csv.csv')

test

test_df = pd.DataFrame.sparse.from_spmatrix(test,)

test_df.isnull().any().any()

xgb.DMatrix(test)

print(test.getcol(-1).mean())
print(test.getcol(-2).mean())
print(np.unique(test.getcol(-2).data))
print(np.unique(test.getcol(-1).data))





# +
# ####################################################################################
# ### all data (6182_comment_21)
# ####################################################################################
# data_dir = '/home/lpatel/aki/inputs_6182_comment_21/preproc'

# dtrain_path  = os.path.join(data_dir,'stg2up_2d_full_train.txt')
# dtest_path = os.path.join(data_dir,'stg2up_2d_full_test.txt')
# weight_path = os.path.join(data_dir,'weight_csv.csv')
# col_path = os.path.join(data_dir,'stg2up_2d_full_auxCol_svmlite.csv')

# weight = pd.read_csv(weight_path)
# w1 = weight.wt1.tolist()
# w2 = weight.wt2.tolist()
# w3 = weight.wt3.tolist()

# w = {
#     'w1': w1,
#     'w2': w2,
#     'w3': w3,
#     # 'w4': w4,
#     # 'w5': w5
# }

# dtrain = xgb.DMatrix(dtrain_path)
# dtest = xgb.DMatrix(dtest_path)
# X_train = pd.read_csv(col_path) #X_train cols


# # from sklearn.datasets import load_svmlight_file
# # train_data = load_svmlight_file(dtrain_path,zero_based=False)
# # X_train = train_data[0].toarray()
# # y_train = train_data[1]
# -



for current_w in w:

    # current_w = 'w5'

    print("\n current_w : %s \n" % (current_w))

    params = weighted_resampling_params(
        w[current_w], max_depth, min_child_weight, eta, subsample,
        colsample_bytree
    )
    model, _ = model_iterate(model_iteration, params, dtrain, dtest, MyCallback,
                             max_depth, min_child_weight, eta,
                             subsample, colsample_bytree, X_train
                             )

    t = datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    df = pd.DataFrame(model.get_score(importance_type='gain'), index=[0])
    df.to_csv("/home/lpatel/aki/results/feature_importance_python_api_%s_%s.csv" %
              (t, current_w), index=False)

    score = model.predict(dtest)
    auc = roc_auc_score(y_test, score)
    #break

    AUC_LIST = []
    LOG_LOSS_LIST = []
    ITERbest_LIST = []
    PARAM_LIST = []

    #dtrain = xgb.DMatrix(X_train, label=y_train)
    #before making changes : min(LOG_LOSS_LIST): 0.3015150713676214  ; max(AUC_LIST) : 0.5794087861305971
    #After making changes : min(LOG_LOSS_LIST): 0.3015216667348146  ; max(AUC_LIST) : 0.5716324485697313
    #after making changes: min(LOG_LOSS_LIST): 0.30152141156852247  ; max(AUC_LIST) : 0.5794901731598273
    # after making changes: min(LOG_LOSS_LIST): 0.3015126913970709  ; max(AUC_LIST) : 0.5794369585637922

    XGB_BO = BayesianOptimization(XGB_CV, {
        'max_depth': (4, 10),
        #                                      'n_estimators': (1, 10),
        'colsample_bytree': (0.5, 0.9),
        'subsample': (0.5, 0.8),
        'min_child_weight': (1, 10),
        'eta': (0.05, 0.3),
        #'X_train':X_train
    })

    # +
    t = datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    log_file = open('/home/lpatel/aki/results/test.log'+t, 'a')
    log_file.flush()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        XGB_BO.maximize(init_points=10, n_iter=100)

    # +
    df = pd.DataFrame({"auc": AUC_LIST, "log": LOG_LOSS_LIST,
                       "round": ITERbest_LIST, "param": PARAM_LIST})
    df['param'] = df['param'].astype(str)

    t = datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    df.to_csv("/home/lpatel/aki/results/cv_result_baysian_%s_%s.csv" %
              (t, current_w), sep="|")
    # -

    print(len(ITERbest_LIST), len(PARAM_LIST),
          len(LOG_LOSS_LIST), len(AUC_LIST))
    print("min(LOG_LOSS_LIST): %s  ; max(AUC_LIST) : %s" %
          (min(LOG_LOSS_LIST), max(AUC_LIST)))
    # break

    # started at wed Sep 16 1:50 PM


# +
# from *_y use only col y

trainX_path = os.path.join(data_dir,'stg2up_2d_full_trainX.csv')
trainy_path = os.path.join(data_dir,'stg2up_2d_full_trainy.csv')
testX_path  = os.path.join(data_dir,'stg2up_2d_full_testX.csv')
testy_path  = os.path.join(data_dir,'stg2up_2d_full_testy.csv')

trainX = pd.read_csv(trainX_path)
trainy = pd.read_csv(trainy_path)
testX  = pd.read_csv( testX_path)
testy  = pd.read_csv( testy_path)

# use this https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.transpose.html instead of transpose
#trainX.transpose()

print(trainX.shape)
print(trainy.shape)
# -


trainXT = trainX.transpose()

trainXT

trainy.head()

'''
rowid = id
variable =key
val = value 
-----------
long_to_sparse_matrix<-function(df,id,variable,val,binary=FALSE){
  if(binary){
    x_sparse<-with(df,
                   sparseMatrix(i=as.numeric(as.factor(get(id))),
                                j=as.numeric(as.factor(get(variable))),
                                x=1,
                                dimnames=list(levels(as.factor(get(id))),
                                              levels(as.factor(get(variable))))))
  }else{
    x_sparse<-with(df,
                   sparseMatrix(i=as.numeric(as.factor(get(id))),
                                j=as.numeric(as.factor(get(variable))),
                                x=ifelse(is.na(get(val)),1,as.numeric(get(val))),
                                dimnames=list(levels(as.factor(get(id))),
                                              levels(as.factor(get(variable))))))
  }
  
  return(x_sparse)
}

'''

# +
# https://github.com/kumc-bmi/AKI_CDM/blob/master/R/DSGBT_BayesOpt.R#L122
# -

'''
1. transpose test and train with efficient method 
2. make sure columns are in the same order
3. if column does not exist in testing, then add empty col with null value
4. other way arround issue, drop those cols
'''

import pandas as pd
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype

# +
trainX_path = os.path.join(data_dir,'stg2up_2d_full_trainX.csv')
trainy_path = os.path.join(data_dir,'stg2up_2d_full_trainy.csv')
testX_path  = os.path.join(data_dir,'stg2up_2d_full_testX.csv')
testy_path  = os.path.join(data_dir,'stg2up_2d_full_testy.csv')

tr_long = pd.read_csv(trainX_path)
tr_y = pd.read_csv(trainy_path)
ts_long = pd.read_csv(testX_path)
ts_y = pd.read_csv(testy_path)

# +


#dat_wide = dat_long.pivot('ROW_ID','key') #duplicates exist

#take the latest value for each (ROW_ID,key)
tr_long = tr_long.loc[tr_long.groupby(['ROW_ID','key'])['dsa'].idxmin()]
del tr_long['dsa']

#instead of using pivot, we will convert dat_long into sparse Matrix
ROW_ID_c = CategoricalDtype(sorted(tr_long.ROW_ID.unique()), ordered=True)
key_c = CategoricalDtype(sorted(tr_long.key.unique()), ordered=True)

row = tr_long.ROW_ID.astype(ROW_ID_c).cat.codes
col = tr_long.key.astype(key_c).cat.codes
sparse_mt_tr = csr_matrix((tr_long["value"], (row, col)), \
                           shape=(ROW_ID_c.categories.size, key_c.categories.size))

#print out sparse matrix dimension
sparse_mt_tr

#sort tr_y by ROW_ID in sparse_mt_tr
#ref: https://stackoverflow.com/questions/23482668/sorting-by-a-custom-list-in-pandas
tr_y = tr_y[ROW_ID_c,'y']



# +


#take the latest value for each (ROW_ID,key)
ts_long = ts_long.loc[dat_long.groupby(['ROW_ID','key'])['dsa'].idxmin()]

#instead of using pivot, we will convert ts_long into sparse Matrix
ROW_ID_c = CategoricalDtype(sorted(ts_long.ROW_ID.unique()), ordered=True)

row = ts_long.ROW_ID.astype(ROW_ID_c).cat.codes
col = ts_long.key.astype(key_c).cat.codes
sparse_mt_ts = csr_matrix((ts_long["value"], (row, col)), \
                           shape=(ROW_ID_c.categories.size, key_c.categories.size))

#print out sparse matrix dimension
sparse_mt_ts

#sort ts_y by ROW_ID in sparse_mt_ts
ts_y = ts_y[ROW_ID_c,'y']


# -

# As suggested by https://stackoverflow.com/questions/40817459/xgboost-and-sparse-matrix, sparse_mt_tr and tr_y can be directly used in xgboost model.



