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

# # print feature importance for each iteration

# +
import xgboost as xgb
import pprint

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter()

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

def MyCallback():
    def callback(env):
        pass
        trees = env.model.get_dump(with_stats=True)
        feature_weight = fmap(trees)
        pp.pprint(trees)
        print(feature_weight)
        print(env.model.get_score(importance_type='gain'))
    return callback

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {'objective':'reg:squarederror', 'eval_metric': 'rmse'}

# -

# # f_target_weight

# +
v1=0.5
v2=0.3
v3=0.2

f_target_weight = [v1/2,v1/2,
                   v2/3,v2/3,v2/3,
                   v3/5,v3/5,v3/5,v3/5,v3/5,
                   ]
sum(f_target_weight)
# -



bst = xgb.train(params, dtrain, num_boost_round=2, evals=[(dtrain, 'train'), (dtest, 'test')],
        callbacks=[MyCallback()])

fig, ax = plt.subplots(figsize=(30, 30))
xgb.plot_tree(bst,ax=ax)

bst.get_score(importance_type='gain')

bst.get_fscore()


