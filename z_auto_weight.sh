#conda activate auto_weight
set -x
echo "----------------------------------------------------------------------------------------------"
#conda create -n xgboost_w0 python=3
env=/opt/anaconda3/envs/xgboost_w0/bin
export PATH=/opt/anaconda3/envs/xgboost_w0/bin:$PATH
export PATH=/opt/rh/devtoolset-8/root/usr/bin:$PATH


python3 feature_importance_for_each_iteration.py
