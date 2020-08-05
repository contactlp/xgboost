set -x
echo "----------------------------------------------------------------------------------------------"
#conda create -n auto_weight python=3
conda_env=auto_weight
env=/opt/anaconda3/envs/$conda_env/bin
export PATH=/opt/anaconda3/envs/$conda_env/bin:$PATH
export PATH=/opt/rh/devtoolset-8/root/usr/bin:$PATH

date
sudo chown -R lpatel ~/projects/repos/xgboost/

#sudo rm -rf /home/lpatel/projects/repos/xgboost/build # clear cache
cd /home/lpatel/projects/repos/xgboost/build
cmake3 .. 
# cmake .. 
make -j 16 

#exit 0
cd /home/lpatel/projects/repos/xgboost/python-package 
# sudo /usr/bin/pip3 uninstall -y xgboost   || true
# sudo /usr/bin/python3 setup.py install #--no-cache-dir

pip uninstall -y xgboost || true
sudo $env/pip uninstall -y xgboost   || true
$env/python3 setup.py install #--no-cache-dir
#$env/pip freeze > requiremnets.txt
$env/pip install -r requiremnets.txt

#/usr/bin/python3 /home/lpatel/projects/repos/xgboost/z_xgboost_example.py
$env/python3 /home/lpatel/projects/repos/xgboost/z_python_api.py
