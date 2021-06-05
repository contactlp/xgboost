set -x
username='lpatel'
echo "----------------------------------------------------------------------------------------------"
#conda create -n xgboost_env python=3
export PATH=/opt/anaconda3/envs/xgboost_env/bin:$PATH
export PATH=/opt/rh/devtoolset-8/root/usr/bin:$PATH

date
sudo chown -R $username ~/projects/repos/xgboost/

#sudo rm -rf /home/$username/projects/repos/xgboost/build # clear cache
cd /home/$username/projects/repos/xgboost/build
cmake3 .. 
# cmake .. 
make -j 16 

#python part, change this below part for R
#exit 0
cd /home/$username/projects/repos/xgboost/python-package 
# sudo /usr/bin/pip3 uninstall -y xgboost   || true
# sudo /usr/bin/python3 setup.py install #--no-cache-dir
env=/opt/anaconda3/envs/xgboost_env/bin

sudo $env/pip uninstall -y xgboost   || true
$env/python3 setup.py install #--no-cache-dir
$env/pip freeze > requiremnets.txt

#/usr/bin/python3 /home/$username/projects/repos/xgboost/z_xgboost_example.py
$env/python3 /home/$username/projects/repos/xgboost/z_xgboost_aki_tesing.py
