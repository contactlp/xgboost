set -x
echo "start ----------------------------------------------------------------------------------------------"
#sudo chown -R jenkins.lnx-mi-users /opt/anaconda3
#sudo chmod -R ug+rwX /opt/anaconda3
#sudo chmod -R o+rX /opt/anaconda3
#sudo rsync -rv  /home/xsong/AKI/AKI_MultiView/data/preproc preproc/ && sudo chown -R lpatel /home/lpatel/aki/

ls -rt /home/lpatel/aki/results/cv* |tail -10 && ps -ef|grep python_api

BUILD_DIR=/home/lpatel/projects/repos/xgboost/build 
#sudo yum install boost boost-thread boost-devel
#conda create -n auto_weight python=3
conda_env=auto_weight
env=/opt/anaconda3/envs/$conda_env/bin
export PATH=/opt/anaconda3/envs/$conda_env/bin:$PATH
export PATH=/opt/rh/devtoolset-8/root/usr/bin:$PATH

date
sudo chown -R lpatel ~/projects/repos/xgboost/

mv /home/lpatel/projects/repos/xgboost/lib/libxgboost.so /tmp/ || true
rm /home/lpatel/projects/repos/xgboost/src/common/choice_cummulative || true
#sudo rm -rf $BUILD_DIR && mkdir $BUILD_DIR # clear cache
cd $BUILD_DIR
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
$env/pip freeze > requiremnets.txt
$env/pip install -r requiremnets.txt

#/usr/bin/python3 /home/lpatel/projects/repos/xgboost/z_xgboost_example.py
$env/python3 /home/lpatel/projects/repos/xgboost/z_python_api.py

#ls -lrt /home/lpatel/aki/results/ |tail -10
ls -rt /home/lpatel/aki/results/cv* |tail -10 && ps -ef|grep python_api
#python -m cProfile z_python_api.py &> c_profile_nohup.out &

# nohup python -m cProfile z_python_api.py > nohup2.out 2>&1 &
#  python -m cProfile -o myscript.cprof myscript.py
#nohup python -m cProfile -o nohup_profile.cprof z_python_api.py > nohup_profile.out 2>&1 &

echo "end ----------------------------------------------------------------------------------------------"
