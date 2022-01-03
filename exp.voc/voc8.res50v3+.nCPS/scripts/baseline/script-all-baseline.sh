CONDA_HOME="/home/ubuntu/anaconda3"
source $CONDA_HOME/bin/activate semiseg
cd /home/ubuntu/ncps/exp.voc/voc8.res50v3+.tCPS/

export NGPUS=4
export batch_size=8
export learning_rate=0.0025
export snapshot_iter=1

export burnup_step=0
export cps_weight=1.5
export threshold=0.0
export normalising_const=0
export resnet=50

#############

for i in 2 3 4 5 6
do
    export num_networks=$i    
    ./script-ncps.sh
    sleep 5
done
