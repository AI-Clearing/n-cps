CONDA_HOME="/home/ptempczyk/anaconda3"
source $CONDA_HOME/bin/activate semiseg
cd /home/ptempczyk/df/TorchSemiSeg/exp.voc/voc8.res50v3+.tCPS/

export NGPUS=4
export batch_size=8
export learning_rate=0.0025
export snapshot_iter=1
export burnup_step=0
export cps_weight=1.5
export threshold=0.0
export normalising_const=0

export labeled_ratio=2
export nepochs=60

for i in 2 3 4 5 6
do
    export num_networks=$i    
    ./script-ncps.sh
done