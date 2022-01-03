CONDA_HOME="/home/ubuntu/anaconda3"
source $CONDA_HOME/bin/activate semiseg
cd /home/ubuntu/ncps/exp.voc/voc8.res50v3+.tCPS/

export NGPUS=4
export batch_size=8
export learning_rate=0.0025
export snapshot_iter=1

export labeled_ratio=8
export nepochs=34
export burnup_step=0

#############
for CPS_WEIGHT in 1.5 3.0 4.5
do
    export cps_weight=$CPS_WEIGHT
    for THRESHOLD in 0.98 0.96 0.94 0.92 0.90
    do
        export threshold=$THRESHOLD
        ./script-lowpass.sh
    done
done