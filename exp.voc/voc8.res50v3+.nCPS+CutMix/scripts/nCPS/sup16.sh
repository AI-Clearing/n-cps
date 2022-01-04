CONDA_HOME="/home/ubuntu/anaconda3"
source $CONDA_HOME/bin/activate semiseg
cd /home/ubuntu/ncps/exp.voc/voc8.res50v3+.nCPS+CutMix/

export NGPUS=8
export batch_size=8
export learning_rate=0.0025
export snapshot_iter=1
export burnup_step=0
export cps_weight=1.5
export threshold=0.0
export normalising_const=0

for r in 50 101
do
    export resnet=$r
    
    export labeled_ratio=16
    export nepochs=32
    for i in 2 3
    do
        export num_networks=$i    
        ./script-ncps.sh
        sleep 5
    done

    export labeled_ratio=8
    export nepochs=34
    for i in 2 3
    do
        export num_networks=$i    
        ./script-ncps.sh
        sleep 5
    done

    export labeled_ratio=4
    export nepochs=40
    for i in 2 3
    do
        export num_networks=$i    
        ./script-ncps.sh
        sleep 5
    done

    export labeled_ratio=2
    export nepochs=60
    for i in 2 3
    do
        export num_networks=$i    
        ./script-ncps.sh
        sleep 5
    done
done