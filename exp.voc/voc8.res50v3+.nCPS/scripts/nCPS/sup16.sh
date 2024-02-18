CONDA_HOME="/home/dfilipiak/miniconda3"
source $CONDA_HOME/bin/activate semiseg
cd /home/dfilipiak/projects/ncps/exp.voc/voc8.res50v3+.nCPS/

export NGPUS=4
export batch_size=4
export learning_rate=0.00125 
# export NGPUS=8 # 4
# export batch_size=8 # 4
# export learning_rate=0.0025 #0.00125 

export snapshot_iter=1
export burnup_step=0
export cps_weight=1.5
export threshold=0.0
export normalising_const=0

# GPU = 4, r18
# n=2, bs=16 (10GB util, 6.5m/epoch)
# n=4 bs=4 27m/epoch
# n=6, bs=4 (10GB, 40m/epoch)
# n=7 oom
# n=8 oom

# steven ok TITAN V 12 GB
# eval asusgpu2 RTX ok
# sylvester ok
# furnace/tools/gluon2pytorch.py + https://github.com/osmr/imgclsmob/blob/master/gluon/README.md
# srun --time=1-00:00:00 --partition=common --qos=16gpu4d --gres=gpu:8  -w asusgpu2 bash exp.voc/voc8.res50v3+.nCPS/scripts/nCPS/sup16.sh
# srun --time=1-00:00:00 --partition=common --qos=16gpu4d --gres=gpu:4 bash exp.voc/voc8.res50v3+.nCPS/scripts/nCPS/sup16.sh
for r in 18 #50 #101
do
    export resnet=$r
    
    # export labeled_ratio=16d
    # export nepochs=32 # 40 # 32
    # for i in 6 #5 # 5 4 3 2 # 2
    # do
    #     export num_networks=$i    
    #     ./script-ncps.sh
    #     sleep 5
    # done

    # export labeled_ratio=8
    # export nepochs=50  # 34
    # # THIS HAS NOT BEEN SENT to ncps1
    # for i in 4 # done: 2, 3 (untested), 5 (ncp3 progress), 6 (in progress @ ncps2)
    # do
    #     export num_networks=$i    
    #     ./script-ncps.sh
    #     sleep 5
    # done

    # export labeled_ratio=4
    # export nepochs=80 #40
    # # RUN THIS
    # for i in 2 3 # done: 4 5 6 
    # do
    #     export num_networks=$i    
    #     ./script-ncps.sh
    #     sleep 5
    # done

    # dfilipiak_a6000

    # 2 D on 4 GPU
    export labeled_ratio=2
    export nepochs=100 # 60
    for i in 6 # done: 5+ 6(?)
    do
        export num_networks=$i
        ./script-ncps.sh
        sleep 5
    done
done
