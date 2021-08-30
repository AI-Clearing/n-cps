# CONDA_HOME="/home/ptempczyk/anaconda3"
# source $CONDA_HOME/bin/activate semiseg

# export NGPUS=4
# export batch_size=8
# export learning_rate=0.0025
# export snapshot_iter=1

# export labeled_ratio=8
# export nepochs=34
# export threshold=0.75
# export burnup_step=0
# export cps_weight=1.5

OUTPUT_POSTFIX="voc8.res50v3+.tCPS/sup${labeled_ratio}-cpsw${cps_weight}-t${threshold}-b${burnup_step}"

export volna="/home/ptempczyk/df/TorchSemiSeg/"
export OUTPUT_PATH="${volna}output/${OUTPUT_POSTFIX}"
export snapshot_dir=$OUTPUT_PATH/snapshot
export log_dir=$OUTPUT_PATH/log
export tb_dir=$OUTPUT_PATH/tb

python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py
export TARGET_DEVICE=$[$NGPUS-1]
python eval.py -e 0-${nepochs} -d 0-$TARGET_DEVICE #--save_path $OUTPUT_PATH/results

# following is the command for debug
# export NGPUS=1
# export batch_size=2
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --debug 1
