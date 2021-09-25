# OUTPUT_POSTFIX="voc8.res50v3+.tCPS/sup${labeled_ratio}-cpsw${cps_weight}-t${threshold}-b${burnup_step}-$"
OUTPUT_POSTFIX="voc${labeled_ratio}.res${resnet}v3+.nCPS/n${num_networks}-cpsw${cps_weight}-t${threshold}-nc${normalising_const}"

export volna="/home/ptempczyk/df/TorchSemiSeg/"
export OUTPUT_PATH="${volna}output/${OUTPUT_POSTFIX}"
export snapshot_dir=$OUTPUT_PATH/snapshot
export log_dir=$OUTPUT_PATH/log
export tb_dir=$OUTPUT_PATH/tb

python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py
export TARGET_DEVICE=$[$NGPUS-1]
python eval.py -e 0-${nepochs} -d 0-$TARGET_DEVICE #--save_path $OUTPUT_PATH/results
