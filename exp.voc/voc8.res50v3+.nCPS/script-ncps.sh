OUTPUT_POSTFIX_FOLDER="voc${labeled_ratio}.res${resnet}v3+.nCPS"
OUTPUT_POSTFIX="${OUTPUT_POSTFIX_FOLDER}/n${num_networks}-cpsw${cps_weight}-t${threshold}-nc${normalising_const}"

export repo_name='TorchSemiSeg-prod'
export volna="/home/ptempczyk/df/TorchSemiSeg/"
export OUTPUT_PATH="${volna}output/${OUTPUT_POSTFIX}"
export snapshot_dir=$OUTPUT_PATH/snapshot
export log_dir=$OUTPUT_PATH/log
export tb_dir=$OUTPUT_PATH/tb

# Training
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py
sleep 5

# Evaluation
export TARGET_DEVICE=$[$NGPUS-1]
export eval_mode="single"
python eval.py -e 0-${nepochs} -d 0-$TARGET_DEVICE #--save_path $OUTPUT_PATH/results
sleep 5

export eval_mode="max_confidence"
python eval.py -e 0-${nepochs} -d 0-$TARGET_DEVICE #--save_path $OUTPUT_PATH/results
sleep 5

# Archive results
mv $OUTPUT_PATH "/home/ubuntu/volume-cold/cps/${OUTPUT_POSTFIX_FOLDER}/"
# TODO: fix symlinks in the moved output
# TODO mkdir -p
