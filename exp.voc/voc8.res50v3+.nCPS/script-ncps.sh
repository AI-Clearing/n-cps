OUTPUT_POSTFIX_FOLDER="voc${labeled_ratio}.res${resnet}v3+.nCPS"
OUTPUT_POSTFIX="${OUTPUT_POSTFIX_FOLDER}/n${num_networks}-cpsw${cps_weight}-t${threshold}-nc${normalising_const}"

export repo_name='ncps'
# ln -s /local_storage_1 /home/dfilipiak/data/
export volna="/home/dfilipiak/projects/ncps/"
# /home/dfilipiak/data/local_storage_1/dfilipiak/ncps/
# ln -s "/home/dfilipiak/projects/ncps/DATA" ${volna}
export OUTPUT_PATH="/local_storage_1/dfilipiak/ncps/output/${OUTPUT_POSTFIX}"
mkdir -p $OUTPUT_PATH

export snapshot_dir=$OUTPUT_PATH/snapshot
export log_dir=$OUTPUT_PATH/log
export tb_dir=$OUTPUT_PATH/tb

EXPORT_PATH="/home/dfilipiak/projects/ncps/output/${OUTPUT_POSTFIX}"
# todo: add copying best and last model
export snapshot_dir_export="${EXPORT_PATH}/snapshot"
export log_dir_export="${EXPORT_PATH}/log"
export tb_dir_export="${EXPORT_PATH}/tb"

mkdir -p ${tb_dir_export} ${log_dir_export} ${snapshot_dir_export}




# Training
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py
# sleep 5

# Evaluation
export TARGET_DEVICE=$[$NGPUS-1]

# TODO: crashes
# export eval_mode="all"
# echo $eval_mode
# python eval.py -e 0-${nepochs} -d 0-$TARGET_DEVICE #--save_path $OUTPUT_PATH/results
# sleep 5

export eval_mode="single"
echo $eval_mode
python eval.py -e 0-${nepochs} -d 0-$TARGET_DEVICE #--save_path $OUTPUT_PATH/results
sleep 5

export eval_mode="max_confidence_softmax"
echo $eval_mode
python eval.py -e 0-${nepochs} -d 0-$TARGET_DEVICE #--save_path $OUTPUT_PATH/results
sleep 5

export eval_mode="soft_voting"
echo $eval_mode
python eval.py -e 0-${nepochs} -d 0-$TARGET_DEVICE #--save_path $OUTPUT_PATH/results
sleep 5

# Archive results

hostname >> "${EXPORT_PATH}/hostname"
rsync -chavzP "${tb_dir}" "${tb_dir_export}/"
rsync -chavzP "${log_dir}" "${log_dir_export}/"
# TODO: fix symlinks in the moved output
# TODO mkdir -p
