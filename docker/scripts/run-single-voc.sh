export CPS_HOME=/home/ubuntu/ncps
export CPS_HOME_DOCKER=/home/ubuntu/ncps
export DATA_HOME=${CPS_HOME}/DATA

docker container run \
  --rm \
  --gpus all \
  --ipc=host \
  --env labeled_ratio \
  --env nepochs \
  --env num_networks \
  --env resnet \
  --env-file $env_file \
  --volume /home/ubuntu/volume-cold/cps/:/home/ubuntu/volume-cold/cps/ \
  --volume ${CPS_HOME}/output:/home/ubuntu/ncps/output \
  --volume ${DATA_HOME}/pytorch-weight:${CPS_HOME_DOCKER}/DATA/pytorch-weight \
  --volume ${DATA_HOME}/pascal_voc/train.txt:${CPS_HOME_DOCKER}/DATA/pascal_voc/train.txt \
  --volume ${DATA_HOME}/pascal_voc/train_aug:${CPS_HOME_DOCKER}/DATA/pascal_voc/train_aug \
  --volume ${DATA_HOME}/pascal_voc/train_aug.txt:${CPS_HOME_DOCKER}/DATA/pascal_voc/train_aug.txt \
  --volume ${DATA_HOME}/pascal_voc/val:${CPS_HOME_DOCKER}/DATA/pascal_voc/val \
  --volume ${DATA_HOME}/pascal_voc/val.txt:${CPS_HOME_DOCKER}/DATA/pascal_voc/val.txt \
  ncps:0.0.1
  # -it ncps:0.0.1 /bin/bash