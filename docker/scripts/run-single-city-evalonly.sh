export CPS_HOME=/home/ubuntu/volume/ncps-results
export CPS_HOME_DOCKER=/home/ubuntu/ncps
export DATA_HOME=/home/ubuntu/volume/cps/DATA

# Remember to change script-ncps-for_docker.sh files accordingly (comment out the training)!

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
  --volume ${CPS_HOME}:/home/ubuntu/ncps/output \
  --volume ${DATA_HOME}/pytorch-weight:${CPS_HOME_DOCKER}/DATA/pytorch-weight \
  --volume ${DATA_HOME}/city/config_new:${CPS_HOME_DOCKER}/DATA/city/config_new \
  --volume ${DATA_HOME}/city/generate_colored_gt.py:${CPS_HOME_DOCKER}/DATA/city/generate_colored_gt.py \
  --volume ${DATA_HOME}/city/images:${CPS_HOME_DOCKER}/DATA/city/images \
  --volume ${DATA_HOME}/city/segmentation:${CPS_HOME_DOCKER}/DATA/city/segmentation \
  ncps:0.0.1
  # -it ncps:0.0.1 /bin/bash