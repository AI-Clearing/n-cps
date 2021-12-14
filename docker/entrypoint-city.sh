#!/bin/bash --login
set -e
conda activate $CONDA_ENV
# exec "$@"
# exec /home/ubuntu/ncps/exp.city/city8.res50v3+.nCPS/script-ncps-for_docker.sh && /home/ubuntu/ncps/exp.city/city8.res50v3+.nCPS+CutMix/script-ncps-for_docker.sh
# exec /home/ubuntu/ncps/exp.city/city8.res50v3+.nCPS/script-ncps-for_docker.sh && /home/ubuntu/ncps/exp.city/city8.res50v3+.nCPS+CutMix/script-ncps-for_docker.sh
exec /home/ubuntu/ncps/exp.city/city8.res50v3+.nCPS+CutMix/script-ncps-for_docker.sh