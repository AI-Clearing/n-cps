# Installation
There are two ways to run the code -  with conda or with Docker. 
The first option is recommended for most users.
The second one is experimental and is not recommended.
The code is developed and tested using 4 or 8 (required for Cityscapes) Tesla V100 GPUs.

## (1) Data Preparation
##### Download the data (VOC, Cityscapes) and pre-trained models from  [OneDrive link](https://pkueducn-my.sharepoint.com/:f:/g/personal/pkucxk_pku_edu_cn/EtjNKU0oVMhPkOKf9HTPlVsBIHYbACel6LSvcUeP4MXWVg?e=139icd) provided by the CPS paper authors: 

```
DATA/
|-- city
|-- pascal_voc
|-- pytorch-weight
|   |-- resnet50_v1c.pth
|   |-- resnet101_v1c.pth
```

You'll need two directories - one (preferably on SSD) for the repo, data and current calculations (`ncps`) and one for cold results storage (`volume-cold`), where all the models will be moved after the training.
For the purpose of this guide, we assume that both directories are contained in `/home/ubuntu/`.
We will also assume that `/home/ubuntu/` is your home directory containing this repository (`ncps`).
Replace this with your own path and directories.

## (2) Running the code

There are two ways to run the code - without or without Docker. 
The first option is recommended for most users.
The code is developed and tested using 4 (some VOC models) or 8 (VOC/Cityscapes) Tesla V100 GPUs.
For Cityscapes and ResNet-101, GPUs with more memory will be needed (perhaps A100 will be sufficient, but it was not tested).

### Running without Docker

**(1) Create a conda environment:**

This requires Anaconda and was tested on Ubuntu 20.04.
This tutorial assumes that your user is `ubuntu` and the repository (`ncps`) is stored in `/home/ubuntu`.
First, create the conda environment:

```bash
conda env create -f semiseg.yaml
conda activate semiseg
```

**(2) Install Apex (needs CUDA)**

```bash
cd ./furnace/apex
python setup.py install --cpp_ext --cuda_ext
```

**(3) Run the training**

Then, you'll have to adjust scripts in (for example) `exp.voc/voc8.res50v3+.nCPS+CutMix` (Pascal VOC on nCPS with CutMix) to your own paths.

First, you need to adjust `script-ncps.sh`.
Edit `repo_name` and `volna` to provide your own repository name and repository path.
In lines 30 and 31, you will need to adjust `volume-cold` location.

Then, you'll have to adjust `scripts/nCPS/sup16.sh`.
The most important variables are `NGPUS` (numbers of GPUs available).
The code was not developed for single GPU training.
Make sure you also set the correct paths in the first three lines (`CONDA_HOME` for Anaconda and the experiment path).

After that, just run `exp.voc/voc8.res50v3+.nCPS+CutMix/scripts/nCPS/sup16.sh`.
It contains the training/evaluation loop for multiple hyper-parameters.

Please note that, for fair comparison, we control the total iterations during training in each experiment similar (almost the same), including the supervised baseline and semi-supervised methods. Therefore, the nepochs for different partitions are different. We list the nepochs for different datasets and partitions in the below.

| Dataset    | 1/16 | 1/8  | 1/4  | 1/2  |
| ---------- | ---- | ---- | ---- | ---- |
| VOC        | 32   | 34   | 40   | 60   |
| Cityscapes | 128  | 137  | 160  | 240  |


### Running it with Docker (experimental)

There are some potential issues with this mode and it is not recommended now.
It is assumed that you have Docker installed and configured to work with GPUs.
Before running, check all the paths in the `docker` directory.

**(1) Entrypoint**

In `Dockerfile`, you have to set the entrypoint in line 74.  It can be either `entrypoint-voc.sh` for Pascal VOC or `entrypoint-city.sh` for Cityscapes.
We will assume using VOC in this guide:
```dockerfile
COPY docker/entrypoint-voc.sh /usr/local/bin/entrypoint.sh
```
In `entrypoint-voc.sh`, you have to decide whether you want to run the experiment with or without CutMix.
```dockerfile
exec /home/ubuntu/ncps/exp.voc/voc8.res50v3+.nCPS/script-ncps-for_docker.sh
# exec /home/ubuntu/ncps/exp.voc/voc8.res50v3+.nCPS+CutMix/script-ncps-for_docker.sh
```
Uncomment only one option.

**(2) Shell scripts**

In `docker/scripts` directory you will find the main experiment loops (`run-voc.sh` and `run-city.sh`).
Both these are prepared for running the code for 1/16, 1/8, 1/4 and 1/2 of the training data with ResNet-50 and ResNet-101 (so, 16 training procedures will be run for both of these).
If you want to shorten this process, please comment out the lines that are not needed.

You also have to change paths in `run-single-voc.sh` (or `run-single-city.sh`) to your own paths: `CPS_HOME` (repo directory) and `volume-cold` location (14th line).

**(3) Env variables**

`docker/scripts/voc.env` (or `city.env`) is a file containing the environment variables for the experiment.
You will need to adjust the number of GPUs (`NGPUS`).
The project was not tested with one GPU.

To run the image (needs `"default-runtime": "nvidia"` on `/etc/docker/daemon.json` due to Apex compilation during the build!)
```bash
cd docker
docker image build --file Dockerfile --tag ncps:0.0.1 ../ && scripts/run-voc.sh
```
The first build might take 30 minutes or so.
