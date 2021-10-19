# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 12345

remoteip = os.popen('pwd').read()
C.volna = os.getenv('volna', '/home/cxk/msra_container/')

"""please config ROOT_dir and user when u first using"""
C.repo_name = os.getenv('repo_name', 'TorchSemiSeg-prod')
C.abs_dir = osp.realpath(".")
C.this_dir = C.abs_dir.split(osp.sep)[-1]


C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]
C.log_dir = os.environ['log_dir']
C.tb_dir = os.environ['tb_dir']

C.log_dir_link = osp.join(C.log_dir, 'log')

# snapshot dir that stores checkpoints
if os.getenv('snapshot_dir'):
    C.snapshot_dir = osp.join(os.environ['snapshot_dir'], "snapshot")
else:
    C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

""" Data Dir and Weight Dir """
C.dataset_path = osp.join(C.volna, 'DATA/city')
C.img_root_folder = C.dataset_path
C.gt_root_folder = C.dataset_path
if os.getenv('resnet'):
    C.resnet = str(os.environ['resnet'])
else:
    C.resnet = '50'
C.pretrained_model = C.volna + f'DATA/pytorch-weight/resnet{C.resnet}_v1c.pth'

""" Path Config """
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir, 'furnace'))

''' Experiments Setting '''
# ratio of labeled set
if os.getenv('labeled_ratio'):
    C.labeled_ratio = int(os.environ['labeled_ratio'])
else:
    C.labeled_ratio = 8
C.train_source = osp.join(C.dataset_path, "config_new/subset_train/train_aug_labeled_1-{}.txt".format(C.labeled_ratio))
C.unsup_source = osp.join(C.dataset_path, "config_new/subset_train/train_aug_unlabeled_1-{}.txt".format(C.labeled_ratio))
C.eval_source = osp.join(C.dataset_path, "config_new/val.txt")
C.test_source = osp.join(C.dataset_path, "config_new/test.txt")
C.demo_source = osp.join(C.dataset_path, "config_new/demo.txt")

C.is_test = False
C.fix_bias = True
C.bn_eps = 1e-5
C.bn_momentum = 0.1

if os.getenv('cps_weight'):
    C.cps_weight = float(os.environ['cps_weight'])
else:
    C.cps_weight = 6

"""Image Config"""
C.num_classes = 19
C.background = -1
C.image_mean = np.array([0.485, 0.456, 0.406])  # 0.485, 0.456, 0.406
C.image_std = np.array([0.229, 0.224, 0.225])
C.image_height = 800
C.image_width = 800
C.num_train_imgs = 2975 // C.labeled_ratio
C.num_eval_imgs = 500
C.num_unsup_imgs = 2975 - C.num_train_imgs

"""Train Config"""
if os.getenv('normalising_const'): # 0=OFF, 1=ON
    C.normalising_const = int(os.environ['normalising_const'])
else:
    C.normalising_const = 0

if os.getenv('num_networks'):
    C.num_networks = int(os.environ['num_networks'])
else:
    C.num_networks = 2
    
if os.getenv('tcps_pass'):
    C.tcps_pass = str(os.environ['tcps_pass'])
else:
    C.tcps_pass = "normal"

if os.getenv('learning_rate'):
    C.lr = float(os.environ['learning_rate'])
else:
    C.lr = 0.04

if os.getenv('batch_size'):
    C.batch_size = int(os.environ['batch_size'])
else:
    C.batch_size = 16

if os.getenv('threshold'):
    C.threshold = float(os.environ['threshold'])
else:
    C.threshold = 0.5

if os.getenv('burnup_step'):
    C.burnup_step = int(os.environ['burnup_step'])
else:
    C.burnup_step = 0

C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 1e-4

if os.getenv('nepochs'):
    C.nepochs = int(os.environ['nepochs'])
else:
    C.nepochs = 137
C.max_samples = max(C.num_train_imgs, C.num_unsup_imgs)     # Define the iterations in an epoch
C.cold_start = 0
C.niters_per_epoch = C.max_samples // C.batch_size

C.num_workers = 8
C.train_scale_array = [0.5, 0.75, 1, 1.5, 1.75, 2.0]

"""Eval Config"""
C.eval_iter = 30
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1, ]  # 0.5, 0.75, 1, 1.5, 1.75
C.eval_flip = False
C.eval_base_size = 800
C.eval_crop_size = 800
C.eval_mode = os.getenv('eval_mode', None)

"""Display Config"""
if os.getenv('snapshot_iter'):
    C.snapshot_iter = int(os.environ['snapshot_iter'])
else:
    C.snapshot_iter = 2
C.record_info_iter = 20
C.display_iter = 50
C.warm_up_epoch = 0

