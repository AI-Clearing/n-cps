#!/usr/bin/env python3
# encoding: utf-8
import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from config import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from seg_opr.metric import hist_info, compute_score
from dataloader import CityScape
from dataloader import ValPre
from network import Network
from tensorboardX import SummaryWriter

try:
    from azureml.core import Run
    azure = True
    run = Run.get_context()
except:
    azure = False

logger = get_logger()

import sys
import torch
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler
from PIL import Image

default_collate_func = dataloader.default_collate

def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]
        

class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        name = data['fn']
        pred = self.sliding_eval(img, config.eval_crop_size, config.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp,
                        'correct': correct_tmp}

        if self.save_path is not None:
            fn = name + '.png'
            cv2.imwrite(os.path.join(self.save_path, fn), pred)
            logger.info('Save the image ' + fn)

        if self.show_image:
            colors = self.dataset.get_class_colors
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results, model_number=None):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iu, mean_IU, _, mean_pixel_acc = compute_score(hist, correct,
                                                       labeled)
        if model_number and model_number.isnumeric():
            step = int(model_number)
            with SummaryWriter(log_dir=config.tb_dir+ '/tb') as tb:
                if config.eval_mode is not None and config.eval_mode != 'single':
                    suffix = f'-{config.eval_mode}'
                else:
                    suffix = ''
                tb.add_scalar(f'test/mIoU{suffix}', np.nanmean(iu) * 100.0, step)
                tb.add_scalar(f'test/mAcc{suffix}', mean_pixel_acc * 100.0, step)

        print(len(dataset.get_class_names()))
        result_line = print_iou(iu, mean_pixel_acc,
                                dataset.get_class_names(), True)
        if azure:
            mean_IU = np.nanmean(iu)*100
            run.log(name='Test/Val-mIoU', value=mean_IU)
        return result_line

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='1', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)
    parser.add_argument('--delete_models', '-m', default=False)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = Network(config.num_classes, criterion=None, norm_layer=nn.BatchNorm2d, num_networks=config.num_networks, resnet_type=f'resnet{config.resnet}')
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}
    val_pre = ValPre()
    dataset = CityScape(data_setting, 'val', val_pre)

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.image_mean,
                                 config.image_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image)
        segmentor.run(config.snapshot_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file, args.delete_models)
