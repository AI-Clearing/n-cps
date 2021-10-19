from __future__ import division
import os.path as osp
import os
import sys
import time
import argparse
import math
from typing import List
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from config import config
from dataloader import get_train_loader
from network import Network
from dataloader import CityScape
from utils.init_func import init_weight, group_weight
from engine.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d, bce2d
# from seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d
from tensorboardX import SummaryWriter
from itertools import combinations

try:
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")

azure = False
IGNORE_INDEX = 255
THRESHOLD = config.threshold
TCPS_PASS = config.tcps_pass

parser = argparse.ArgumentParser()

os.environ['MASTER_PORT'] = '169711'

is_debug = os.getenv('debug', 'False').lower() in ('true', '1', 't')

def get_mask(pred, THRESHOLD, TCPS_PASS='normal'):
    max_value_per_pixel = nn.functional.softmax(pred, dim=1).max(dim=1)[0]
    if TCPS_PASS == 'lowpass':
        mask = max_value_per_pixel < THRESHOLD
    else:
        mask = max_value_per_pixel > THRESHOLD
    return mask

def seed_everything(engine):
    """Sets seed in torch and cuda"""
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_tb_logger(engine):
    tb_dir = f'{config.tb_dir}/{time.strftime("%b%d_%d-%H-%M", time.localtime())}'
    generate_tb_dir =  f'{config.tb_dir}/tb'
    logger = SummaryWriter(log_dir=tb_dir)
    engine.link_tb(tb_dir, generate_tb_dir)
    return logger

def build_model(criterion):
    """define and init the model"""
    model = Network(config.num_classes, criterion=criterion, pretrained_model=config.pretrained_model,
                    norm_layer=SyncBatchNorm, num_networks=config.num_networks, resnet_type=f'resnet{config.resnet}')
    for branch in model.branches:
        init_weight(branch.business_layer, nn.init.kaiming_normal_, SyncBatchNorm, config.bn_eps, 
                    config.bn_momentum, mode='fan_in', nonlinearity='relu')
                    
    return model

def get_base_lr(engine):
    """define the learning rate"""
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr
    return base_lr

def build_optimizer(model, base_lr):
    """define the optimizers"""
    params_list = []
    for branch in model.branches:        
        params_list = group_weight(params_list, branch.backbone, SyncBatchNorm, base_lr)
        for module in branch.business_layer:
            params_list = group_weight(params_list, module, SyncBatchNorm, base_lr) # head lr * 10
    optimizer = torch.optim.SGD(params_list,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)
                                
    return optimizer

def get_lr_policy(base_lr):
    """config lr policy"""
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)
    return lr_policy

def distribute_model(engine, model):
    if engine.distributed:
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model)
    else:
        print('Not distributed !!')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DataParallelModel(model, device_ids=engine.devices)
        model.to(device)
    return model

def build_pbar(is_debug):
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar_range = 10 if is_debug else config.niters_per_epoch
    pbar = tqdm(range(pbar_range), file=sys.stdout, bar_format=bar_format)
    return pbar

def get_data(dataloader, unsupervised_dataloader) -> List[torch.Tensor]:
    """Get data from dataloaders"""
    minibatch = dataloader.next()
    unsup_minibatch = unsupervised_dataloader.next()
    imgs = minibatch['data']
    gts = minibatch['label']
    unsup_imgs = unsup_minibatch['data']
    imgs = imgs.cuda(non_blocking=True)
    unsup_imgs = unsup_imgs.cuda(non_blocking=True)
    gts = gts.cuda(non_blocking=True)
    return imgs,gts,unsup_imgs

def forward_all_models(model: nn.Module, imgs: torch.Tensor, unsup_imgs: torch.Tensor) -> List[torch.Tensor]:
    """Performs forward pass of `imgs` and `unsup_imgs` on all models"""
    pred_sup_list, pred_unsup_list = [], []
    for i in range(1, config.num_networks + 1):
        pred_sup_list.append(model(imgs, step=i)[1])
        pred_unsup_list.append(model(unsup_imgs, step=i)[1])
    return pred_sup_list, pred_unsup_list

def calc_cps_loss(engine, criterion: nn.Module, pred_sup_list: List[torch.Tensor], pred_unsup_list: List[torch.Tensor]) -> torch.Tensor:
    """CPS loss calculation"""
    n = config.num_networks
    cps_loss = torch.Tensor([0.]).to(device=pred_sup_list[0].device)
    for pair in combinations(range(n), 2):
        l, r = pair[0], pair[1]
        pred_sup_l = pred_sup_list[l]
        pred_unsup_l = pred_unsup_list[l]
        pred_sup_r = pred_sup_list[r]
        pred_unsup_r = pred_unsup_list[r]

        pred_l = torch.cat([pred_sup_l, pred_unsup_l], dim=0)
        pred_r = torch.cat([pred_sup_r, pred_unsup_r], dim=0)

        ### for cps loss ###                
        max_l = torch.max(pred_l, dim=1)[1].long()
        max_r = torch.max(pred_r, dim=1)[1].long()

        # thresholding
        if THRESHOLD != 0:
            raise Exception('Thresholding is not maintained for now.')
            # mask_l = get_mask(pred_l, THRESHOLD, TCPS_PASS)
            # mask_r = get_mask(pred_r, THRESHOLD, TCPS_PASS)
            # max_l[~mask_l.squeeze()] = IGNORE_INDEX
            # max_r[~mask_r.squeeze()] = IGNORE_INDEX
                
        cps_loss += criterion(pred_l, max_r) + criterion(pred_r, max_l)
        
    dist.all_reduce(cps_loss, dist.ReduceOp.SUM)
    cps_loss = cps_loss / engine.world_size
    
    return cps_loss / (n - 1)

def calc_sup_loss(engine, criterion, gts, pred_sup_list) -> torch.Tensor:
    ### standard cross entropy loss ###
    n = config.num_networks
    
    loss_sup = torch.Tensor([0.]).to(device=gts.device) 
    
    for i in range(n):
        loss_sup += criterion(pred_sup_list[i], gts)
        
    dist.all_reduce(loss_sup, dist.ReduceOp.SUM)
    loss_sup = loss_sup / engine.world_size
    
    return loss_sup  #  * (2/n)

def calc_loss(L_sup: torch.Tensor, L_cps: torch.Tensor) -> torch.Tensor:
    """Calculates the overall loss"""
    n = config.num_networks
    normalising_const = 2 / n if config.normalising_const else 1
    return L_sup + (config.cps_weight * normalising_const * L_cps)

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    cudnn.benchmark = True
    seed_everything(engine)

    # data loader + unsupervised data loader
    train_loader, train_sampler = get_train_loader(engine, CityScape, train_source=config.train_source, unsupervised=False)
    unsupervised_train_loader, unsupervised_train_sampler = get_train_loader(engine, CityScape, train_source=config.unsup_source, unsupervised=True)

    if engine.distributed and (engine.local_rank == 0):
        logger = get_tb_logger(engine)

    # config network and criterion
    pixel_num = 50000 * config.batch_size // engine.world_size
    criterion = ProbOhemCrossEntropy2d(ignore_label=IGNORE_INDEX, thresh=0.7,
                                       min_kept=pixel_num, use_weight=False)
    criterion_cps = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)

    model = build_model(criterion)
    base_lr = get_base_lr(engine)
    optimizer = build_optimizer(model, base_lr)
    lr_policy = get_lr_policy(base_lr)

    model = distribute_model(engine, model)

    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    model.train()
    print('begin train')

    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)

        pbar = build_pbar(is_debug)

        dataloader = iter(train_loader)
        unsupervised_dataloader = iter(unsupervised_train_loader)

        tb_sum_loss_sup, tb_sum_cps = 0, 0
        sum_unsup_passed_percent_l, sum_unsup_passed_percent_r = 0, 0

        ''' main train loop '''
        for idx in pbar:
            current_idx = epoch * config.niters_per_epoch + idx
            optimizer.zero_grad()
            engine.update_iteration(epoch, idx)
            start_time = time.time()

            imgs, gts, unsup_imgs = get_data(dataloader, unsupervised_dataloader)

            pred_sup_list, pred_unsup_list = forward_all_models(model, imgs, unsup_imgs)

            L_cps = calc_cps_loss(engine, criterion_cps, pred_sup_list, pred_unsup_list)
            L_sup = calc_sup_loss(engine, criterion, gts, pred_sup_list)

            # reset the learning rate
            lr = lr_policy.get_lr(current_idx)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            loss = calc_loss(L_sup, L_cps)

            loss.backward()
            optimizer.step()

            tb_sum_loss_sup += L_sup.item()
            tb_sum_cps += L_cps.item()

            print_str = f'Epoch {epoch}/{config.nepochs} Iter {idx+1}/{config.niters_per_epoch}: lr={lr:.2e} loss_sup={L_sup.item():.2f} loss_cps={L_cps.item():.4f}'
            pbar.set_description(print_str, refresh=False)

            end_time = time.time()

        if engine.distributed and (engine.local_rank == 0):
            logger.add_scalar('train_loss_sup', tb_sum_loss_sup / len(pbar), epoch)
            logger.add_scalar('train_loss_cps', tb_sum_cps / len(pbar), epoch)

        # save all intermediate models
        if (engine.distributed and (engine.local_rank == 0)) or not engine.distributed:
            engine.save_and_link_checkpoint(config.snapshot_dir, config.log_dir, config.log_dir_link)
