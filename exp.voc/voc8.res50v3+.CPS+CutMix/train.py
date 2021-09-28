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
from dataloader import VOC
from utils.init_func import init_weight, group_weight
from engine.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
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

is_debug = os.getenv('debug', False)


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

def build_model(criterion): #?
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
        base_lr = config.lr * engine.world_size
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

def get_data(dataloader, unsupervised_dataloader_0, unsupervised_dataloader_1):
    """Get data from dataloaders"""
    minibatch = dataloader.next()
    unsup_minibatch_0 = unsupervised_dataloader_0.next()
    unsup_minibatch_1 = unsupervised_dataloader_1.next()

    imgs = minibatch['data'].cuda(non_blocking=True)
    gts = minibatch['label'].cuda(non_blocking=True)
    unsup_imgs_0 = unsup_minibatch_0['data'].cuda(non_blocking=True)
    unsup_imgs_1 = unsup_minibatch_1['data'].cuda(non_blocking=True)
    mask_params = unsup_minibatch_0['mask_params'].cuda(non_blocking=True)

    return imgs,gts,unsup_imgs_0,unsup_imgs_1,mask_params

def forward_all_models(model: nn.Module, imgs: torch.Tensor, unsup_imgs: torch.Tensor) -> List[torch.Tensor]:
    """Performs forward pass of `imgs` and `unsup_imgs` on all models"""
    pred_sup_list, pred_unsup_list = [], []
    for i in range(1, config.num_networks + 1):
        pred_sup_list.append(model(imgs, step=i)[1])
        pred_unsup_list.append(model(unsup_imgs, step=i)[1])
    return pred_sup_list, pred_unsup_list

def calc_cps_loss(engine, criterion: nn.Module, pseudo_max_list: List[torch.Tensor], unsup_imgs_mixed_list: List[torch.Tensor]) -> torch.Tensor:
    """CPS loss calculation"""
    pseudo_max_l, pseudo_max_r = pseudo_max_list[0], pseudo_max_list[1]
    n = config.num_networks
    cps_loss = torch.Tensor([0.]).to(device=unsup_imgs_mixed_list[0].device)
    for pair in combinations(range(n), 2):
        l, r = pair[0], pair[1]
        pred_l = unsup_imgs_mixed_list[l]
        pred_r = unsup_imgs_mixed_list[r]

        # thresholding
        mask_l = get_mask(pred_l, THRESHOLD, TCPS_PASS)
        mask_r = get_mask(pred_r, THRESHOLD, TCPS_PASS)
        pseudo_max_r[~mask_r.squeeze()] = IGNORE_INDEX
        pseudo_max_l[~mask_l.squeeze()] = IGNORE_INDEX

        cps_loss += criterion(pred_l, pseudo_max_r) + criterion(pred_r, pseudo_max_l)
    
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


'''
For CutMix.
'''
import mask_gen
from custom_collate import SegCollate
mask_generator = mask_gen.BoxMaskGenerator(prop_range=config.cutmix_mask_prop_range, n_boxes=config.cutmix_boxmask_n_boxes,
                                           random_aspect_ratio=not config.cutmix_boxmask_fixed_aspect_ratio,
                                           prop_by_area=not config.cutmix_boxmask_by_size, within_bounds=not config.cutmix_boxmask_outside_bounds,
                                           invert=not config.cutmix_boxmask_no_invert)

add_mask_params_to_batch = mask_gen.AddMaskParamsToBatch(
    mask_generator
)
collate_fn = SegCollate()
mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)



def get_data_cutmix(model, unsup_imgs_0, unsup_imgs_1, mask_params):
    ''' CutMix data augmentation '''
    batch_mix_masks = mask_params
    unsup_imgs_mixed = unsup_imgs_0 * (1 - batch_mix_masks) + unsup_imgs_1 * batch_mix_masks
    with torch.no_grad():
        # Estimate the pseudo-label with branch#1 & supervise branch#2
        _, logits_u0_tea_1 = model(unsup_imgs_0, step=1)
        _, logits_u1_tea_1 = model(unsup_imgs_1, step=1)
        logits_u0_tea_1 = logits_u0_tea_1.detach()
        logits_u1_tea_1 = logits_u1_tea_1.detach()
        # Estimate the pseudo-label with branch#2 & supervise branch#1
        _, logits_u0_tea_2 = model(unsup_imgs_0, step=2)
        _, logits_u1_tea_2 = model(unsup_imgs_1, step=2)
        logits_u0_tea_2 = logits_u0_tea_2.detach()
        logits_u1_tea_2 = logits_u1_tea_2.detach()

    # Mix teacher predictions using same mask
    # It makes no difference whether we do this with logits or probabilities as
    # the mask pixels are either 1 or 0
    logits_cons_tea_1 = logits_u0_tea_1 * (1 - batch_mix_masks) + logits_u1_tea_1 * batch_mix_masks
    pseudo_max_l = torch.max(logits_cons_tea_1, dim=1)[1].long()
    logits_cons_tea_2 = logits_u0_tea_2 * (1 - batch_mix_masks) + logits_u1_tea_2 * batch_mix_masks
    pseudo_max_r = torch.max(logits_cons_tea_2, dim=1)[1].long()
    return unsup_imgs_mixed, pseudo_max_l, pseudo_max_r

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    cudnn.benchmark = True
    seed_everything(engine)

    # data loader + unsupervised data loader
    train_loader, train_sampler = get_train_loader(engine, VOC, train_source=config.train_source, unsupervised=False, collate_fn=collate_fn)
    unsupervised_train_loader_0, unsupervised_train_sampler_0 = get_train_loader(engine, VOC, train_source=config.unsup_source, unsupervised=True, collate_fn=mask_collate_fn)
    unsupervised_train_loader_1, unsupervised_train_sampler_1 = get_train_loader(engine, VOC, train_source=config.unsup_source, unsupervised=True, collate_fn=collate_fn)

    if engine.distributed and (engine.local_rank == 0):
        logger = get_tb_logger(engine)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
    criterion_csst = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)

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
        unsupervised_dataloader_0 = iter(unsupervised_train_loader_0)
        unsupervised_dataloader_1 = iter(unsupervised_train_loader_1)

        tb_sum_loss_sup, tb_sum_cps = 0, 0
        ''' main train loop '''
        for idx in pbar:
            current_idx = epoch * config.niters_per_epoch + idx
            optimizer.zero_grad()
            engine.update_iteration(epoch, idx)
            start_time = time.time()

            imgs, gts, unsup_imgs_0, unsup_imgs_1, mask_params = get_data(dataloader, unsupervised_dataloader_0, unsupervised_dataloader_1)
            unsup_imgs_mixed, pseudo_max_list = get_data_cutmix(model, unsup_imgs_0, unsup_imgs_1, mask_params)

            pred_sup_list, pred_unsup_mixed_list = forward_all_models(model, imgs, unsup_imgs_mixed)

            L_cps = calc_cps_loss(engine, criterion, pseudo_max_list ,pred_unsup_mixed_list)
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

