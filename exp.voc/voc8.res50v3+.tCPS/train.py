from __future__ import division
import os.path as osp
import os
import sys
import time
import argparse
import math
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

try:
    from azureml.core import Run
    azure = True
    run = Run.get_context()
except:
    azure = False

parser = argparse.ArgumentParser()

os.environ['MASTER_PORT'] = '169711'

if os.getenv('debug') is not None:
    is_debug = os.environ['debug']
else:
    is_debug = False

def get_mask(pred, THRESHOLD, TCPS_PASS='normal'):
    max_value_per_pixel = nn.functional.softmax(pred, dim=1).max(dim=1)[0]
    if TCPS_PASS == 'lowpass':
        mask = max_value_per_pixel < THRESHOLD
    else:
        mask = max_value_per_pixel > THRESHOLD
    return mask

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True

    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader + unsupervised data loader
    train_loader, train_sampler = get_train_loader(engine, VOC, train_source=config.train_source, \
                                                   unsupervised=False)
    unsupervised_train_loader, unsupervised_train_sampler = get_train_loader(engine, VOC, \
                train_source=config.unsup_source, unsupervised=True)

    if engine.distributed and (engine.local_rank == 0):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        logger = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    # config network and criterion
    IGNORE_INDEX = 255
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
    criterion_csst = nn.MSELoss(reduction='mean')

    if engine.distributed:
        BatchNorm2d = SyncBatchNorm

    # define and init the model
    model = Network(config.num_classes, criterion=criterion,
                    pretrained_model=config.pretrained_model,
                    norm_layer=BatchNorm2d, num_networks=config.num_networks)
    for branch in model.branches:
        init_weight(branch.business_layer, nn.init.kaiming_normal_,
                    BatchNorm2d, config.bn_eps, config.bn_momentum,
                    mode='fan_in', nonlinearity='relu')
    
    # define the learning rate
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr * engine.world_size

    # define the two optimizers

    params_list = []
    for branch in model.branches:        
        params_list = group_weight(params_list, branch.backbone,
                                BatchNorm2d, base_lr)
        for module in branch.business_layer:
            params_list = group_weight(params_list, module, BatchNorm2d,
                                    base_lr)        # head lr * 10

    optimizer = torch.optim.SGD(params_list,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        print('distributed !!')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DataParallelModel(model, device_ids=engine.devices)
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    model.train()
    print('begin train')

    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'

        if is_debug:
            pbar = tqdm(range(10), file=sys.stdout, bar_format=bar_format)
        else:
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)

        dataloader = iter(train_loader)
        unsupervised_dataloader = iter(unsupervised_train_loader)

        sum_loss_sup = 0
        sum_loss_sup_r = 0
        sum_cps = 0
        sum_unsup_passed_percent_l, sum_unsup_passed_percent_r = 0, 0

        THRESHOLD = config.threshold
        BURNUP_STEP = config.burnup_step  # TODO: optimise, check if it messes with logging values (len(pbar))
        THRESHOLDING_TYPE = "cut"  # "zero" or "cut"
        TCPS_PASS = config.tcps_pass

        # TODO: at least two possibilities to proceed here
        # 1) zero all classes in pixels below the threshold (in a single tensor) - "zero"
        # 2) cut all the things below the threshold (in both tensors) - "cut"

        ''' supervised part '''
        for idx in pbar:
            optimizer.zero_grad()
            engine.update_iteration(epoch, idx)
            start_time = time.time()

            minibatch = dataloader.next()
            unsup_minibatch = unsupervised_dataloader.next()
            imgs = minibatch['data']
            gts = minibatch['label']
            unsup_imgs = unsup_minibatch['data']
            imgs = imgs.cuda(non_blocking=True)
            unsup_imgs = unsup_imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)

            b, c, h, w = imgs.shape

            for pair in combinations(range(1, config.num_networks + 1), 2):
                l, r = pair[0], pair[1]
                _, pred_sup_l = model(imgs, step=l)
                _, pred_unsup_l = model(unsup_imgs, step=l)
                _, pred_sup_r = model(imgs, step=r)
                _, pred_unsup_r = model(unsup_imgs, step=r)

                ### unsupervised thresholding ###
                current_idx = epoch * config.niters_per_epoch + idx

                cps_loss_all, loss_sup_all = [], []

                cps_loss = torch.Tensor([0.]).to(device=pred_sup_l.device)
                if current_idx >= BURNUP_STEP:
                    pred_l = torch.cat([pred_sup_l, pred_unsup_l], dim=0)
                    pred_r = torch.cat([pred_sup_r, pred_unsup_r], dim=0)

                    if THRESHOLDING_TYPE == "cut":
                        # valid mask generation for thresholding
                        mask_l = get_mask(pred_l, THRESHOLD, TCPS_PASS)
                        mask_r = get_mask(pred_r, THRESHOLD, TCPS_PASS)
                        
                        # for logging
                        # cps_passed_percent_l = mask_l.sum().float() / mask_l.numel()
                        # dist.all_reduce(cps_passed_percent_l, dist.ReduceOp.SUM)
                        # sum_unsup_passed_percent_l += cps_passed_percent_l / engine.world_size
                        # cps_passed_percent_r = mask_r.sum().float() / mask_r.numel()
                        # dist.all_reduce(cps_passed_percent_r, dist.ReduceOp.SUM)
                        # sum_unsup_passed_percent_r += cps_passed_percent_r / engine.world_size
                    else:
                        raise Exception(f"THRESHOLDING_TYPE={THRESHOLDING_TYPE} not implemented yet.")

                    ### cps loss ###
                    
                    _, max_l = torch.max(pred_l, dim=1)
                    _, max_r = torch.max(pred_r, dim=1)
                    max_l = max_l.long()
                    max_r = max_r.long()

                    if THRESHOLDING_TYPE == "cut":
                        # Fill low confidence pixels with ignored mask 
                    # Fill low confidence pixels with ignored mask 
                        # Fill low confidence pixels with ignored mask 
                        max_r[~mask_r.squeeze()] = IGNORE_INDEX
                        max_l[~mask_l.squeeze()] = IGNORE_INDEX
                        
                    cps_loss = criterion(pred_l, max_r) + criterion(pred_r, max_l)
                    dist.all_reduce(cps_loss, dist.ReduceOp.SUM)
                    cps_loss = cps_loss / engine.world_size
                    cps_loss_all.append(cps_loss)


                ### standard cross entropy loss ###
                loss_sup = criterion(pred_sup_l, gts)
                dist.all_reduce(loss_sup, dist.ReduceOp.SUM)
                loss_sup = loss_sup / engine.world_size

                loss_sup_r = criterion(pred_sup_r, gts)
                dist.all_reduce(loss_sup_r, dist.ReduceOp.SUM)
                loss_sup_r = loss_sup_r / engine.world_size
                
                loss_sup_all.append(loss_sup)
                loss_sup_all.append(loss_sup_r)

            unlabeled_loss = False

            lr = lr_policy.get_lr(current_idx)

            # reset the learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            n = config.num_networks
            normalising_const = 2 / n if config.normalising_const else 1
                
            L_sup = torch.sum(torch.stack(loss_sup_all)) / n
            L_cps =  normalising_const * (torch.sum(torch.stack(cps_loss_all)) / (n - 1))
            loss = L_sup + (config.cps_weight * L_cps)
            
            loss.backward()
            optimizer.step()

            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss_sup=%.2f' % L_sup.item() \
                        + ' loss_cps=%.4f' % L_cps.item()
                        # + ' loss_sup_r=%.2f' % loss_sup_r.item() \

            sum_loss_sup += L_sup.item()
            sum_cps += L_cps.item()
            pbar.set_description(print_str, refresh=False)

            end_time = time.time()


        if engine.distributed and (engine.local_rank == 0):
            logger.add_scalar('train_loss_sup', sum_loss_sup / len(pbar), epoch)
            # logger.add_scalar('train_loss_sup_r', sum_loss_sup_r / len(pbar), epoch)
            logger.add_scalar('train_loss_cps', sum_cps / len(pbar), epoch)

            # if THRESHOLDING_TYPE=='cut':
                ### unsupervised thresholding - for logging ###
                # logger.add_scalar('thresholding/cps_passed_l_percent', sum_unsup_passed_percent_l / len(pbar), epoch)                
                # logger.add_scalar('thresholding/cps_passed_r_percent', sum_unsup_passed_percent_r / len(pbar), epoch)
                # max_value_per_pixel_r_list = [torch.zeros_like(max_value_per_pixel_r) for _ in range(engine.world_size)]
                # dist.all_gather(max_value_per_pixel_r_list, max_value_per_pixel_r)
                # logger.add_histogram('thresholding/max_value_per_pixel_r', torch.stack(max_value_per_pixel_r_list))

        if azure and engine.local_rank == 0:
            run.log(name='Supervised Training Loss', value=sum_loss_sup / len(pbar))
            run.log(name='Supervised Training Loss right', value=sum_loss_sup_r / len(pbar))
            run.log(name='Supervised Training Loss CPS', value=sum_cps / len(pbar))


        if True:  # save all intermediate models
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
