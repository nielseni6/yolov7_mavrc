import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread
import wandb
import socket

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.datasets import LoadStreams, LoadImages, LoadImagesAndLabels

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model, ModelPGT
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr, non_max_suppression, xyxy2xywh
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss, ComputeLossOTA, ComputePGTLossOTA
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
import sys
from PIL import Image
import torchvision

import plaus_functs
from plaus_functs import get_gradient, get_plaus_score, get_detections, get_labels, get_distance_grids, \
    get_plaus_loss, get_attr_corners
from plaus_functs_original import generate_vanilla_grad, eval_plausibility

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
# import torch
# torch.manual_seed(8)

def train(hyp, opt, device, tb_writer=None):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank, opt.freeze

    print(f'Seed: {opt.seed}')
    
    #################################################################################
    # Set PGT learning rate scheduler
    pgt_coeff_list = []
    pgt_coeff = opt.pgt_coeff
    for k_epoch in range(opt.epochs):
        if ((k_epoch % opt.pgt_lr_decay_step) == 0) and (k_epoch != 0):
            pgt_coeff *= opt.pgt_lr_decay
            # if pgt_coeff == 0.0:
            #     pgt_coeff += opt.pgt_lr_decay_add
        pgt_coeff_list.append(pgt_coeff)
        # print(f'PGT learning rate decayed to {opt.pgt_coeff} at epoch {epoch}')
    #################################################################################
    
    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    is_coco = opt.data.endswith('coco.yaml')

    # Logging- Doing this before checking the dataset. Might update data_dict
    loggers = {'wandb': None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights, map_location=device).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        #####################################################################################
        # if rank in [-1, 0]:
        #     wandb_logger.wandb_run.starting_step = ckpt['epoch'] + 1
        #     wandb_logger.wandb_run.step = wandb_logger.wandb_run.starting_step
        #####################################################################################
        model = ModelPGT(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors'), 
                         iou_thres=opt.iou_thres, classes=opt.classes, 
                         agnostic=opt.agnostic_nms).to(device)
        # model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = ModelPGT(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        # model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']

    # Freeze
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):           
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imc'):
            if hasattr(v.imc, 'implicit'):           
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imb'):
            if hasattr(v.imb, 'implicit'):           
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imo'):
            if hasattr(v.imo, 'implicit'):           
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):           
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'):   
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'):   
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'):  
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'):  
                pg0.append(v.attn.relative_position_bias_table)
        if hasattr(v, 'rbr_dense'):
            if hasattr(v.rbr_dense, 'weight_rbr_origin'):  
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'): 
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):  
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'): 
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):   
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, 'vector'):   
                pg0.append(v.rbr_dense.vector)

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear 
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
        

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '),
                                            k_fold = opt.k_fold, k_fold_num = opt.k_fold_num, k_fold_train = True, 
                                            )
    # dataset_w_labels = LoadImagesAndLabels(train_path, img_size=imgsz, batch_size=batch_size, stride=gs,
    #                                        augment=True, hyp=hyp, cache=opt.cache_images, rect=opt.rect, 
    #                                        image_weights=opt.image_weights, prefix=colorstr('train: '))
    
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        testloader = create_dataloader(test_path, imgsz_test, batch_size * 2, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val: '), k_fold = opt.k_fold, 
                                       k_fold_num = opt.k_fold_num, k_fold_train = False, 
                                       )[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                #plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and rank != -1:
        print('-------------- DDP mode --------------')
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    # scaler = amp.GradScaler(enabled=False)#cuda)
    compute_pgt_loss = ComputePGTLossOTA(model)  # init loss class for plausibility guided training
    compute_loss_ota = ComputeLossOTA(model)  # init loss class
    compute_loss = ComputeLoss(model)  # init loss class
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    torch.save(model, wdir / 'init.pt')
    
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        plaus_score_conf_total = 0.0
        conf_i = 0
        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        num_losses = 5
        mloss = torch.zeros(num_losses, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 11) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'plaus', 'total', 'pscore', 'psavg', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        
        #################################################################################
        # Set PGT learning rate scheduler
        opt.pgt_coeff = pgt_coeff_list[epoch]
        #################################################################################
        
        plaus_loss_total_train, plaus_score_total_train = 0.0, 0.0
        dist_reg_total_train = 0.0
        # num_batches = 1 if (opt.pgt_coeff == 0.0) else 0
        num_batches = 0
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            targets = targets.to(device)
            
            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
            
            # current_memory = torch.cuda.memory_reserved()
            # Forward
            # with amp.autocast(enabled=cuda):
            if True:
                # Use PGT built-into the model 
                use_pgt = ((opt.pgt_coeff != 0.0) or opt.show_plaus_score) and (opt.inherently_explainable)
                # use_pgt = False

                out = model(imgs.requires_grad_(True), pgt = use_pgt, out_nums = opt.out_num_attrs)  # forward

                # Get predicted labels for generating predicted attribution maps, 
                # only needed for loss attributions since they are target specific
                if ((opt.pgt_coeff != 0.0) or opt.show_plaus_score) and opt.loss_attr and opt.pred_targets:
                    ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
                    ema.update(model)
                    det_out, out_ = get_detections(ema.ema, imgs.clone().detach())
                    pred_labels = get_labels(det_out, imgs, targets.clone(), opt)
                else:
                    pred_labels = None # targets
                
                # if use_pgt and (len(out) > 3):#use_pgt:
                if opt.inherently_explainable or (len(out) > 3):
                    pred, attr = out[:3], out[3]#[...,0]
                else:
                    pred = out
                    attr = None
                    # attr = out[0][...,0]
                
                compute_pgt_loss.nl = 3
                compute_loss_ota.nl = 3
                
                if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
                    print('Using loss_ota') if i == 0 else None
                    loss, loss_items = compute_pgt_loss(pred, targets, opt, imgs, attr, pgt_coeff = opt.pgt_coeff, metric=opt.loss_metric, pred_labels=pred_labels)  # loss scaled by batch_size
                    # if (attr is None) and (compute_pgt_loss.attr is not None)
                    if not opt.inherently_explainable:
                        attr = compute_pgt_loss.attr
                else:
                    print('Using loss') if i == 0 else None
                    # This is currently broken due to the addition of plausibility loss
                    loss, loss_items = compute_loss(pred, targets, metric=opt.loss_metric)  # loss scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.
                # model.zero_grad()
                #########################################################
                #            ADDED PLAUSIBILITY LOSS BELOW              #
                #########################################################
                    
                if ((opt.pgt_coeff != 0.0) or opt.show_plaus_score):
                    plaus_loss_total_train += loss_items[3]
                    plaus_score_total_train += compute_pgt_loss.plaus_score #(1 - (loss_items[3] / opt.pgt_coeff))
                num_batches += 1
                
            ##########################################################  
            # Confirm plausibility scores
            if opt.test_plaus_confirm and (attr is not None):
                with torch.no_grad():
                    plaus_score_conf = get_plaus_score(targets_out = targets.to(imgs.device), attr = attr)
                    conf_i += 1 if not math.isnan(plaus_score_conf) else 0
                    plaus_score_conf = plaus_score_conf if not math.isnan(plaus_score_conf) else 0.0
                    plaus_score_conf_total += plaus_score_conf
                    # plaus_score_total_train += plaus_score_conf
                # plaus_loss, (plaus_score, dist_reg, plaus_reg,) = get_plaus_loss(targets, attribution_map, opt)
            ##########################################################

            # Backward
            # scaler.scale(loss).backward() 
            loss.backward()
            t3_pgt = time.time() 

            # Optimize
            if ni % accumulate == 0:
                # scaler.step(optimizer)  # optimizer.step
                optimizer.step()
                # scaler.update()
                
                optimizer.zero_grad()
                # model.zero_grad()
                if ema:
                    ema.update(model)
            
            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * int(num_losses + 2 + 2)) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, compute_pgt_loss.plaus_score, (plaus_score_total_train / num_batches), targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if plots and ni < 10:
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs.detach(), targets, paths, f), daemon=True).start()
                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])  # add model graph
                elif plots and ni == 10 and wandb_logger.wandb:
                    wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                  save_dir.glob('train*.jpg') if x.exists()]})
            
            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------
        
        plaus_score_total_train /= (len(dataloader))
        dist_reg_total_train /= len(dataloader)
        plaus_loss_total_train /= len(dataloader)
        
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1
                test_results = test.test(data_dict,
                                                 batch_size=batch_size * 2,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 verbose=nc < 50 and final_epoch,
                                                 plots=plots and final_epoch,
                                                 wandb_logger=wandb_logger,
                                                 compute_loss=compute_loss,
                                                 is_coco=is_coco,
                                                 v5_metric=opt.v5_metric,
                                                 loss_metric=opt.loss_metric,
                                                 plaus_results = opt.plaus_results)
            if opt.plaus_results:
                results, maps, times, plaus_result = test_results
            else:
                results, maps, times = test_results
                plaus_result = 0.0
            plaus_score_total_test = plaus_result
            plaus_loss_total_test = (1 - plaus_result) * opt.pgt_coeff
            
            
            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * len(results) % results + '\n')  # append metrics, val_loss
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss', 'train/plaus_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2',  # params
                    ] + ['plaus_loss_train', 'plaus_score_train', 'dist_reg_train', 'plaus_score_confirmation', 'pgt_coeff']
            ps_conf_avg = (plaus_score_conf_total / conf_i) if conf_i != 0 else 0.0
            for x, tag in zip(list(mloss[:-1]) + list((results)) + lr + [plaus_loss_total_train, 
                                                                         plaus_score_total_train, dist_reg_total_train, 
                                                                         ps_conf_avg, opt.pgt_coeff,]
                                                                         , tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})  # W&B
            
            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Save model
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if (best_fitness == fi) and (epoch >= 200):
                    torch.save(ckpt, wdir / 'best_{:03d}.pt'.format(epoch))
                if epoch == 0:
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                elif ((epoch+1) % 25) == 0:
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                elif epoch >= (epochs-5):
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(
                            last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt
            
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})
        # Test best.pt
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        if opt.data.endswith('coco.yaml') and nc == 80:  # if COCO
            for m in (last, best) if best.exists() else (last):  # speed, mAP tests
                results, _, _ = test.test(opt.data,
                                          batch_size=batch_size * 2,
                                          imgsz=imgsz_test,
                                          conf_thres=0.001,
                                          iou_thres=0.7,
                                          model=attempt_load(m, device).half(),
                                          single_cls=opt.single_cls,
                                          dataloader=testloader,
                                          save_dir=save_dir,
                                          save_json=True,
                                          plots=False,
                                          is_coco=is_coco,
                                          v5_metric=opt.v5_metric)

        # Strip optimizers 
        final = best if best.exists() else last  # final model 
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload
        if wandb_logger.wandb and not opt.evolve:  # Log the stripped model
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                            name='run_' + wandb_logger.wandb_run.id + '_model',
                                            aliases=['last', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--weights', type=str, default='weights/phase1.pt', help='initial weights path') 
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path') 
    parser.add_argument('--data', type=str, default='data/real_world.yaml', help='data.yaml path') 
    parser.add_argument('--hyp', type=str, default='data/hyp.real_world.yaml', help='hyperparameters path') 
    parser.add_argument('--epochs', type=int, default=300) 
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs') # 16 for coco
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes') 
    parser.add_argument('--rect', action='store_true', help='rectangular training') 
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training') 
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint') 
    parser.add_argument('--notest', action='store_true', help='only test final epoch') 
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check') 
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters') 
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket') 
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training') 
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training') 
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu') 
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%') 
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class') 
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer') 
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode') 
    if socket.gethostname() == 'lambda05':
        parser.add_argument('--local-rank', type=int, default=-1, help='DDP parameter, do not modify') 
    else:
        parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify') 
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers') 
    parser.add_argument('--project', default='runs/pgt/train-pgt-yolov7', help='save to project/name') 
    parser.add_argument('--entity', default=None, help='W&B entity') 
    parser.add_argument('--name', default=f'pgt{socket.gethostname()[-1]}_', help='save to project/name') 
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment') 
    parser.add_argument('--quad', action='store_true', help='quad dataloader') 
    parser.add_argument('--linear-lr', action='store_true', help='linear LR') 
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon') 
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table') 
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B') 
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch') 
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used') 
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone of yolov7=50, first3=0 1 2') 
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation') 
    parser.add_argument('--loss_metric', type=str, default="CIoU",help='metric to minimize: CIoU, NWD') 
    ############################################################################ 
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS') 
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS') 
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3') 
    ############################## PGT Variables ############################### 
    parser.add_argument('--seed', type=int, default=None, help='reproduce results') 
    parser.add_argument('--pgt-coeff', type=float, default=0.5, help='learning rate for plausibility gradient') 
    parser.add_argument('--pgt-lr-decay', type=float, default=0.75, help='learning rate decay for plausibility gradient') 
    parser.add_argument('--pgt-lr-decay-step', type=int, default=25.0, help='learning rate decay step for plausibility gradient') 
    # parser.add_argument('--pgt-lr-decay-add', type=float, default=0.3, help='learning rate decay to add to pgt coeff') 
    parser.add_argument('--n-max-attr-labels', type=int, default=100, help='maximum number of attribution maps generated for each image') 
    parser.add_argument('--out_num_attrs', nargs='+', type=int, default=[2,], help='Output for generating attribution maps (for loss_attr 0: box, 1: obj, 2: cls)') 
    parser.add_argument('--clean_plaus_eval', action='store_true', help='If true, calculate plausibility on clean, non-augmented images and labels') 
    parser.add_argument('--class_specific_attr', action='store_true', help='If true, calculate attribution maps for each class individually') 
    parser.add_argument('--seg-labels', action='store_true', help='If true, calculate plaus score with segmentation maps rather than bbox') 
    parser.add_argument('--seg_size_factor', type=float, default=1.0, help='Factor to reduce weight of segmentation maps that cover entire image') 
    parser.add_argument('--save_hybrid', action='store_true', help='If true, save hybrid attribution maps') 
    parser.add_argument('--plaus_results', action='store_true', help='If true, calculate plausibility on clean, non-augmented images and labels during testing') 
    ################################### PGT Loss Variables ################################### 
    parser.add_argument('--dist_reg_only', type=bool, default=True, help='If true, only calculate distance regularization and not plausibility') 
    parser.add_argument('--focus_coeff', type=float, default=0.3, help='focus_coeff') 
    parser.add_argument('--iou_coeff', type=float, default=0.075, help='iou_coeff') 
    parser.add_argument('--dist_coeff', type=float, default=1.0, help='dist_coeff') 
    parser.add_argument('--bbox_coeff', type=float, default=0.0, help='bbox_coeff') 
    parser.add_argument('--dist_x_bbox', type=bool, default=False, help='If true, zero all distance regularization values to 0 within bbox region') 
    parser.add_argument('--pred_targets', type=bool, default=False, help='If true, use predicted targets for plausibility loss') 
    parser.add_argument('--iou_loss_only', type=bool, default=False, help='If true, only calculate iou loss, no distance regularizers') 
    parser.add_argument('--weighted_loss_attr', type=bool, default=False, help='If true, weight individual loss terms before used to generate attribution maps') 
    ########################################################################################## 
    parser.add_argument('--k_fold', type=int, default=10, help='Number of folds for k-fold cross validation') 
    parser.add_argument('--k_fold_num', type=int, default=1, help='Fold number to use for training') 
    parser.add_argument('--inherently_explainable', type=bool, default=False, help='If true, use inherently explainable model') 
    parser.add_argument('--test_plaus_confirm', type=bool, default=True, help='If true, test plausibility confirmation') 
    parser.add_argument('--lplaus_only', type=bool, default=False, help='If true, only calculate plausibility loss') 
    parser.add_argument('--loss_attr', type=bool, default=True, help='If true, use loss to generate attribution maps') 
    parser.add_argument('--attr_out_indiv', type=bool, default=False, help='If true, calculate plaus_loss for each output head attribution map individually (loss_attr must be False)')
    parser.add_argument('--show_plaus_score', type=bool, default=False, help='If true, show plausibility score, even if not calculating plausibility loss') 
    ########################################################################################## 
    opt = parser.parse_args() 
    print(opt) 
    # opt.plaus_results = True # this is broken 
    # opt.save_hybrid = True 
    # opt.out_num_attrs = [0,1,2,] # unused if opt.loss_attr == True 
    
    opt.data = check_file(opt.data)  # check file 
    opt.no_trace = True 
    opt.save_dir = str('runs/' + opt.name + '_lr' + str(opt.pgt_coeff)) 
    # opt.device = '5' 
    # opt.device = "0,1,2,3" 
    
    # lambda03 Console Commands 
    # source /home/nielseni6/envs/yolo/bin/activate 
    # cd /home/nielseni6/PythonScripts/yolov7_mavrc 

    # nohup python train_pgt.py --device 2 --k_fold_num 2 > ./output_logs/gpu2.log 2>&1 &
    # nohup python train_pgt.py --device 2 > ./output_logs/gpu2.log 2>&1 & 
    # nohup python -m torch.distributed.launch --nproc_per_node 4 --master_port 9528 train_pgt.py --sync-bn > ./output_logs/gpu2360.log 2>&1 & 
    
    # Resume run
    # nohup python train_pgt.py --resume runs/pgt/train-pgt-yolov7/pgt5_1270/weights/last.pt --weights runs/pgt/train-pgt-yolov7/pgt5_1270/weights/last.pt > ./output_logs/gpu5_resume.log 2>&1 & 
    # nohup python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train_pgt.py --sync-bn --resume runs/pgt/train-pgt-yolov7/pgt5_214/weights/last.pt > ./output_logs/gpu1245_coco_pgtlr0_25.log 2>&1 & 
    # opt.resume = "runs/pgt/train-pgt-yolov7/pgt5_1269/weights/last.pt" 
    # opt.weights = 'runs/pgt/train-pgt-yolov7/pgt5_1269/weights/last.pt' 
    
    # opt.dataset = 'coco' 
    opt.dataset = 'real_world_drone' 
    # opt.sync_bn = True 
    
    if opt.seed is None: 
        opt.seed = random.randrange(sys.maxsize) 
    # if opt.seed is None: 
    #     opt.seed = random.randrange(sys.maxsize) 
    rng = random.Random(opt.seed) 
    torch.manual_seed(opt.seed) 
    print(f'Seed: {opt.seed}') 
    
    opt.entity = os.popen('whoami').read().strip() 
    opt.host_name = socket.gethostname() 
    username = os.getenv('USER') 
    os.environ["WANDB_ENTITY"] = username 
    opt.username = username 
    
    # set environment variables for parallel training 
    os.environ["OMP_NUM_THREADS"] = "1" 
    # opt.local_rank = -1 # os.environ["LOCAL_RANK"] 
    
    pretrained = '.pretrained' if opt.weights.endswith('.pt') and os.path.isfile(opt.weights) else ''
    pre = True if opt.weights.endswith('.pt') and os.path.isfile(opt.weights) else False
    if opt.dataset == 'real_world_drone': 
        if ('lambda02' == opt.host_name) or ('lambda03' == opt.host_name) or ('lambda05' == opt.host_name): 
            opt.source = '/data/Koutsoubn8/ijcnn_v7data/Real_world_test/images' 
            opt.data = 'data/real_world.yaml' 
            opt.hyp = f'data/hyp.real_world{pretrained}.yaml' 
        if ('lambda01' == opt.host_name): 
            opt.source = '/data/nielseni6/ijcnn_v7data/Real_world_test/images' 
            opt.data = 'data/real_world_lambda01.yaml' 
            opt.hyp = 'data/hyp.real_world_lambda01.yaml' 
        if opt.k_fold: 
            opt.source = [f'/data/nielseni6/drone_data/k_fold{int(i + (int(i>=opt.k_fold_num)))}/images' for i in range(9)] 
            opt.hyp = f'data/hyp.real_world_kfold{opt.k_fold_num if not pre else pretrained}.yaml' 
            opt.data = f'data/real_world_kfold{opt.k_fold_num}.yaml' 

        # opt.weights = ''
        if opt.inherently_explainable:
            opt.cfg = 'cfg/training/yolov7-tiny-drone-pgt.yaml' 
        else:
            opt.cfg = 'cfg/training/yolov7-tiny-drone.yaml' 
        
    if opt.dataset == 'coco': 
        opt.source = "/data/nielseni6/coco/images" 
        ######### scratch #########
        opt.cfg = 'cfg/training/yolov7.yaml' 
        # opt.weights = '' 
        opt.hyp = 'data/hyp.scratch.p5.yaml' 
        ###########################
        # ######## pretrained #######
        # opt.cfg = 'cfg/training/yolov7.yaml'
        # opt.weights = 'weights/yolov7_training.pt'
        # opt.hyp = 'data/hyp.scratch.custom.yaml'
        # ###########################
        # ####### pretrained-x ######
        # opt.cfg = 'cfg/training/yolov7x.yaml'
        # opt.weights = 'weights/yolov7x_training.pt'
        # opt.hyp = 'data/hyp.scratch.custom.yaml'
        # ###########################
        opt.data = 'data/coco_lambda01.yaml' 
        
        
        # opt.clean_plaus_eval = True 

    opt.data = check_file(opt.data)  # check file 
    
    # print(opt.host_name, opt.entity)
    
    # Set CUDA device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device 
    
    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    # if opt.global_rank in [-1, 0]:
    #     check_git_status()
    #     check_requirements()
    print(opt)
    
    if opt.loss_metric=="NWD":
        print("USING NWD LOSS")
    else:
        print("USING CIOU LOSS")

    print(opt)
    
    
    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run
    
    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        train(hyp, opt, device, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),   # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0),  # segment copy-paste (probability)
                'paste_in': (1, 0.0, 1.0)}    # segment copy-paste (probability)
        
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
                
        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
