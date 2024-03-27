import argparse
import json
import os
from pathlib import Path
from threading import Thread
import numpy as np
import torch
import yaml
from tqdm import tqdm
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel

from xai.Perturbation import Perturbation
from plaus_functs import get_gradient, get_gaussian, get_plaus_score
import socket
from plot_functs import imshow

from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
import wandb
from models.yolo import Model, ModelPGT
from utils.loss import ComputeLoss, ComputeLossOTA, ComputePGTLossOTA
# from torchattacks import PGD, FGSM
from xai.Attacks import PGD, FGSM

# parser = argparse.ArgumentParser(prog='test_pgt.py')
# parser.add_argument('--debug', action='store_true', help='debug mode for visualizing figures')
# import logging

# logger = logging.getLogger(__name__)

def test_pgt(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=False, ############ TURNED OFF DUE TO ERROR SHOULD FIX ############
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         trace=False,
         is_coco=False,
         v5_metric=False,
         loss_metric="CIoU",
         device=None,
         opt=None):
    opt.plots = plots
    opt.loss_metric = loss_metric
    opt.save_txt = save_txt
    # Initialize/load model and set device
    training = model is not None
    opt.training = training
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        if device is None:  # set device
            device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        opt.save_dir = save_dir
        
        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size
        
        if compute_loss is not None:
            compute_loss = compute_loss(model)
        opt.compute_loss = compute_loss
        
        if trace:
            model = TracedModel(model, device, imgsz)
    opt.imgsz = imgsz
    opt.device = device
    
    # Half
    half = False
    # half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()
    opt.half = half
    
    # Configure
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    opt.nc = nc
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    opt.iouv = iouv
    niou = iouv.numel()
    opt.niou = niou
    opt.is_coco = is_coco
    
    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    opt.wandb_logger = wandb_logger
    opt.log_imgs = log_imgs
    opt.save_json = save_json
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    if v5_metric:
        print("Testing with YOLOv5 AP metric...")
    
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    opt.confusion_matrix = confusion_matrix
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    opt.names = names
    coco91class = coco80_to_coco91_class() 
    opt.coco91class = coco91class 
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    
    ##########################################################################################
    # evalattai = Perturbation(model, opt, nsteps = 10, epsilon = 0.05)
    # evalattai.__init_attr__(attr_method = get_gradient, norm=True, keepmean=True, absolute=False, grayscale=False)
    
    start=0
    if opt.eval_type == 'robust' or opt.eval_type == 'robust2':
        nsteps = 2
        start=1
        # desired_snr = 10.0
    elif opt.eval_type == 'evalattai':
        nsteps = 10
        # desired_snr = 5.0
    elif opt.eval_type == 'default':
        nsteps = 1
        # desired_snr = 1e+100
    elif opt.eval_type == 'robust_snr_vary':
        nsteps = 10
        # desired_snr = 1e+100
        
    
    if opt.atk == 'none':
        desired_snr = 1e+100
    else:
        desired_snr = opt.desired_snr
    
    torchattacks_used = (opt.atk == 'pgd') or (opt.atk == 'fgsm')
    robust_eval = Perturbation(model, opt, nsteps = nsteps, desired_snr = desired_snr, start=start, torchattacks_used=torchattacks_used)
    
    if opt.atk == 'grad':
        robust_eval.__init_attr__(attr_method = get_gradient, norm=False, keepmean=False, absolute=False, grayscale=False)
    if opt.atk == 'gaussian':
        robust_eval.__init_attr__(attr_method = get_gaussian, norm=True, keepmean=True, absolute=False, grayscale=False)
    if opt.atk == 'pgd':
        attr_method = PGD(model, loss=compute_loss, metric=loss_metric, eps=4/255, alpha=1/255, steps=4)
        robust_eval.__init_attr__(attr_method = attr_method, torchattacks_used=True)
    if opt.atk == 'fgsm':
        attr_method = FGSM(model, loss=compute_loss, metric=loss_metric, eps=4/255)
        robust_eval.__init_attr__(attr_method = attr_method, torchattacks_used=True)
    if opt.atk == 'none':
        robust_eval.__init_attr__(attr_method = None, norm=True, keepmean=True, absolute=False, grayscale=False)
    ##########################################################################################
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        
        debug = False
        if debug:
            from plot_functs import imshow
            for i_num in range(len(img)):
                imshow(img[i_num].float(), save_path='figs/img')
                
        img = img.half() if half else img.float()  # uint8 to fp16/32 
        img /= 255.0  # 0 - 255 to 0.0 - 1.0 
        targets = targets.to(device) 
        img_, targets_ = img.clone().detach(), targets.clone().detach() 
        nb, _, height, width = img.shape  # batch size, channels, height, width

        ##########################################################################################
        # evalattai.collect_stats(img_, targets_, paths, shapes)
        robust_eval.collect_stats(img_, targets_, paths, shapes, batch_i)
        ##########################################################################################

############################################################################################
    ##########################################################################################
    # evalattai_results = evalattai.compute_stats()
    robust_eval_results, stats_all = robust_eval.compute_stats()
    for ir in range(len(robust_eval_results)):
        r_loss = torch.zeros(3, device=device)
        ((r_mp, r_mr, r_map50, r_map, r_loss[0], r_loss[1], r_loss[2], r_plaus), r_maps, r_t) = robust_eval_results[ir][0]
        robust_eval_results[ir][0] = (r_mp, r_mr, r_map50, r_map, *(r_loss.cpu()).tolist(), r_plaus), r_maps, r_t
    if opt.eval_type == 'robust_snr_vary':
        return robust_eval_results, robust_eval.snr_list
    else:
        return robust_eval_results
    ##########################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/r_test', help='save to project/name')
    parser.add_argument('--name', default=f'test{socket.gethostname()[-1]}_', help='save to project/name') 
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment') 
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model') 
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation') 
    parser.add_argument('--half-precision', action='store_true', help='use half precision') 
    ############################################################################ 
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B') 
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used') 
    ############################################################################ 
    parser.add_argument('--dataset', default='real_world_drone', help='coco or real_world_drone') 
    # parser.add_argument('--hyp', type=str, default='data/hyp.coco.yaml', help='') 
    # parser.add_argument('--atk', type=str, default='gaussian', help='grad, pgd, gaussian') 
    parser.add_argument('--eval_type', type=str, default='robust', help='robust, evalattai, default') 
    parser.add_argument('--desired_snr', type=float, default=40.0, help='desired snr') 
    parser.add_argument('--atk_list', nargs='+', type=str, default=['none', 'gaussian', 'fgsm', 'pgd',], help='atk list') 
    parser.add_argument('--weights_dir', type=str, default='weights/eval_coco', help='models folder') 
    parser.add_argument('--entire_folder', action='store_true', help='entire folder') 
    # parser.add_argument('--allow_val_change', type=bool, default=True, help='allow val change') 
    # parser.add_argument('--debug', action='store_true', help='debug mode for visualizing figures') 
    parser.add_argument('--loss_attr', action='store_true', help='loss attr') 
    parser.add_argument('--out_num_attrs', nargs='+', type=int, default=[2,], help='Default output for generating attribution maps')
    opt = parser.parse_args() 

    opt.entire_folder = True 
    opt.loss_attr = True 
    # opt.weights_dir = 'weights/pgt_runs_best' 
    # opt.weights_dir = 'weights/pgt_runs' 
    # opt.weights_dir = 'weights/pgt_runs2' 
    opt.weights_dir = 'weights/baselines_kfold' 
    # check_requirements() 
    
    # opt.eval_type = 'default' 
    # opt.atk_list = ['none',] 
    
    # opt.eval_type = 'robust' 
    # # opt.atk_list = ['none', 'gaussian', 'pgd', 'fgsm'] # Evaluate adversarial robustness 
    # opt.atk_list = ['none', 'pgd', 'fgsm'] # Evaluate adversarial robustness 

    # opt.eval_type = 'robust2' 
    # opt.atk_list = ['none', 'grad'] # 'pgd', 'fgsm' 
    
    opt.eval_type = 'robust_snr_vary'
    opt.atk_list = ['gaussian']
    # opt.atk_list = ['pgd']
    # # opt.atk_list = ['fgsm']
    
    atk_list = opt.atk_list 
    
    opt.atk = '' 
    for atkname in atk_list: 
        opt.atk = f'{opt.atk}{atkname}' 
    
    opt.entity = os.popen('whoami').read().strip() 
    opt.host_name = socket.gethostname() 
    username = os.getenv('USER') 
    os.environ["WANDB_ENTITY"] = username 
    opt.username = username 
    
    opt.half_precision = True 
    # opt.device = '4' 
    device_num = opt.device 
    
    
    if opt.dataset == 'real_world_drone':
        if ('lambda02' == opt.host_name) or ('lambda03' == opt.host_name) or ('lambda05' == opt.host_name):    
            opt.source = '/data/Koutsoubn8/ijcnn_v7data/Real_world_test/images' 
            opt.data = 'data/real_world.yaml' 
            opt.hyp = 'data/hyp.real_world.yaml' 
        if ('lambda01' == opt.host_name):
            opt.source = '/data/nielseni6/ijcnn_v7data/Real_world_test/images' 
            opt.data = 'data/real_world_lambda01.yaml' 
            opt.hyp = 'data/hyp.real_world_lambda01.yaml' 
    if opt.dataset == 'coco':
        opt.source = "/data/nielseni6/coco/images"
        opt.cfg = 'cfg/training/yolov7.yaml'
        opt.hyp = 'data/hyp.scratch.p5.yaml'
        opt.data = 'data/coco_lambda01.yaml'
    
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    # print(opt)
    
    ########## CHANGE THIS TO CHANGE DATASET ##########
    opt.dataset = 'real_world_drone'
    # opt.weights_dir = 'weights/drone_eval'
    opt.batch_size = 16
    ###################################################
    # opt.dataset = 'coco'
    # opt.weights_dir = 'weights/coco_eval'
    # opt.batch_size = 8
    ###################################################
    opt.models_folder = opt.weights_dir
    initname = opt.name
    # if 'pgt' in opt.weights:
    #     opt.name += 'pgt'
    weights_list = os.listdir(opt.weights_dir)
    opt.weights = f'{opt.weights_dir}/{weights_list[0]}'
    ###################################################
    print(opt)
    
    for weight_i in range(len(weights_list)):
        opt.weights = f'{opt.weights_dir}/{weights_list[weight_i]}'
        
        opt.name = initname + weights_list[weight_i] # weights_dir.split('/')[-1]
        # wandb.config.update(opt)
        opt.allow_val_change=True
        # allow_val_change=True to config.update()
        
        if opt.task in ('train', 'val', 'test'):  # run normally
            
            opt.resume, opt.upload_dataset, opt.epochs = False, False, 1

            with open(opt.data) as f:
                data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
            
            save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | False)  # increment run
            (Path(save_dir) / 'labels' if opt.save_txt else Path(save_dir)).mkdir(parents=True, exist_ok=True)  # make dir
            
            loggers = {'wandb': None}  # loggers dict
            
            weights = opt.weights
            device = select_device(device_num, batch_size=opt.batch_size)
            # run_id = torch.load(weights, map_location=device).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
            run_id = None
            wandb_logger = WandbLogger(opt, Path(save_dir).stem, run_id, data_dict)
            loggers['wandb'] = wandb_logger.wandb
            data_dict = wandb_logger.data_dict

            for atk in atk_list:
                opt.atk = atk
                results = test_pgt(opt.data,
                    opt.weights,
                    opt.batch_size,
                    opt.img_size,
                    opt.conf_thres,
                    opt.iou_thres,
                    opt.save_json,
                    opt.single_cls,
                    opt.augment,
                    opt.verbose,
                    save_txt=opt.save_txt | opt.save_hybrid,
                    save_hybrid=opt.save_hybrid,
                    save_conf=opt.save_conf,
                    half_precision=opt.half_precision,
                    trace=not opt.no_trace,
                    #  trace=opt.no_trace,
                    v5_metric=opt.v5_metric,
                    opt = opt,
                    wandb_logger=wandb_logger,
                    compute_loss=ComputeLoss,
                    device=device,
                    )
                if opt.eval_type == 'robust_snr_vary':
                    results_snr = results
                    results, snr_list = results
                
                
                # Log
                tags = ['metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                        'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                        'plaus_score',
                        ]
                # for ir in range(len(robust_eval_results)):
                #     r_loss = torch.zeros(3, device=device)
                #     ((r_mp, r_mr, r_map50, r_map, r_loss[0], r_loss[1], r_loss[2]), r_maps, r_t, r_plaus) = robust_eval_results[ir][0]
                #     results = (r_mp, r_mr, r_map50, r_map, *(r_loss.cpu()).tolist()), r_maps, r_t, r_plaus
                for i_step, res in enumerate(results):
                    (result, maps, t) = res[0]
                    # wandb_logger.current_epoch = i_step
                    for x, tag in zip(list(result), tags):
                        if wandb_logger.wandb:
                            wandb_logger.log({tag: x})  # W&B
                    wandb_logger.end_epoch()
                
                if opt.eval_type == 'robust_snr_vary':
                    tags = ['metrics_vs_SNR/precision', 'metrics_vs_SNR/recall', 'metrics_vs_SNR/mAP_0.5', 'metrics_vs_SNR/mAP_0.5:0.95',
                            'val_vs_SNR/box_loss', 'val_vs_SNR/obj_loss', 'val_vs_SNR/cls_loss',  # val loss
                            'metrics_vs_SNR/plaus_score', 'snr/step'
                            ]
                    # define our custom x axis metric
                    wandb.define_metric("snr/step")
                    # set all other train/ metrics to use this step
                    wandb.define_metric("metrics_vs_SNR/*", step_metric="snr/step")
                    wandb.define_metric("val_vs_SNR/*", step_metric="snr/step")

                    for i_step, res in enumerate(results_snr):
                        (result, maps, t) = res[0]
                        # wandb_logger.current_epoch = i_step
                        for x, tag in zip(list(result), tags):
                            if wandb_logger.wandb:
                                wandb_logger.log({tag: x})  # W&B
                        wandb_logger.end_epoch()

        
        wandb_logger.finish_run()
        
    # elif opt.task == 'speed':  # speed benchmarks
    #     for w in opt.weights:
    #         test_pgt(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, opt=opt, save_json=False, plots=False, v5_metric=opt.v5_metric)

    # elif opt.task == 'study':  # run over a range of settings and save/plot
    #     # python test_pgt.py --task study --data coco.yaml --iou 0.65 --weights yolov7.pt
    #     x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
    #     for w in opt.weights:
    #         f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
    #         y = []  # y axis
    #         for i in x:  # img-size
    #             print(f'\nRunning {f} point {i}...')
    #             r, _, t = test_pgt(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
    #                            plots=False, v5_metric=opt.v5_metric, opt=opt)
    #             y.append(r + t)  # results and times
    #         np.savetxt(f, y, fmt='%10.4g')  # save
    #     os.system('zip -r study.zip study_*.txt')
    #     plot_study_txt(x=x)  # plot

    