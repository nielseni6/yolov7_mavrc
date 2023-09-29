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

from detect import SegmentationModelOutputWrapper, SemanticSegmentationTarget
from pytorch_grad_cam.utils.find_layers import find_layer_predicate_recursive
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam import EigenCAM, GradCAM, ScoreCAM, EigenGradCAM, FullGrad
from PIL import Image
import cv2
from utils.datasets import LoadStreams, LoadImages, LoadImagesAndLabels
import Perturb_Functions as pert_func
from utils.loss import ComputeLoss, ComputeLossOTA
import torchvision
import matplotlib.pyplot as plt
from utils.plots import plot_one_box
from numpy import random

import sys
import traceback

from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr

class TracePrints(object):
  def __init__(self):    
    self.stdout = sys.stdout
  def write(self, s):
    self.stdout.write("Writing %r\n" % s)
    traceback.print_stack(file=self.stdout)

def VisualizeImageGrayscale(image_3d):
  r"""Returns a 3D tensor as a grayscale normalized between 0 and 1 2D tensor.
  """
  vmin = torch.min(image_3d)
  image_2d = image_3d - vmin
  vmax = torch.max(image_2d)
  return (image_2d / vmax)

def VisualizeNumpyImageGrayscale(image_3d):
  r"""Returns a 3D tensor as a grayscale normalized between 0 and 1 2D tensor.
  """
  vmin = np.min(image_3d)
  image_2d = image_3d - vmin
  vmax = np.max(image_2d)
  return (image_2d / vmax)

def format_img(img_):
    img_ = img_     # unnormalize
    np_img = img_.numpy()
    tp_img = np.transpose(np_img, (1, 2, 0))
    return tp_img

def imshow(img, save_path=None):
    img = img     # unnormalize
    try:
        npimg = img.cpu().detach().numpy()
    except:
        npimg = img
    tpimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(tpimg)
    if save_path != None:
        plt.savefig(str(str(save_path) + ".png"))
    #plt.show()

def imshow_img_tensor(img, imsave_path):
    npimg = VisualizeNumpyImageGrayscale(img)
    # npimg = np.transpose(npimg, (2, 0, 1))
    imshow(npimg, save_path=imsave_path)
    # img = img     # unnormalize
    # try:
    #     npimg = img.cpu().detach().numpy()
    # except:
    #     npimg = img
    # tpimg = np.transpose(npimg, (1, 2, 0))
    # plt.imshow(tpimg)
    # if save_path != None:
    #     plt.savefig(str(str(save_path) + ".png"))
    print("Saving image as ", imsave_path)

def test(opt,
         data,
         weights=None,
         batch_size=32,
         imgsz=480,
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
         ):
    # Initialize/load model and set device

    opt.device = 'cpu'
    device = 'cpu'
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir_attr = []
        for itl, cam_name in enumerate(opt.cam_list):
            save_dir_attr.append(Path(increment_path(Path(opt.project) / str(opt.name + '_' + cam_name), exist_ok=opt.exist_ok)))  # increment run
            (save_dir_attr[itl] / 'labels' if save_txt else save_dir_attr[itl]).mkdir(parents=True, exist_ok=True)  # make dir

        # # Directories
        # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        
        data_type = 'real_world'
        run_num = 1
        # save_dir = "perturb_eval/"
        try:
            # dir_list_figs = os.listdir(str("./results/" + save_dir.__str__()))
            dir_list_figs = os.listdir(str(save_dir.__str__()))
            for j, x in enumerate(dir_list_figs):
                #print(x)
                if (data_type in x) and (str('acc' + str(j)) not in x):
                    run_num = j
        except:
            print("No figs yet in", str(save_dir.__str__()))

        def load_model(imgsz):
            # Load model
            model = attempt_load(weights, map_location=device)  # load FP32 model
            # print(model.model.num_features)

            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check img_size

            if trace:
                model = TracedModel(model, device, opt.img_size)

            # if half:
            #     model.half()  # to FP16

            model.stride = stride
            
            return model, imgsz, stride
        
        model, imgsz, stride = load_model(imgsz) # load model
        # # Load model
        # model = attempt_load(weights, map_location=device)  # load FP32 model
        # # model, imgsz, stride = load_model(imgsz) # load model
        # stride = int(model.stride.max())
        gs = stride
        # gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        # imgsz = check_img_size(imgsz, s=gs)  # check img_size
        
        # if trace:
        #     model = TracedModel(model, device, imgsz)
    
    dataset = LoadImages(opt.source, img_size=imgsz, stride=stride)
    dataset_w_labels = LoadImagesAndLabels(opt.source, img_size=imgsz, batch_size=1, augment=opt.augment, stride=stride)#, hyp=opt.hyp)
    
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # # Configure
    # model.eval()
    # if isinstance(data, str):
    #     is_coco = data.endswith('hyp.real_world.yaml')
    #     with open(data) as f:
    #         data = yaml.load(f, Loader=yaml.SafeLoader)
    # check_dataset(data)  # check
    # nc = 1 if single_cls else int(data['nc'])  # number of classes
    nc = 2
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    # nc = opt.classes
    
    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    # if not training:
    #     if device.type != 'cpu':
    #         model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    #     task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
    #     dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
    #                                    prefix=colorstr(f'{task}: '))[0]

    if v5_metric:
        print("Testing with YOLOv5 AP metric...")
    
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    # names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    # coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    
    ########### EvalAttAI Variables ###########
    increment = 1
    t_ = [round(float(p_it * increment),2) for p_it in range(10)]
    epsilon = 0.25
    add_or_subtract_attr = False
    normalize = True
    grayscale = True
    n_samples = opt.n_samples
    samples_processed = 0
    robust = ''
    ###########################################
    if grayscale:
        gray = '_grayscale'
    else:
        gray = ''
    squared_maps = False
    add_or_subtract_attr = True
    if add_or_subtract_attr:
        addsub = 'AddAttr'
    else:
        addsub = 'SubtractAttr'

    normalize = False
    if normalize:
        norm = '_normalized'
    else:
        norm = ''
    Z = 1.960
    save_result_img = True
    ###########################################
    img_num = 0
    skip = False
    # model_unseg = model

    opt.hyp = 'data/hyp.real_world.yaml'
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = 0.0
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    compute_loss = ComputeLoss(model)
    loss_averages_list = []
    accuracy_averages_list, accuracy_CI_list = [], []
    results = []
    n_steps_t = 10
    
    model, imgsz, stride = load_model(imgsz)
    
    dataset = LoadImages(opt.source, img_size=imgsz, stride=stride)
    
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    model, imgsz, stride = load_model(imgsz) # load model
    model_unseg = model
    img_num = 0
    dataset.__iter__()
    for batch_i, data in enumerate(dataset_w_labels):
        img, labels, path, shapes = data

        path_, img_, im0s, vid_cap_ = dataset.__next__()
        im0 = im0s
        
        # seg_box = torch.zeros_like(img)# zeros with white pixels inside bbox
        # y1, y2, x1, x2 = labels[0][2], labels[0][4], labels[0][3], labels[0][5]
        # y1, y2, x1, x2 = int(y1*opt.img_size), int(y2*opt.img_size), int(x1*opt.img_size), int(x2*opt.img_size)
        # seg_box[:, y1:y2, x1:x2] = 1
        # imshow_img_tensor(seg_box, str("./figs/test_bbox"))
    # for batch_i, (path, img, im0s, vid_cap) in enumerate(dataset):
        
        # img, label, path, shapes = dataset_w_labels
    # for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        paths = [path,]
        process_img = img_num > 0 #11 # if false the image will be skipped
        # print('targets:', targets)
        # print('shapes:', shapes)
        if samples_processed > (n_samples + 1):
            break
        # if not process_img:
        #     if skip:
        #         print("Skipping image", img_num, end='\r')
        #     else:
        #         print("Skipping image", img_num)
        #     skip = True
        # else:
        #     print("Processing image", img_num)
        #     skip = False
        img_num += 1
        if process_img and (samples_processed < (n_samples + 1)):
        ###################################### def generate_attr(): ######################################
            samples_processed += 1
            print("Processing Sample", samples_processed, "...       img_num:", img_num)
            # print("Inference")
            # Inference
            t1 = time_synchronized()

            # img = img.to(dtype=torch.float32).requires_grad_(True)
            img = img.to(device)
            # img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]

            # print("Inference")
            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()
            print("Inference time: ", str(t2-t1))

            # print("Apply NMS")
            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            # if classify:
            #     pred = apply_classifier(pred, modelc, img, im0s)


            # Apply Classifier
            # if classify:
            #     pred = apply_classifier(pred, modelc, img, im0s)

            # store pred label
            # print(pred)
            # pred_classify = apply_classifier(pred, modelc, img, im0s)
            
            # dataset = data
            # source = data['test']
            # dataset = LoadImages(source, img_size=imgsz, stride=stride)
            # print("Process detections")
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                # if webcam:  # batch_size >= 1
                #     p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                # else:
                # try:
                #     path_, img_, im0s, vid_cap = dataset.__next__()
                # except:
                #     dataset.__iter__()
                #     path_, img_, im0s, vid_cap = dataset.__next__()

                # p, s, frame = path, '', getattr(dataset, 'frame', 0)
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                
                # p = paths[i]
                # s = ''
                # im0 = im0s
                # frame = getattr(dataset, 'frame', 0)
                # p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = []
                for itl, cam_name in enumerate(opt.cam_list):
                    save_path.append(str(save_dir_attr[itl] / p.name))  # img.jpg

                # save_path = str(save_dir_attr / p.name)  # img.jpg # original in case loop fails
                txt_path = str(save_dir_attr[itl] / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls_ in reversed(det):
                        # print(xyxy)
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls_, *xywh, conf) if opt.save_conf else (cls_, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if opt.save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls_)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls_)], line_thickness=1)
                    
                    npimg = VisualizeNumpyImageGrayscale(im0)
                    npimg = np.transpose(npimg, (2, 0, 1))
                    imsave_path = str(str("./figs/im0s_test")+str(1))
                    # npimg = np.array([npimg])
                    # cv2.imwrite(imsave_path, im0)
                    imshow(npimg, save_path=imsave_path)
                    print("Saving image as ", imsave_path)
                    
                if len(det) or not opt.only_detect_true:
                    if len(det):
                        print('|| Object detected ', s,'||')
                    # Print time (inference + NMS)
                    print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                    # Stream results
                    # if view_img:
                    #     cv2.imshow(str(p), im0)
                    #     cv2.waitKey(1)  # 1 millisecond

                    for cam_num, cam in enumerate(opt.cam_list):
                        # model.zero_grad()
                        #eigen cam
                        # cam=opt.cam
                        opt.cam = cam
                        save_path += s
                        if cam != None:
                            print("Calculating", cam, "...")
                            model = model_unseg
                            model = model.eval()
                            targets = [ClassifierOutputTarget(1)]
                            if cam != 'eigen':
                                # if not model_seg_complete:
                                model_seg = SegmentationModelOutputWrapper(model) # model_seg
                                #     model_seg_complete = True
                                # else:
                                #     model_seg = model
                            # else:
                            #     model = model_unseg

                            if cam == 'eigen':# or cam == 'eigengrad':# or cam == 'score':
                                target_layers = [model.model[1]]
                                targets = [ClassifierOutputTarget(1)]
                            elif cam == 'fullgrad' or cam == 'eigengrad' or cam == 'score':
                                def layer_with_2D_bias(layer):
                                    bias_target_layers = [torch.nn.Conv2d, torch.nn.BatchNorm2d]
                                    if type(layer) in bias_target_layers and layer.bias is not None:
                                        return True
                                    return False
                                target_layers = find_layer_predicate_recursive(
                                    model, layer_with_2D_bias)
                                # targets = [ClassifierOutputTarget(1)]
                                # model = model_seg # model_seg
                                output = model_seg(img) # model_seg
                                normalized_masks = torch.nn.functional.softmax(output[0], dim=1).cpu()
                                mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
                                mask_float = np.float32(mask == 0)
                                targets = [SemanticSegmentationTarget(0, mask_float)]
                            elif cam == 'grad':
                                output = model_seg(img) # model_seg
                                normalized_masks = torch.nn.functional.softmax(output[0], dim=1).cpu()
                                mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
                                mask_float = np.float32(mask == 0)
                                target_layers = [model_seg.model.model[1]] # model_seg

                                # upsample = torch.nn.UpsamplingBilinear2d(
                                #     size=input_tensor.shape[-2:])
                                # activation_tensor = torch.from_numpy(activations)

                                # upsampled = upsample(activation_tensor)

                                targets = [SemanticSegmentationTarget(0, mask_float)]
                            
                            if cam == 'eigen':
                                cam_ = EigenCAM(model, target_layers, use_cuda=False)          # exit()
                                grayscale_cam = cam_(input_tensor=img,targets=targets,eigen_smooth=True,aug_smooth=False)
                            
                            if cam == 'grad':
                                cam_ = GradCAM(model_seg, target_layers, use_cuda=False) # model_seg
                                # grayscale_cam = cam_(input_tensor=img.requires_grad_(True),targets=targets,eigen_smooth=False,aug_smooth=False)
                                # print('pre-cam')
                                
                                grayscale_cam = cam_(input_tensor=img.requires_grad_(True),
                                                    targets=targets)#[0, :]
                                # print('post-cam')
                                print("gradcam complete")
                            if cam == "eigengrad": 
                                cam_ = EigenGradCAM(model_seg, target_layers, use_cuda=False) # model_seg
                                grayscale_cam = cam_(input_tensor=img.requires_grad_(True), targets=targets,
                                                    eigen_smooth=True,aug_smooth=False)
                            if cam == "fullgrad": 
                                cam_ = FullGrad(model_seg, target_layers, use_cuda=False) # model_seg
                                grayscale_cam = cam_(input_tensor=img.requires_grad_(True), targets=targets)
                            if cam == "score":
                                cam_ = ScoreCAM(model_seg, target_layers, use_cuda=False) # model_seg
                                grayscale_cam = cam_(input_tensor=img.requires_grad_(True), targets=targets)
                            # if cam == 'vanilla_grad':
                            #     grayscale_cam = returnGrad(img, model=model, device=device, augment=opt.augment)
                            imshape = (img.shape)
                            if cam == "random":
                                # cam_ = GradCAM(model_seg, target_layers, use_cuda=False) # model_seg
                                # grayscale_cam = cam_(input_tensor=img.requires_grad_(True),
                                #                     targets=targets)#[0, :] # Placeholder CAM to be replaced with random
                                
                                # rand_grad = torch.tensor(np.float32(np.random.rand(1, 3, 512, 512)))
                                rand_grad = torch.tensor(np.float32(np.random.rand(imshape[0],imshape[1],imshape[2],imshape[3])))
                                # grayscale_cam = torch.tensor(np.float32(np.random.rand(img.shape)))
                                grayscale_cam = np.array(rand_grad)
                                
                            print(cam)
                            # else:  # default now ScoreCAM (might change to gradient)
                            #     cam_ = ScoreCAM(model, target_layers, use_cuda=False)
                            # imshape = (img.shape)
                            # Generate random placeholder attribution
                            rand_attr = torch.tensor(np.float32(np.random.rand(imshape[0],imshape[1],imshape[2],imshape[3])))# * scale
                            
                            gray_cam_tensor = torch.tensor(grayscale_cam)
                            # print(str(grayscale_cam.shape[1] == 3))
                            if grayscale and not (grayscale_cam.shape[1] == 3):
                                # gmap = torch.sum(rand_grad, dim = 1)
                                rand_attr[0][0] = gray_cam_tensor[0]
                                rand_attr[0][1] = gray_cam_tensor[0]
                                rand_attr[0][2] = gray_cam_tensor[0]

                                attr_multi_channel = rand_attr
                            else:
                                attr_multi_channel = gray_cam_tensor
                                gray_cam_tensor = torch.mean(gray_cam_tensor, axis=1)
                                grayscale_cam = np.array(gray_cam_tensor)

                            results.append((attr_multi_channel, img, labels, cam))

                            grayscale_cam = grayscale_cam[0, :] #.clone().detach().requires_grad_()
                            # im0=Image.open(source).convert("L")
                            # im0=Image.fromarray(im0s).convert("L")
                            im0copy=im0.copy()
                            # print("before",type(im0),im0.shape)
                            im0=np.float32(im0) / 255
                            im0=cv2.resize(im0,(img.shape[3],img.shape[2]))
                        
                            im0 = show_cam_on_image(im0, grayscale_cam, use_rgb=True)
                            Image.fromarray(im0).save("eigenout_d.png")
                            im0=cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
                            # print(type(im0),im0.shape)
                            im0=cv2.resize(im0,(im0copy.shape[1],im0copy.shape[0]))
                            # normalize=False
                            # if normalize:
                            #     renormalize_cam_in_bounding_boxes(xyxy, colors,im0copy, grayscale_cam)
                            #     Image.fromarray(renormalized_cam_image).save("eignenorm_out_d.png")
                            #     print(renormalized_cam_image.shape)
    else:
            try:
                print('|| No object detected for path', paths[img_num % batch_size], '||', end='\r')        
            except:
                print('|| No object detected ||', end='\r')
            ###################################### end of generate_attr ######################################
    for current_cam in opt.cam_list:
        loss_totals = [0] * n_steps_t
        loss_averages = [0] * n_steps_t
        loss_averages_CI = [0] * n_steps_t
        for ite in range(n_steps_t):
            print('Running t_:', t_[ite])     
            ##################### INDENTED, ADD BELOW #############################
            # sys.stdout = TracePrints()
            num_sampled = 0
            # loss = 0.0
            for j, (attr_multi_channel, img, targets, cam) in enumerate(results):
                if current_cam == cam:
                    t_iter = t_[ite]
                    # img = results[j][1]
                    # attr_multi_channel = results[j][0]
                    # convert grayscale_cam to three channel
                    img_reg = (img / 255.0)
                    formatted_img = pert_func.attack_data_format((img / 255.0), attr_multi_channel, 
                                                            n_steps = int(t_iter), epsilon = epsilon, 
                                                            add_or_subtract_attr = add_or_subtract_attr, 
                                                            normalize=normalize).float()
                    
                    # output = model(formatted_img)
                    formatted_img.requires_grad_(False)
                    formatted_img = formatted_img.to(device, non_blocking=True)
                    formatted_img = formatted_img.half() if half else formatted_img.float()  # uint8 to fp16/32
                    formatted_img *= 255.0  # 0.0 - 1.0 to 0 - 255 
                    # formatted_img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    
                    targets = torch.tensor(targets)
                    # targets = targets.to(device)
                    nb, _, height, width = formatted_img.shape  # batch size, channels, height, width

                    with torch.no_grad():
                        # Run model
                        t = time_synchronized()
                        out, train_out = model(formatted_img, augment=augment)  # inference and training outputs
                        t0 += time_synchronized() - t

                        # Compute loss
                        if compute_loss:
                            loss = compute_loss([x.float() for x in train_out], targets, metric=loss_metric)[1][:3]  # box, obj, cls
                            # loss, loss_items = compute_loss([x.float() for x in train_out], labels, metric=loss_metric)[1][:3]  # box, obj, cls

                        loss_totals[int(ite)] += float(torch.sum(loss))
                        num_sampled += 1
                        
                        # Run NMS
                        targets[:, 2:] *= torch.Tensor([width, height, width, height])#.to(device)  # to pixels
                        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
                        t = time_synchronized()
                        out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
                        t1 += time_synchronized() - t

                    # Statistics per image
                    for si, pred in enumerate(out):
                        labels = targets[targets[:, 0] == si, 1:]
                        nl = len(labels)
                        tcls = labels[:, 0].tolist() if nl else []  # target class
                        # path = Path(paths[si])
                        seen += 1

                        if len(pred) == 0:
                            if nl:
                                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                            continue

                        # Predictions
                        predn = pred.clone()
                        # scale_coords(formatted_img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred
                        scale_coords(formatted_img[si].shape[1:], predn[:, :4], shapes[0], shapes[1])  # native-space pred

                        # Append to text file
                        if save_txt:
                            # gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                            gn = torch.tensor(shapes[0])[[1, 0, 1, 0]]  # normalization gain whwh
                            for *xyxy, conf, cls in predn.tolist():
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                with open(save_dir_attr / 'labels' / (path.stem + '.txt'), 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        # W&B logging - Media Panel Plots
                        if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                            if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                                box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                            "class_id": int(cls),
                                            "box_caption": "%s %.3f" % (names[cls], conf),
                                            "scores": {"class_score": conf},
                                            "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                                boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                                wandb_images.append(wandb_logger.wandb.Image(formatted_img[si], boxes=boxes, caption=path.name))
                        wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

                        # Append to pycocotools JSON dictionary
                        if save_json:
                            # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                            image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                            box = xyxy2xywh(predn[:, :4])  # xywh
                            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                            for p, b in zip(pred.tolist(), box.tolist()):
                                jdict.append({'image_id': image_id,
                                            'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                            'bbox': [round(x, 3) for x in b],
                                            'score': round(p[4], 5)})

                        # Assign all predictions as incorrect
                        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
                        if nl:
                            detected = []  # target indices
                            tcls_tensor = labels[:, 0]

                            # target boxes
                            tbox = xywh2xyxy(labels[:, 1:5])
                            # scale_coords(formatted_img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                            scale_coords(formatted_img[si].shape[1:], tbox, shapes[0], shapes[1])  # native-space labels
                            if plots:
                                confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                            # Per target class
                            for cls in torch.unique(tcls_tensor):
                                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                                pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                                # Search for detections
                                if pi.shape[0]:
                                    # Prediction to target ious
                                    ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                                    # Append detections
                                    detected_set = set()
                                    for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                        d = ti[i[j]]  # detected target
                                        if d.item() not in detected_set:
                                            detected_set.add(d.item())
                                            detected.append(d)
                                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                            if len(detected) == nl:  # all targets already located in image
                                                break

                        # Append statistics (correct, conf, pcls, tcls)
                        # stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
                        # total_correct = int(sum(sum((correct))))
                        # accuracy_averages = total_correct
                        # print('Accuracy Average ', t_iter, ': ', accuracy_averages, '\r')
                    
                        ################################## MAKE SURE TO TAKE ACC FOR MULTIPLE SAMPLES ##################################
                        
                        
                        # accuracy_CI_list.append(accuracy_averages_CI)
                        # np.savetxt(str("./results/" + save_dir.__str__() + "faithful_eval_acc" + str(run_num) + "_" + addsub + 
                        
                        # # ADD CONFIDENCE INTERVAL CALCULATION
                        # acc_total = np.mean(acc_saved, axis=0)
                        # delta_y = 1 - np.array(acc_saved)
                        # stdev = np.std(acc_saved)
                        # acc_CI = Z * (stdev / (len(acc_saved) ** 0.5))
                        
                        # accuracy_averages_CI = acc_CI

                    # loss_totals[int(ite)] += float(torch.sum(loss))
                    # loss = 0.0
                    # num_sampled += 1

                    # Plot images
                    if plots and batch_i < 3:
                        f = save_dir_attr / f'test_batch{batch_i}_labels.jpg'  # labels
                        Thread(target=plot_images, args=(formatted_img, targets, paths, f, names), daemon=True).start()
                        f = save_dir_attr / f'test_batch{batch_i}_pred.jpg'  # predictions
                        Thread(target=plot_images, args=(formatted_img, output_to_target(out), paths, f, names), daemon=True).start()
        ########################################################################################
            loss_averages[ite] = loss_totals[ite] / num_sampled # (j) # len(results)
            print('Loss Average ', t_iter, ': ', loss_averages[ite], '\r')
            # print("loss_averages: ", loss_averages)
    
        # accuracy_averages_list.append(accuracy_averages)
        loss_averages_list.append(loss_averages)
        # print("loss_averages_list: ", loss_averages_list)
        print('Saving csv results in', save_dir.__str__())
        # np.savetxt(str(save_dir.__str__() + "faithful_eval_acc" + str(run_num) + "_" + addsub + 
        #                             "_" + norm + '_inc' + str(increment) + '_nsamples' + str(n_samples) + gray + "_" + 
        #                             robust +'_robust_' + '_img_num' + str(img_num)+ ".csv"), accuracy_averages_list, 
        #                             delimiter = ", ", fmt = "% s")
        img_num_str = str('_img_num' + str(img_num))
        for cam in opt.cam_list:
            img_num_str = str(img_num_str + '_' + cam)
        save_to = str(save_dir.__str__() + "/faithful_eval_loss" + str(run_num) + "_" + addsub + 
                                    "_" + norm + '_inc' + str(increment) + '_nsamples' + str(n_samples) + gray + "_" + 
                                    robust +'_robust_' + img_num_str)
        np.savetxt(str(save_to + ".txt"), opt.cam_list, 
                                    delimiter = ", ", fmt = "% s")
        np.savetxt(str(save_to + ".csv"), loss_averages_list, 
                                    delimiter = ", ", fmt = "% s")
        print("Full csv results saved to ", save_to)
    
    
    
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names) # replace save_dir[0] zero to make work with multiple attributions
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
  
    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = './coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    
    return (mp, mr, map50, map, *(loss_totals)), maps, t
    # return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/hyp.real_world.yaml', help='*.data path')
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
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument('--save-img', action='store_true', help='save image with bounding box')
    ############################################################################
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')    
    parser.add_argument('--cam_list', nargs='+', type=str, help='a list of cam methods to use')
    parser.add_argument('--only-detect-true', action='store_true', help='don`t save images that model didnt detect object in')
    parser.add_argument('--eval_method', default='evalattai', help='evalattai (faithfulness) or plausibility')
    ############################################################################
    opt = parser.parse_args()
    # opt.save_json |= opt.data.endswith('real_world.yaml')
    opt.save_json = False
    opt.data = check_file(opt.data)  # check file
    print(opt)
    #check_requirements()

    # opt.cam_list = ["random", "fullgrad", "grad", "eigen", "eigengrad",]
    # opt.cam_list = ["fullgrad", "grad", "eigen"]
    # opt.cam_list = ["random","grad",]
    opt.cam_list = ["random",]
    # opt.cam_list = ["grad",]
    opt.source = '/data/Koutsoubn8/ijcnn_v7data/Real_world_test/images'
    opt.no_trace = True
    opt.conf_thres = 0.50 
    opt.batch_size = 1 
    opt.data = 'data/real_world.yaml'
    # opt.data = 'data/sls.yaml'
    opt.img_size = 480 
    opt.name = 'test1' 
    opt.weights = 'weights/yolov7-tiny.pt' 
    opt.task = 'val'
    opt.n_samples = 20
    # opt.n_samples = 100
    opt.save_img = True
    # opt.classes = 2
    opt.only_detect_true = True
    opt.eval_method = 'evalattai'
    # opt.eval_method = 'plausibility'

    # opt.single_cls = True

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt,
             opt.data,
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
             trace=not opt.no_trace,
             v5_metric=opt.v5_metric,
             )

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, v5_metric=opt.v5_metric)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.65 --weights yolov7.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, v5_metric=opt.v5_metric)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
