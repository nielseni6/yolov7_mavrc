import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from PIL import Image
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from pytorch_grad_cam import EigenCAM, GradCAM, ScoreCAM, EigenGradCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from pytorch_grad_cam.utils.find_layers import find_layer_predicate_recursive

# from archive.cam_utils import renormalize_cam_in_bounding_boxes
# from torchsummary import summary

def returnGrad(img, model, device, augment):
    model.to(device)
    img = img.to(device)
    img.requires_grad_(True)
    pred = model(img, augment)
    pred = torch.tensor([sum(sum(sum(pred[0])))])
    pred.backward()
    # loss = criterion(pred, target.to(device))
    # loss.backward()
    
#    S_c = torch.max(pred[0].data, 0)[0]
    Sc_dx = img.grad
    
    return Sc_dx

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    print('Trace:', trace)
    # Directories
    save_dir = []
    for itl, cam_name in enumerate(opt.cam_list):
        save_dir.append(Path(increment_path(Path(opt.project) / str(opt.name + '_' + cam_name), exist_ok=opt.exist_ok)))  # increment run
        (save_dir[itl] / 'labels' if save_txt else save_dir[itl]).mkdir(parents=True, exist_ok=True)  # make dir

    # save_dir = Path(increment_path(Path(opt.project) / str(opt.name + '_' + opt.cam), exist_ok=opt.exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    def load_model(imgsz):
        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        # print(model.model.num_features)

        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if trace:
            model = TracedModel(model, device, opt.img_size)

        if half:
            model.half()  # to FP16

        return model, imgsz, stride

    model, imgsz, stride = load_model(imgsz)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    model, imgsz, stride = load_model(imgsz) # load model
    model_unseg = model
    model_seg_complete = False
    img_num = 0
    
    skip = False
    for path, img, im0s, vid_cap in dataset:
        process_img = True #img_num > 0 # if false the image will be skipped
        if not process_img:
            if skip:
                print("Skipping image", img_num, end='\r')
            else:
                print("Skipping image", img_num)
            skip = True
        else:
            print("Processing image", img_num)
            skip = False
        img_num += 1
        if process_img:
            
            # model, imgsz, stride = load_model(imgsz)
            # print(path)
            # print(type(im0s))
            # im0s=Image.open(im0s)convert("L")
            # Image.fromarray(im0s).convert("L").save("dtest.png")
            # exit()

            img = torch.from_numpy(img).to(device)
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
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # store pred label
            # print(pred)
            # pred_classify = apply_classifier(pred, modelc, img, im0s)

            # print("Process detections")
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = []
                for itl, cam_name in enumerate(opt.cam_list):
                    save_path.append(str(save_dir[itl] / p.name))  # img.jpg

                # save_path = str(save_dir / p.name)  # img.jpg # original in case loop fails
                txt_path = str(save_dir[itl] / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                if len(det) or not opt.only_detect_true:
                    if len(det):
                        print('|| Object detected ', s,'||')
                    # Print time (inference + NMS)
                    print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                    # Stream results
                    if view_img:
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 millisecond

                    for cam_num, cam in enumerate(opt.cam_list):
                        # model.zero_grad()
                        #eigen cam
                        # cam=opt.cam
                        opt.cam = cam
                        save_path += s
                        if cam != None:
                            print("Calculating ", cam, "...")
                            model = model_unseg
                            model = model.eval()
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
                                grayscale_cam = cam_(input_tensor=img.requires_grad_(True),
                                                    targets=targets)#[0, :]
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
                            if cam == 'vanilla_grad':
                                grayscale_cam = returnGrad(img, model=model, device=device, augment=opt.augment)
                            
                            print(cam)
                            # else:  # default now ScoreCAM (might change to gradient)
                            #     cam_ = ScoreCAM(model, target_layers, use_cuda=False)
                            
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
                            norm=False
                            if norm:
                                renormalize_cam_in_bounding_boxes(xyxy, colors,im0copy, grayscale_cam)
                                Image.fromarray(renormalized_cam_image).save("eignenorm_out_d.png")
                                print(renormalized_cam_image.shape)


                        # Save results (image with detections)
                        if save_img:
                            if dataset.mode == 'image':
                                cv2.imwrite(save_path[cam_num], im0)
                                print(f" The image with the result is saved in: {save_path[cam_num]}")
                            else:  # 'video' or 'stream'
                                if vid_path != save_path[cam_num]:  # new video
                                    vid_path = save_path[cam_num]
                                    if isinstance(vid_writer, cv2.VideoWriter):
                                        vid_writer.release()  # release previous video writer
                                    if vid_cap:  # video
                                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                        
                                    else:  # stream
                                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                                        save_path[cam_num] += str(cam)
                                        save_path[cam_num] += '.mp4'
                                    vid_writer = cv2.VideoWriter(save_path[cam_num], cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                                vid_writer.write(im0)
                    # model, imgsz, stride = load_model(imgsz)
                else:
                    print('|| No object detected for path', path, '||')
    if save_txt or save_img:
        for itl in range(len(opt.cam_list)):
            s = f"\n{len(list(save_dir[itl].glob('labels/*.txt')))} labels saved to {save_dir[itl] / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir[itl]}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()

class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model): 
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x)[-1]
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--cam', default='eigen', help='enable eigencam view') # USE cam_list INSTEAD

    parser.add_argument('--only-detect-true', action='store_true', help='don`t save images that model didnt detect object in')
    parser.add_argument('--cam_list', nargs='+', type=str, help='a list of cam methods to use')

    # parser.add_argument('--cam', action="store_true", help='enable eigencam view')
    opt = parser.parse_args()

    ### FOR DEBUGGING PURPOSES (DELETE ONCE DEBUGGING COMPLETE) ###
    # parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt.device = 'cpu' 
    # parser.add_argument('--weights', nargs='+', type=str, default='weights/best.pt', help='model.pt path(s)')
    # opt.weights = 'weights/best.pt'
    opt.weights = 'weights/yolov7-tiny.pt'
    opt.conf_thres = 0.50 
    opt.img_size = 480 
    opt.no_trace = True
    # opt.view_img = True
    # opt.name = "practice_data" 
    # opt.name = "Real_world_test"
    opt.name = "Birds_Animals"
    opt.cam_list = ["grad", "eigen"]
    # cam_list = ["score","fullgrad",]
    # opt.cam_list = ["fullgrad", "eigengrad", "grad", "eigen", ]
    # opt.cam = "grad"
    # opt.cam = "eigen"
    # opt.cam = 'vanilla_grad'
    # opt.cam = "grad_test"
    # opt.source = '/home/nielseni6/PythonScripts/yolov7_mavrc/practice_data/small_set/yolo/images' 
    # opt.source = '/data/Koutsoubn8/ijcnn_v7data/Real_world_test/images'
    opt.source = '../../../../data/nielseni6/Birds_Animals/images'

    opt.only_detect_true = True

    print('opt.cam_list:', opt.cam_list)

    ###############################################################

    print(opt)


    #check_requirements(exclude=('pycocotools', 'thop'))

    
    # with torch.no_grad():
    if opt.update:  # update all models (to fix SourceChangeWarning)
        for opt.weights in ['yolov7.pt']:
            detect()
            strip_optimizer(opt.weights)
    else:
        detect()