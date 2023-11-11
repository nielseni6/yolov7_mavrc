import torch
import numpy as np
# from plot_bboxes import plot_one_box_PIL_, plot_one_box_seg
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
import cv2
import random
from plot_functs import *
from utils.plots import plot_one_box
    
    

def generate_vanilla_grad(model, input_tensor, opt, mlc, targets = None, norm=False, device='cpu'):
    """
    Generates an attribution map using vanilla gradient method.

    Args:
        model (torch.nn.Module): The PyTorch model to generate the attribution map for.
        input_tensor (torch.Tensor): The input tensor to the model.
        norm (bool, optional): Whether to normalize the attribution map. Defaults to False.
        device (str, optional): The device to use for the computation. Defaults to 'cpu'.

    Returns:
        numpy.ndarray: The attribution map.
    """
    # maybe add model.train() at the beginning and model.eval() at the end of this function

    # Set requires_grad attribute of tensor. Important for computing gradients
    input_tensor.requires_grad = True
    
    # Forward pass
    train_out = model(input_tensor) # training outputs (no inference outputs in train mode)

    num_classes = 2
    
    # Zero gradients
    model.zero_grad()
    
    # train_out[1] = torch.Size([4, 3, 80, 80, 7]) HxWx(#anchorxC) cls (class probabilities)
    # train_out[0] = torch.Size([4, 3, 160, 160, 7]) HxWx(#anchorx4) reg (location and scaling)
    # train_out[2] = torch.Size([4, 3, 40, 40, 7]) HxWx(#anchorx1) obj (objectness score or confidence)
    
    gradients = torch.autograd.grad(train_out[1], input_tensor, 
                                    grad_outputs=torch.ones_like(train_out[1]), 
                                    retain_graph=True, create_graph=True)

    # Convert gradients to numpy array
    gradients = gradients[0].detach().cpu().numpy()

    if norm:
        # Take absolute values of gradients
        gradients = np.absolute(gradients)

        # Sum across color channels
        attribution_map = np.sum(gradients, axis=0)

        # Normalize attribution map
        attribution_map /= np.max(attribution_map)
    else:
        # Sum across color channels
        attribution_map = gradients

    # Set model back to training mode
    # model.train()
    
    return torch.tensor(attribution_map, dtype=torch.float32, device=device)


def eval_plausibility(imgs, targets, attr_tensor):
    """
    Evaluate the plausibility of an object detection prediction by computing the Intersection over Union (IoU) between
    the predicted bounding box and the ground truth bounding box.

    Args:
        im0 (numpy.ndarray): The input image.
        targets (list): A list of targets, where each target is a list containing the class label and the ground truth
            bounding box coordinates in the format [class_label, x1, y1, x2, y2].
        attr (torch.Tensor): A tensor containing the normalized attribute values for the predicted
            bounding box.

    Returns:
        float: The total IoU score for all predicted bounding boxes.
    """
    # if len(targets) == 0:
    #     return 0
    # MIGHT NEED TO NORMALIZE OR TAKE ABS VAL OF ATTR
    # ALSO MIGHT NORMALIZE FOR THE SIZE OF THE BBOX
    eval_totals = 0
    eval_individual_data = []
    targets_ = [[targets[i] for i in range(len(targets)) if int(targets[i][0]) == j] for j in range(int(max(targets[:,0])))]
    for i, im0 in enumerate(imgs):
        if len(targets[i]) == 0:
            eval_individual_data.append([torch.tensor(0),])
        else:
            IoU_list = []
            for targs in targets_:
                for j in range(len(targs)):
                    xyxy_pred = targs[j][2:] # * torch.tensor([im0.shape[2], im0.shape[1], im0.shape[2], im0.shape[1]])
                    seg_box = np.zeros_like(im0.detach().cpu(), dtype=np.float32) # zeros with white pixels inside bbox
                    seg_box = plot_one_box_seg(xyxy_pred, seg_box, label=None, color=(255,255,255), line_thickness=-1)
                    # imshow(im0[i], "./figs/im0_i")
                    # imshow(seg_box, "./figs/seg_box")
                    attr = VisualizeNumpyImageGrayscale(abs(attr_tensor[0].detach().cpu().numpy()))
                    seg_box_norm = seg_box / float(np.max(seg_box))
                    seg_box_t = seg_box_norm[0] # np.transpose(seg_box_norm[0], (2, 0, 1))
                    attr_and_segbox = attr * seg_box_t
                    IoU_num = float(np.sum(attr_and_segbox))
                    IoU_denom = float(np.sum(attr))
                    IoU = IoU_num / IoU_denom
                    IoU_list.append(IoU)
            eval_totals += np.mean(IoU_list)
            eval_individual_data.append(torch.tensor(IoU_list))

    return torch.tensor(eval_totals).requires_grad_(True)

def plot_one_box_seg(x, img, color=None, label=None, line_thickness=-1, center_coords = True):
    def corners_coords(center_xywh):
        center_x, center_y, w, h = center_xywh
        x = center_x - w/2
        y = center_y - h/2
        return np.array([x, y, x+w, y+h])

    if center_coords:
        x = corners_coords(x) * np.array([img.shape[1], img.shape[2], img.shape[1], img.shape[2]])
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[2] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    for i in range(img.shape[0]):
        img[i,c1[1]:c2[1], c1[0]:c2[0]] = color[i]
    return img
#     test_im = cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
#     # imshow(test_im, "./figs/test_im")
#     # imshow(img, "./figs/img")
#     return test_im