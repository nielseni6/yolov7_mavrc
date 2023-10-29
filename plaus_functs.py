import torch
import numpy as np
from plot_bboxes import plot_one_box_PIL_, plot_one_box_seg
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr

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

    # Set model to evaluation mode
    model.eval()
    model.to(device)

    # Set requires_grad attribute of tensor. Important for computing gradients
    input_tensor.requires_grad = True

    # Forward pass
    out, train_out = model(input_tensor.to(device)) # inference and training outputs

    # # Apply NMS
    # out = non_max_suppression(out, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    
    num_classes = 2
    
    if targets is None:
        # Find index of class with highest score
        index = torch.argmax(out[:, :, -num_classes:], dim=-1) # num_classes = 2
    else:
        index = targets
    
    # Zero gradients
    model.zero_grad()
    
    gradients = torch.autograd.grad(train_out[2], input_tensor, grad_outputs=torch.ones_like(train_out[2]), retain_graph=True, create_graph=True)

    # # Calculate gradients of output with respect to input
    # out[:, :, -1].backward()

    # # Get gradients
    # gradients = input_tensor.grad.data
    
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
    model.train()
    
    return torch.tensor(attribution_map, dtype=torch.float32, device=device)


def eval_plausibility(im0, targets, attr_tensor):
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
    if len(targets) == 0:
        return 0
    # MIGHT NEED TO NORMALIZE OR TAKE ABS VAL OF ATTR
    eval_totals = 0
    eval_individual_data = []
    seg_box = np.zeros_like(im0.detach().cpu(), dtype=np.float32) # zeros with white pixels inside bbox
    xyxy_gnd = targets[0][2:]
    plot_one_box_seg(xyxy_gnd, seg_box, label=None, color=(255,255,255), line_thickness=-1)
    attr = (attr_tensor[0].detach().cpu().numpy())
    seg_box_norm = seg_box / float(np.max(seg_box))
    seg_box_t = seg_box_norm[0] # np.transpose(seg_box_norm[0], (2, 0, 1))
    attr_and_segbox = attr * seg_box_t
    IoU_num = float(np.sum(attr_and_segbox))
    IoU_denom = float(torch.sum(attr_tensor))
    IoU = IoU_num / IoU_denom
    eval_totals += IoU
    eval_individual_data.append(IoU)

    return eval_totals
