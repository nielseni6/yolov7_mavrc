import torch
import numpy as np
from plot_functs import * 
from plot_functs import normalize_tensor
import math   
import time

def generate_vanilla_grad(model, input_tensor, loss_func = None, 
                          targets_list=None, metric=None, out_num = 1, 
                          n_max_labels=8, norm=True, abs=True, grayscale=True, 
                          class_specific_attr = True, device='cpu'):    
    """
    Computes the vanilla gradient of the input tensor with respect to the output of the given model.

    Args:
        model (torch.nn.Module): The model to compute the gradient with respect to.
        input_tensor (torch.Tensor): The input tensor to compute the gradient for.
        loss_func (callable, optional): The loss function to use. If None, the gradient is computed with respect to the output tensor.
        targets (torch.Tensor, optional): The target tensor to use with the loss function. Defaults to None.
        metric (callable, optional): The metric function to use with the loss function. Defaults to None.
        out_num (int, optional): The index of the output tensor to compute the gradient with respect to. Defaults to 1.
        norm (bool, optional): Whether to normalize the attribution map. Defaults to False.
        device (str, optional): The device to use for computation. Defaults to 'cpu'.
    
    Returns:
        torch.Tensor: The attribution map computed as the gradient of the input tensor with respect to the output tensor.
    """
    # Set model.train() at the beginning and revert back to original mode (model.eval() or model.train()) at the end
    train_mode = model.training
    if not train_mode:
        model.train()
    
    input_tensor.requires_grad = True # Set requires_grad attribute of tensor. Important for computing gradients
    model.zero_grad() # Zero gradients
    inpt = input_tensor
    # Forward pass
    train_out = model(inpt) # training outputs (no inference outputs in train mode)
    
    # train_out[1] = torch.Size([4, 3, 80, 80, 7]) HxWx(#anchorxC) cls (class probabilities)
    # train_out[0] = torch.Size([4, 3, 160, 160, 7]) HxWx(#anchorx4) reg (location and scaling)
    # train_out[2] = torch.Size([4, 3, 40, 40, 7]) HxWx(#anchorx1) obj (objectness score or confidence)
    
    n_attr_list, index_classes = [], []
    for i in range(len(input_tensor)):
        if len(targets_list[i]) > n_max_labels:
            targets_list[i] = targets_list[i][:n_max_labels]
        if targets_list[i].numel() != 0:
            # unique_classes = torch.unique(targets_list[i][:,1])
            class_numbers = targets_list[i][:,1]
            index_classes.append([[0, 1, 2, 3, 4, int(uc)] for uc in class_numbers])
            num_attrs = len(targets_list[i])
            # index_classes.append([0, 1, 2, 3, 4] + [int(uc + 5) for uc in unique_classes])
            # num_attrs = 1 #len(unique_classes)# if loss_func else len(targets_list[i])
            n_attr_list.append(num_attrs)
        else:
            index_classes.append([0, 1, 2, 3, 4])
            n_attr_list.append(0)
    
    targets_list_filled = [targ.clone().detach() for targ in targets_list]
    labels_len = [len(targets_list[ih]) for ih in range(len(targets_list))]
    max_labels = np.max(labels_len)
    max_index = np.argmax(labels_len)
    for i in range(len(targets_list)):
        # targets_list_filled[i] = targets_list[i]
        if len(targets_list_filled[i]) < max_labels:
            tlist = [targets_list_filled[i]] * math.ceil(max_labels / len(targets_list_filled[i]))
            targets_list_filled[i] = torch.cat(tlist)[:max_labels].unsqueeze(0)
        else:
            targets_list_filled[i] = targets_list_filled[i].unsqueeze(0)
    for i in range(len(targets_list_filled)-1,-1,-1):
        if targets_list_filled[i].numel() == 0:
            targets_list_filled.pop(i)
    targets_list_filled = torch.cat(targets_list_filled)
    
    n_img_attrs = len(input_tensor) if class_specific_attr else 1
    n_img_attrs = 1 if loss_func else n_img_attrs
    
    attrs_batch = []
    for i_batch in range(n_img_attrs):
        if loss_func:
            i_batch = max_index
        # inpt = input_tensor[i_batch].unsqueeze(0)
        # ##################################################################
        # model.zero_grad() # Zero gradients
        # train_out = model(inpt)  # training outputs (no inference outputs in train mode)
        # ##################################################################
        n_label_attrs = n_attr_list[i_batch] if class_specific_attr else 1
        # n_label_attrs = 1 if loss_func else n_label_attrs
        attrs_img = []
        for i_attr in range(n_label_attrs):
            if loss_func is None:
                grad_wrt = train_out[out_num]
                if class_specific_attr:
                    grad_wrt = train_out[out_num][:,:,:,:,index_classes[i_batch][i_attr]]
                grad_wrt_outputs = torch.ones_like(grad_wrt)
            else:
                # if class_specific_attr:
                #     targets = targets_list[:][i_attr]
                n_targets = len(targets_list[i_batch])
                target_indiv = targets_list_filled[:,i_attr] # batch image input
                # target_indiv = targets_list[i_batch][i_attr].unsqueeze(0) # single image input
                # target_indiv[:,0] = 0 # this indicates the batch index of the target, should be 0 since we are only doing one image at a time
                loss, loss_items = loss_func(train_out, target_indiv, inpt, metric=metric)  # loss scaled by batch_size
                grad_wrt = loss
                grad_wrt_outputs = None
            
            model.zero_grad() # Zero gradients
            gradients = torch.autograd.grad(grad_wrt, inpt, 
                                                grad_outputs=grad_wrt_outputs, 
                                                retain_graph=True, 
                                                # create_graph=True, # Create graph to allow for higher order derivatives but slows down computation significantly
                                                )

            # Convert gradients to numpy array and back to ensure full separation from graph
            # attribution_map = torch.tensor(torch.sum(gradients[0], 1, keepdim=True).clone().detach().cpu().numpy())
            attribution_map = gradients[0].clone().detach() # without converting to numpy
            
            if grayscale: # Convert to grayscale, saves vram and computation time for plaus_eval
                attribution_map = torch.sum(attribution_map, 1, keepdim=True)
            if abs:
                attribution_map = torch.abs(attribution_map) # Take absolute values of gradients
                # attribution_map = np.sum(gradients, axis=0) # Sum across color channels
            if norm:
                # to improve accuracy, normalize outside of this look for all images in batch
                attribution_map = normalize_batch(attribution_map) # Normalize attribution map
                # imshow(attr, save_path='figs/attr')
            # else:
            #     attribution_map = gradients
            attrs_img.append(attribution_map)
        if len(attrs_img) == 0:
            attrs_batch.append((torch.zeros_like(inpt).unsqueeze(0)).to(device))
        else:
            attrs_batch.append(torch.stack(attrs_img).to(device))

    # out_attr = torch.tensor(attribution_map).unsqueeze(0).to(device) if ((loss_func) or (not class_specific_attr)) else torch.stack(attrs_batch).to(device)
    # out_attr = [attrs_batch[0]] * len(input_tensor) if ((loss_func) or (not class_specific_attr)) else attrs_batch
    out_attr = attrs_batch
    # Set model back to original mode
    if not train_mode:
        model.eval()
    
    return out_attr


def eval_plausibility(imgs, targets, attr_tensor, n_max_labels, device='cpu', debug=False):
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
    eval_totals = torch.tensor(0.0).to(device)
    plaus_num_nan = 0
    
    targets_ = targets
    for i in range(len(targets_)):
        if len(targets_[i]) > n_max_labels:
            targets_[i] = targets_[i][:n_max_labels]

    ## attr_tensor[img_num][attr_num][img_num] 
    # # first img_num if for attributions generated from that images targets
    # # second img_num is for all images in batch, but we only care about the one that matches the first img_num
    eval_data_all_attr = []
    for i, im0 in enumerate(imgs):
        eval_individual_data = []
        # for i_attr in range(len(attr_tensor[i % len(attr_tensor)])):
        if len(targets_[i]) == 0:
            eval_individual_data.append([torch.tensor(0).to(device),]) 
        else:
            IoU_list = []
            for j in range(len(targets_[i])):
                # t0 = time.time()
                if not targets_[i].numel() == 0:
                    xyxy_pred = targets_[i][j][2:] # * torch.tensor([im0.shape[2], im0.shape[1], im0.shape[2], im0.shape[1]])
                    xyxy_center = corners_coords(xyxy_pred) * torch.tensor([im0.shape[1], im0.shape[2], im0.shape[1], im0.shape[2]])
                    c1, c2 = (int(xyxy_center[0]), int(xyxy_center[1])), (int(xyxy_center[2]), int(xyxy_center[3]))
                    # might be faster to normalize when generating attrs, but this will normalize across entire batch, causing plaus score discrepancy
                    attr = attr_tensor[i % len(attr_tensor)][j][i].clone().detach()
                    # different attr was generated for each target, indexed by j (i_attr) ^^
                    if torch.isnan(attr).any():
                        attr = torch.nan_to_num(attr, nan=0.0)
                    if debug:
                        test_bbox = torch.zeros_like(im0)
                        test_bbox[:, c1[1]:c2[1], c1[0]:c2[0]] = im0[:, c1[1]:c2[1], c1[0]:c2[0]]
                        imshow(test_bbox, save_path='figs/test_bbox')
                        imshow(im0, save_path='figs/im0')
                        imshow(attr, save_path='figs/attr')
                    IoU_num = (torch.sum(attr[:,c1[1]:c2[1], c1[0]:c2[0]]))
                    IoU_denom = torch.sum(attr)
                    IoU_ = (IoU_num / IoU_denom)
                    IoU = IoU_
                else:
                    IoU = torch.tensor(0.0).to(device)
                IoU_list.append(IoU.clone().detach())
            list_mean = torch.mean(torch.tensor(IoU_list))
            # eval_totals += (list_mean / float(len(attr_tensor[i]))) if len(attr_tensor[i]) > 0 else 0.0 # must be changed to this if doing multiple individual class attribution maps
            eval_totals += ((list_mean / float(len(attr_tensor[i % len(attr_tensor)]))) / float(len(imgs)))
            eval_individual_data.append(IoU_list)
        eval_data_all_attr.append(eval_individual_data)
    
    return eval_totals.clone().detach().requires_grad_(True)#, eval_data_all_attr


def corners_coords(center_xywh):
    center_x, center_y, w, h = center_xywh
    x = center_x - w/2
    y = center_y - h/2
    return torch.tensor([x, y, x+w, y+h])
    
def normalize_batch(x):
    """
    Normalize a batch of tensors along each channel.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        
    Returns:
        torch.Tensor: Normalized tensor of the same shape as the input.
    """
    mins = torch.zeros((x.shape[0], *(1,)*len(x.shape[1:])), device=x.device)
    maxs = torch.zeros((x.shape[0], *(1,)*len(x.shape[1:])), device=x.device)
    for i in range(x.shape[0]):
        mins[i] = x[i].min()
        maxs[i] = x[i].max()
    x_ = (x - mins) / (maxs - mins)
    
    return x_