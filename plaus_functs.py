import torch
import numpy as np
from plot_functs import * 
from plot_functs import normalize_tensor
import math   
import time

def generate_vanilla_grad(model, input_tensor, loss_func = None, 
                          targets=None, targets_list=None, metric=None, out_num = 1, 
                          norm=False, class_specific_attr = True, device='cpu'):    
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
    # Forward pass
    train_out = model(input_tensor) # training outputs (no inference outputs in train mode)
    
    # train_out[1] = torch.Size([4, 3, 80, 80, 7]) HxWx(#anchorxC) cls (class probabilities)
    # train_out[0] = torch.Size([4, 3, 160, 160, 7]) HxWx(#anchorx4) reg (location and scaling)
    # train_out[2] = torch.Size([4, 3, 40, 40, 7]) HxWx(#anchorx1) obj (objectness score or confidence)
    
    n_attr_list, index_classes = [], []
    for i in range(len(input_tensor)):
        if targets_list[i].numel() != 0:
            unique_classes = torch.unique(targets_list[i][:,1])
            # index_classes.append([[int(uc), -4, -3, -2, -1] for uc in unique_classes])
            index_classes.append([int(uc) for uc in unique_classes] + [-4, -3, -2, -1])
            num_attrs = 1 #len(unique_classes)# if loss_func else len(targets_list[i])
            n_attr_list.append(num_attrs)
        else:
            index_classes.append([-4, -3, -2, -1])
            n_attr_list.append(0)
        
    n_img_attrs = len(input_tensor) if class_specific_attr else 1
    n_img_attrs = 1 if loss_func else n_img_attrs
    
    attrs_batch = []
    for i_batch in range(n_img_attrs):
        n_label_attrs = n_attr_list[i_batch] if class_specific_attr else 1
        n_label_attrs = 1 if loss_func else n_label_attrs
        attrs_img = []
        for i_attr in range(n_label_attrs):
            if loss_func is None:
                grad_wrt = train_out[out_num]
                if class_specific_attr:
                    grad_wrt = train_out[out_num][:,:,:,:,index_classes[i_batch]]
                grad_wrt_outputs = torch.ones_like(grad_wrt)
            else:
                # if class_specific_attr:
                #     targets = targets_list[:][i_attr]
                loss, loss_items = loss_func(train_out, targets.to(device), input_tensor, metric=metric)  # loss scaled by batch_size
                grad_wrt = loss
                grad_wrt_outputs = None
                # loss.backward(retain_graph=True, create_graph=True)
                # gradients = input_tensor.grad
            
            gradients = torch.autograd.grad(grad_wrt, input_tensor, 
                                                grad_outputs=grad_wrt_outputs, 
                                                retain_graph=True, 
                                                # create_graph=True, # Create graph to allow for higher order derivatives but slows down computation significantly
                                                )

            gradients = gradients[0].detach().cpu().numpy() # Convert gradients to numpy array
            
            if norm:
                attribution_map = np.absolute(gradients) # Take absolute values of gradients
                # attribution_map = np.sum(gradients, axis=0) # Sum across color channels
                attribution_map = normalize_numpy(attribution_map) # Normalize attribution map
                # imshow(attr, save_path='figs/attr')
            else:
                attribution_map = gradients
            attrs_img.append(torch.tensor(attribution_map))
        if len(attrs_img) == 0:
            attrs_batch.append(torch.zeros_like(input_tensor).to(device))
        else:
            attrs_batch.append(torch.stack(attrs_img).to(device))

    # out_attr = torch.tensor(attribution_map).unsqueeze(0).to(device) if ((loss_func) or (not class_specific_attr)) else torch.stack(attrs_batch).to(device)
    out_attr = [torch.tensor(attribution_map).to(device),] if ((loss_func) or (not class_specific_attr)) else attrs_batch
    # Set model back to original mode
    if not train_mode:
        model.eval()
    
    return out_attr


def eval_plausibility(imgs, targets, attr_tensor, device, debug=False):
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
    eval_totals = 0.0
    plaus_num_nan = 0
    
    targets_ = targets
    # for il in range(len(targets_)):
    #     try:
    #         targets_[il] = torch.stack(targets_[il])
    #     except:
    #         targets_[il] = torch.tensor([[]])
    # targets_ = [[targets[i] for i in range(len(targets)) if int(targets[i][0]) == j] for j in range(len(imgs))]
    eval_data_all_attr = []
    for i, im0 in enumerate(imgs):
        eval_individual_data = []
        for i_attr in range(len(attr_tensor[i])):
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
                        attr = normalize_tensor(torch.abs(attr_tensor[i][i_attr][i].clone().detach()))
                        if torch.isnan(attr).any():
                            attr = torch.nan_to_num(attr, nan=0.0)
                        # if True:
                        #     test_bbox = torch.zeros_like(im0)
                        #     test_bbox[:, c1[1]:c2[1], c1[0]:c2[0]] = im0[:, c1[1]:c2[1], c1[0]:c2[0]]
                        #     imshow(test_bbox, save_path='figs/test_bbox')
                        #     imshow(im0, save_path='figs/im0')
                        #     imshow(attr, save_path='figs/attr')
                        IoU_num = (torch.sum(attr[:,c1[1]:c2[1], c1[0]:c2[0]]))
                        IoU_denom = torch.sum(attr)
                        IoU_ = (IoU_num / IoU_denom)
                        if debug:
                            iou_isnan = torch.isnan(IoU_)
                            if iou_isnan:
                                IoU = torch.tensor([0.0]).to(device)
                                plaus_num_nan += 1
                            else:
                                IoU = IoU_
                        else:
                            IoU = IoU_
                    else:
                        IoU = torch.tensor(0.0).to(device)
                    IoU_list.append(IoU.clone().detach().cpu())
                list_mean = torch.mean(torch.tensor(IoU_list))
                # eval_totals += (list_mean / float(len(attr_tensor[i]))) if len(attr_tensor[i]) > 0 else 0.0 # must be changed to this if doing multiple individual class attribution maps
                eval_totals += list_mean
                eval_individual_data.append(IoU_list)
        eval_data_all_attr.append(eval_individual_data)
    
    
    if debug:
        return torch.tensor(eval_totals).requires_grad_(True), plaus_num_nan
    else:
        return torch.tensor(eval_totals).requires_grad_(True)#, eval_data_all_attr


def corners_coords(center_xywh):
    center_x, center_y, w, h = center_xywh
    x = center_x - w/2
    y = center_y - h/2
    return torch.tensor([x, y, x+w, y+h])
    
