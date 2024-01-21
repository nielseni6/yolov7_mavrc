import torch
import numpy as np
from plot_functs import * 
from plot_functs import imshow
from plot_functs import normalize_tensor
import math   
import time
import matplotlib.path as mplPath
from matplotlib.path import Path

def get_gradient(img, grad_wrt, norm=False, absolute=True, grayscale=True, keepmean=False):
    """
    Compute the gradient of an image with respect to a given tensor.

    Args:
        img (torch.Tensor): The input image tensor.
        grad_wrt (torch.Tensor): The tensor with respect to which the gradient is computed.
        norm (bool, optional): Whether to normalize the gradient. Defaults to True.
        absolute (bool, optional): Whether to take the absolute values of the gradients. Defaults to True.
        grayscale (bool, optional): Whether to convert the gradient to grayscale. Defaults to True.
        keepmean (bool, optional): Whether to keep the mean value of the attribution map. Defaults to False.

    Returns:
        torch.Tensor: The computed attribution map.

    """
    grad_wrt_outputs = torch.ones_like(grad_wrt)
    gradients = torch.autograd.grad(grad_wrt, img, 
                                    grad_outputs=grad_wrt_outputs, 
                                    retain_graph=True,
                                    # create_graph=True, # Create graph to allow for higher order derivatives but slows down computation significantly
                                    )
    attribution_map = gradients[0]
    if absolute:
        attribution_map = torch.abs(attribution_map) # Take absolute values of gradients
    if grayscale: # Convert to grayscale, saves vram and computation time for plaus_eval
        attribution_map = torch.sum(attribution_map, 1, keepdim=True)
    if norm:
        if keepmean:
            attmean = torch.mean(attribution_map)
            attmin = torch.min(attribution_map)
            attmax = torch.max(attribution_map)
        attribution_map = normalize_batch(attribution_map) # Normalize attribution maps per image in batch
        if keepmean:
            attribution_map -= attribution_map.mean()
            attribution_map += (attmean / (attmax - attmin))
        
    return attribution_map

def get_gaussian(img, grad_wrt, norm=True, absolute=True, grayscale=True, keepmean=False):
    """
    Generate Gaussian noise based on the input image.

    Args:
        img (torch.Tensor): Input image.
        grad_wrt: Gradient with respect to the input image.
        norm (bool, optional): Whether to normalize the generated noise. Defaults to True.
        absolute (bool, optional): Whether to take the absolute values of the gradients. Defaults to True.
        grayscale (bool, optional): Whether to convert the noise to grayscale. Defaults to True.
        keepmean (bool, optional): Whether to keep the mean of the noise. Defaults to False.

    Returns:
        torch.Tensor: Generated Gaussian noise.
    """
    
    gaussian_noise = torch.randn_like(img)
    
    if absolute:
        gaussian_noise = torch.abs(gaussian_noise) # Take absolute values of gradients
    if grayscale: # Convert to grayscale, saves vram and computation time for plaus_eval
        gaussian_noise = torch.sum(gaussian_noise, 1, keepdim=True)
    if norm:
        if keepmean:
            attmean = torch.mean(gaussian_noise)
            attmin = torch.min(gaussian_noise)
            attmax = torch.max(gaussian_noise)
        gaussian_noise = normalize_batch(gaussian_noise) # Normalize attribution maps per image in batch
        if keepmean:
            gaussian_noise -= gaussian_noise.mean()
            gaussian_noise += (attmean / (attmax - attmin))
        
    return gaussian_noise
    

def get_plaus_score(imgs, targets_out, attr, debug=False):
    """
    Calculates the plausibility score based on the given inputs.

    Args:
        imgs (torch.Tensor): The input images.
        targets_out (torch.Tensor): The output targets.
        attr (torch.Tensor): The attribute tensor.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.

    Returns:
        torch.Tensor: The plausibility score.
    """
    
    target_inds = targets_out[:, 0].int()
    xyxy_batch = targets_out[:, 2:6]# * pre_gen_gains[out_num]
    num_pixels = torch.tile(torch.tensor([imgs.shape[2], imgs.shape[3], imgs.shape[2], imgs.shape[3]], device=imgs.device), (xyxy_batch.shape[0], 1))
    # num_pixels = torch.tile(torch.tensor([1.0, 1.0, 1.0, 1.0], device=imgs.device), (xyxy_batch.shape[0], 1))
    xyxy_corners = (corners_coords_batch(xyxy_batch) * num_pixels).int()
    co = xyxy_corners
    coords_map = torch.zeros_like(attr, dtype=torch.bool)
    # rows = np.arange(co.shape[0])
    x1, x2 = co[:,1], co[:,3]
    y1, y2 = co[:,0], co[:,2]
    
    for ic in range(co.shape[0]): # potential for speedup here with torch indexing instead of for loop
        coords_map[target_inds[ic], :,x1[ic]:x2[ic],y1[ic]:y2[ic]] = True
    
    if torch.isnan(attr).any():
        attr = torch.nan_to_num(attr, nan=0.0)
    if debug:
        for i in range(len(coords_map)):
            coords_map3ch = torch.cat([coords_map[i][:1], coords_map[i][:1], coords_map[i][:1]], dim=0)
            test_bbox = torch.zeros_like(imgs[i])
            test_bbox[coords_map3ch] = imgs[i][coords_map3ch]
            imshow(test_bbox, save_path='figs/test_bbox')
            imshow(imgs[i], save_path='figs/im0')
            imshow(attr[i], save_path='figs/attr')
    
    IoU_num = (torch.sum(attr[coords_map]))
    IoU_denom = torch.sum(attr)
    IoU_ = (IoU_num / IoU_denom)
    plaus_score = IoU_

    return plaus_score


def point_in_polygon(poly, grid):
    # t0 = time.time()
    num_points = poly.shape[0]
    j = num_points - 1
    oddNodes = torch.zeros_like(grid[..., 0], dtype=torch.bool)
    for i in range(num_points):
        cond1 = (poly[i, 1] < grid[..., 1]) & (poly[j, 1] >= grid[..., 1])
        cond2 = (poly[j, 1] < grid[..., 1]) & (poly[i, 1] >= grid[..., 1])
        cond3 = (grid[..., 0] - poly[i, 0]) < (poly[j, 0] - poly[i, 0]) * (grid[..., 1] - poly[i, 1]) / (poly[j, 1] - poly[i, 1])
        oddNodes = oddNodes ^ (cond1 | cond2) & cond3
        j = i
    # t1 = time.time()
    # print(f'point in polygon time: {t1-t0}')
    return oddNodes
    
def point_in_polygon_gpu(poly, grid):
    num_points = poly.shape[0]
    i = torch.arange(num_points)
    j = (i - 1) % num_points
    # Expand dimensions
    # t0 = time.time()
    poly_expanded = poly.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, grid.shape[0], grid.shape[0])
    # t1 = time.time()
    cond1 = (poly_expanded[i, 1] < grid[..., 1]) & (poly_expanded[j, 1] >= grid[..., 1])
    cond2 = (poly_expanded[j, 1] < grid[..., 1]) & (poly_expanded[i, 1] >= grid[..., 1])
    cond3 = (grid[..., 0] - poly_expanded[i, 0]) < (poly_expanded[j, 0] - poly_expanded[i, 0]) * (grid[..., 1] - poly_expanded[i, 1]) / (poly_expanded[j, 1] - poly_expanded[i, 1])
    # t2 = time.time()
    oddNodes = torch.zeros_like(grid[..., 0], dtype=torch.bool)
    cond = (cond1 | cond2) & cond3
    # t3 = time.time()
    # efficiently perform xor using gpu and avoiding cpu as much as possible
    c = []
    while len(cond) > 1: 
        if len(cond) % 2 == 1: # odd number of elements
            c.append(cond[-1])
            cond = cond[:-1]
        cond = torch.bitwise_xor(cond[:int(len(cond)/2)], cond[int(len(cond)/2):])
    for c_ in c:
        cond = torch.bitwise_xor(cond, c_)
    oddNodes = cond
    # t4 = time.time()
    # for c in cond:
    #     oddNodes = oddNodes ^ c
    # print(f'expand time: {t1-t0} | cond123 time: {t2-t1} | cond logic time: {t3-t2} |  bitwise xor time: {t4-t3}')
    # print(f'point in polygon time gpu: {t4-t0}')
    # oddNodes = oddNodes ^ (cond1 | cond2) & cond3
    return oddNodes


def bitmap_for_polygon(poly, h, w):
    y = torch.arange(h).to(poly.device).float()
    x = torch.arange(w).to(poly.device).float()
    grid_y, grid_x = torch.meshgrid(y, x)
    grid = torch.stack((grid_x, grid_y), dim=-1)
    bitmap = point_in_polygon(poly, grid)
    return bitmap.unsqueeze(0)


def corners_coords(center_xywh):
    center_x, center_y, w, h = center_xywh
    x = center_x - w/2
    y = center_y - h/2
    return torch.tensor([x, y, x+w, y+h])

def corners_coords_batch(center_xywh):
    center_x, center_y = center_xywh[:,0], center_xywh[:,1]
    w, h = center_xywh[:,2], center_xywh[:,3]
    x = center_x - w/2
    y = center_y - h/2
    return torch.stack([x, y, x+w, y+h], dim=1)
    
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

####################################################################################
#### ALL FUNCTIONS BELOW ARE DEPRECIATED AND WILL BE REMOVED IN FUTURE VERSIONS ####
####################################################################################

def generate_vanilla_grad(model, input_tensor, loss_func = None, 
                          targets_list=None, targets=None, metric=None, out_num = 1, 
                          n_max_labels=3, norm=True, abs=True, grayscale=True, 
                          class_specific_attr = True, device='cpu'):    
    """
    Generate vanilla gradients for the given model and input tensor.

    Args:
        model (nn.Module): The model to generate gradients for.
        input_tensor (torch.Tensor): The input tensor for which gradients are computed.
        loss_func (callable, optional): The loss function to compute gradients with respect to. Defaults to None.
        targets_list (list, optional): The list of target tensors. Defaults to None.
        metric (callable, optional): The metric function to evaluate the loss. Defaults to None.
        out_num (int, optional): The index of the output tensor to compute gradients with respect to. Defaults to 1.
        n_max_labels (int, optional): The maximum number of labels to consider. Defaults to 3.
        norm (bool, optional): Whether to normalize the attribution map. Defaults to True.
        abs (bool, optional): Whether to take the absolute values of gradients. Defaults to True.
        grayscale (bool, optional): Whether to convert the attribution map to grayscale. Defaults to True.
        class_specific_attr (bool, optional): Whether to compute class-specific attribution maps. Defaults to True.
        device (str, optional): The device to use for computation. Defaults to 'cpu'.
    
    Returns:
        torch.Tensor: The generated vanilla gradients.
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
    
    if class_specific_attr:
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
        if loss_func and class_specific_attr:
            i_batch = max_index
        # inpt = input_tensor[i_batch].unsqueeze(0)
        # ##################################################################
        # model.zero_grad() # Zero gradients
        # train_out = model(inpt)  # training outputs (no inference outputs in train mode)
        # ##################################################################
        n_label_attrs = n_attr_list[i_batch] if class_specific_attr else 1
        n_label_attrs = 1 if not class_specific_attr else n_label_attrs
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
                # n_targets = len(targets_list[i_batch])
                if class_specific_attr:
                    target_indiv = targets_list_filled[:,i_attr] # batch image input
                else:
                    target_indiv = targets
                # target_indiv = targets_list[i_batch][i_attr].unsqueeze(0) # single image input
                # target_indiv[:,0] = 0 # this indicates the batch index of the target, should be 0 since we are only doing one image at a time
                    
                try:
                    loss, loss_items = loss_func(train_out, target_indiv, inpt, metric=metric)  # loss scaled by batch_size
                except:
                    target_indiv = target_indiv.to(device)
                    inpt = inpt.to(device)
                    for tro in train_out:
                        tro = tro.to(device)
                    print("Error in loss function, trying again with device specified")
                    loss, loss_items = loss_func(train_out, target_indiv, inpt, metric=metric)
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
            attribution_map = gradients[0]#.clone().detach() # without converting to numpy
            
            if grayscale: # Convert to grayscale, saves vram and computation time for plaus_eval
                attribution_map = torch.sum(attribution_map, 1, keepdim=True)
            if abs:
                attribution_map = torch.abs(attribution_map) # Take absolute values of gradients
            if norm:
                attribution_map = normalize_batch(attribution_map) # Normalize attribution maps per image in batch
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

