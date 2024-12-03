import torch
from plaus_functs import get_center_coords, get_distance_grids, get_plaus_loss, get_bbox_map, normalize_batch
from plot_functs import imshow
from torchvision.transforms.functional import gaussian_blur
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# TODO - We need to modify this function to not be grey scale
def subfigimshow(img, ax):
    print(f'img shape: {img.shape}')
    try:
        npimg = img.clone().detach().cpu().numpy()
    except:
        npimg = img
    if len(npimg.shape) == 2:
        # If it's a 2D array, it's likely a grayscale image
        ax.imshow(npimg, cmap='gray')
    elif len(npimg.shape) == 3:
        if npimg.shape[0] == 3 or npimg.shape[0] == 1:
            # If the first dimension is 3 or 1, it's likely in (C, H, W) format
            tpimg = np.transpose(npimg, (1, 2, 0))
        else:
            # It's already in (H, W, C) format
            tpimg = npimg
        
        if tpimg.shape[2] == 1:
            # If it's a 3D array with only one channel, squeeze it
            ax.imshow(np.squeeze(tpimg), cmap='gray')
        else:
            ax.imshow(tpimg)
    else:
        raise ValueError(f"Unexpected image shape: {npimg.shape}")

def draw_bounding_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    # Ensure image is 3-channel RGB
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    
    # Ensure image is uint8 and in range [0, 255]
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    
    image_with_boxes = image.copy()
    for box in boxes:
        x_center, y_center, width, height = box
        x_min = int((x_center - width / 2) * image_with_boxes.shape[1])
        y_min = int((y_center - height / 2) * image_with_boxes.shape[0])
        x_max = int((x_center + width / 2) * image_with_boxes.shape[1])
        y_max = int((y_center + height / 2) * image_with_boxes.shape[0])
        cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), color, thickness)
    
    return image_with_boxes

#NOTE: We want to keep the number of bb at 0 for now (1). This is the number of targets we want to have not BB
def toy_problem(pgt_coeff, focus_coeff, x_coord, y_coord, num_bb=0, alpha=400.0, scheduler=2.0, device="0", dist_coeff=0.5, dist_reg_only=True, iou_coeff=0.5, 
                bbox_coeff=0.0, dist_x_bbox=False, iou_loss_only=False, show_dist_reg=True):
    
    # Create a Namespace object to hold params
    opt = argparse.Namespace()
    # Save all parameters as attributes of the Namespace object
    opt.pgt_coeff = pgt_coeff
    opt.focus_coeff = focus_coeff
    opt.x_coord = x_coord
    opt.y_coord = y_coord
    opt.num_bb = num_bb
    opt.alpha = alpha
    opt.scheduler = scheduler
    opt.device = device
    opt.dist_coeff = dist_coeff
    opt.dist_reg_only = dist_reg_only
    opt.iou_coeff = iou_coeff
    opt.bbox_coeff = bbox_coeff
    opt.dist_x_bbox = dist_x_bbox
    opt.iou_loss_only = iou_loss_only
    opt.show_dist_reg = show_dist_reg

    # Create a list of save dirs for output
    save_dirs = []

    # Set CUDA device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(opt.device))
    
    targets = torch.tensor([
                            [0, 0, opt.x_coord, opt.y_coord, 0.05, 0.05],
    #                        [0, 1, 0.4, 0.6, 0.05, 0.07],
    #                        [1, 0, 0.25, 0.2, 0.04, 0.05],
                        #    [2, 0, 0.8, 0.76, 0.05, 0.05],
                        #    [2, 0, 0.8, 0.2, 0.05, 0.05],
                        #    [0, 0, 0.8, 0.76, 0.05, 0.05],
                        #    [1, 0, 0.8, 0.2, 0.05, 0.05],
    ])
    
    unique_classes = torch.unique(targets[:,0])
    # X = (gaussian_blur(torch.rand(len(unique_classes), 1, 50, 50)**2, 3)**4)
    attr = (gaussian_blur(torch.rand(len(unique_classes), 1, 640, 640)**2, 13)**4).requires_grad_(True)
    plaus_loss = get_plaus_loss(targets, attribution_map=attr, 
                                                        opt=opt, 
                                                        debug=True,
                                                        only_loss=True)
    if opt.iou_loss_only:
        bbox_map = get_bbox_map(targets, attr)
        plaus_score = ((torch.sum((attr * bbox_map))) / (torch.sum(attr)))
        plaus_loss = (1.0 - plaus_score)

    nsamples = 10
    rows = len(attr)  # Number of images
    cols = nsamples + 2  # Define the number of columns for subplots
    size = 3

    # Create a new figure for each i
    fig1 = plt.figure(figsize=(cols * size, rows * size))
    plt.tight_layout()

    # Create the second figure for the remaining 8 attr steps
    fig2 = plt.figure(figsize=(cols * size, rows * size))
    plt.tight_layout()

    # Create a figure for plausibility scores
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    plaus_scores = []

    #THIS IS OLD CODE
    # for j in range(len(attr)):
    #     ax = fig.add_subplot(rows, cols, 2 + (j * cols))
    #     if j == 0:
    #         ax.set_title('Attr Step 0')
    #     if j == len(attr) - 1:
    #         # ax.set_title(f'PGT Loss:\n{round(float(plaus_loss), 5)}')
    #         ax.set_title(f'Attr Step:\n{round(float(plaus_loss), 5)}')
    #     subfigimshow(attr[j], ax)  # Display the image
    #     ax.axis('off')
    
    for i in range(10):
        plaus_loss, (plaus_score, dist_reg, plaus_reg,), distance_map = get_plaus_loss(targets.requires_grad_(True), attribution_map=attr, opt=opt, debug=True)
        
        delta_attr = torch.autograd.grad(plaus_loss, attr, create_graph=True, retain_graph=True)[0] 
        attr = attr - (delta_attr * alpha) 
        alpha *= opt.scheduler

        plaus_loss, (plaus_score, dist_reg, plaus_reg,), distance_map = get_plaus_loss(targets, attribution_map=attr, opt=opt, debug=True)
        if opt.iou_loss_only:
            bbox_map = get_bbox_map(targets, attr)
            plaus_score = ((torch.sum((attr * bbox_map))) / (torch.sum(attr)))
            plaus_loss = (1.0 - plaus_score)
            distance_map = bbox_map

        # attr = attr.clamp(0, 1) 
        attr = normalize_batch(attr)
        plaus_scores.append(float(plaus_loss))
        print(f'step: {i}, plaus_loss: {plaus_loss}, plaus_score: {plaus_score}, dist_reg: {dist_reg}, plaus_reg: {plaus_reg}')

        for j in range(len(attr)):

            # Add a subplot for each image 
            if i == 0 and opt.show_dist_reg: 
                ax = fig1.add_subplot(rows, cols, 1 + (j * cols)) 
                ax.set_title(f'Distance Regularization Map {j}') 
                img_tensor = (1 - distance_map[j]).detach().cpu().squeeze(0)  # Assuming grayscale image
                img_np = img_tensor.numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                bbox_coords = targets[:, 2:6].detach().cpu().numpy()  # This gives us [x_coord, y_coord, width, height] (all bb for now)
                img_with_boxes = draw_bounding_boxes(img_np, bbox_coords)
                subfigimshow(img_with_boxes, ax) 
                ax.axis('off')
             
            else:  
                if i == 1:
                    # Add the first attr step to fig1
                    ax = fig1.add_subplot(rows, cols, 2 + (j * cols))
                    ax.set_title(f'Attr Step {i}' if j == 0 else '')
                    img_tensor = attr[j].detach().cpu().squeeze(0)  # Assuming grayscale image
                    img_np = img_tensor.numpy()
                    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                    bbox_coords = targets[:, 2:6].detach().cpu().numpy()  # This gives us [x_coord, y_coord, width, height] (all bb for now)
                    img_with_boxes = draw_bounding_boxes(img_np, bbox_coords)
                    subfigimshow(img_with_boxes, ax)
                    ax.axis('off')
                else:
                    # Subsequent steps go to fig2
                    ax = fig2.add_subplot(rows, cols, 1 + (i - 1) + (j * cols))
                    ax.set_title(f'Attr Step {i}' if j == 0 else '')
                    subfigimshow(attr[j], ax)
                    ax.axis('off')
    
    # Plot plausibility scores
    ax3.plot(range(nsamples), plaus_scores, marker='o', label='Plausibility Score')
    ax3.set_title('Plausibility Scores Across Steps')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Plausibility Score')
    ax3.grid(True)
    ax3.legend()

    # Save the figures
    fig1.savefig('figs/distance_and_first_step.png', bbox_inches='tight')
    plt.close(fig1)

    fig2.savefig('figs/remaining_attr_steps.png', bbox_inches='tight')
    plt.close(fig2)

    fig3.savefig('figs/plausibility_scores.png', bbox_inches='tight')
    plt.close(fig3)

    print('Figures saved: figs/distance_and_first_step.png, figs/remaining_attr_steps.png, and figs/plausibility_scores.png')
    return 'figs/distance_and_first_step.png', 'figs/remaining_attr_steps.png', 'figs/plausibility_scores.png'

if __name__ == '__main__':

    #TODO - this does not appear to be working correctly
    parser = argparse.ArgumentParser()
    # ##################### Standard Settings #####################
    parser.add_argument('--pgt_coeff', type=float, default=1.0, help='pgt_coeff')
    parser.add_argument('--focus_coeff', type=float, default=0.2, help='focus_coeff')
    parser.add_argument('--alpha', type=float, default=400.0, help='alpha')
    parser.add_argument('--num_bb', type=int, default=0, help='num_bb')
    parser.add_argument('--x_coord', type=float, default=0.2, help='x_coord')
    parser.add_argument('--y_coord', type=float, default=0.35, help='y_coord')
    ########################## Advanced #########################
    parser.add_argument('--scheduler', type=float, default=2.0, help='scheduler for alpha')
    #############################################################
    parser.add_argument('--device', type=str, default='0', help='device')
    parser.add_argument('--dist_coeff', type=float, default=0.5, help='dist_coeff')
    parser.add_argument('--dist_reg_only', type=bool, default=True, help='dist_reg_only')
    parser.add_argument('--iou_coeff', type=float, default=0.5, help='iou_coeff')
    parser.add_argument('--bbox_coeff', type=float, default=0.0, help='bbox_coeff')
    parser.add_argument('--dist_x_bbox', type=bool, default=False, help='dist_x_bbox')
    parser.add_argument('--iou_loss_only', type=bool, default=False, help='iou_loss_only')
    parser.add_argument('--show_dist_reg', type=bool, default=True, help='show distance regularization map in figure')
    opt = parser.parse_args()

    toy_problem(opt.pgt_coeff, opt.focus_coeff, opt.x_coord, opt.y_coord, opt.alpha, opt.num_bb, 
                opt.scheduler, opt.device, opt.dist_coeff, opt.dist_reg_only, opt.iou_coeff, 
                opt.bbox_coeff, opt.dist_x_bbox, opt.iou_loss_only, opt.show_dist_reg)