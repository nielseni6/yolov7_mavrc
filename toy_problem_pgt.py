import torch

from plaus_functs import get_center_coords, get_distance_grids, get_plaus_loss, get_bbox_map, normalize_batch
from plot_functs import imshow
from torchvision.transforms.functional import gaussian_blur
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

def subfigimshow(img, ax):
    try:
        npimg = img.clone().detach().cpu().numpy()
    except:
        npimg = img
    tpimg = np.transpose(npimg, (1, 2, 0))
    ax.imshow(tpimg)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='0', help='device')
    parser.add_argument('--focus_coeff', type=float, default=0.3, help='focus_coeff')
    parser.add_argument('--dist_coeff', type=float, default=1.0, help='dist_coeff')
    parser.add_argument('--dist_reg_only', type=bool, default=True, help='dist_reg_only')
    parser.add_argument('--pgt_coeff', type=float, default=3.0, help='pgt_coeff')
    parser.add_argument('--alpha', type=float, default=500.0, help='alpha')
    parser.add_argument('--iou_coeff', type=float, default=0.1, help='iou_coeff')
    parser.add_argument('--bbox_coeff', type=float, default=0.0, help='bbox_coeff')
    parser.add_argument('--dist_x_bbox', type=bool, default=True, help='dist_x_bbox')
    parser.add_argument('--iou_loss_only', type=bool, default=False, help='iou_loss_only')
    opt = parser.parse_args() 
    print(opt) 
    
    # Set CUDA device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device 
    
    targets = torch.tensor([[0, 1, 0.4, 0.6, 0.05, 0.07],
                           [1, 0, 0.25, 0.2, 0.04, 0.05],
                           [2, 0, 0.8, 0.76, 0.05, 0.05],
                           [2, 0, 0.8, 0.2, 0.05, 0.05],])
    unique_classes = torch.unique(targets[:,0])
    # X = (gaussian_blur(torch.rand(len(unique_classes), 1, 50, 50)**2, 3)**4)
    attr = (gaussian_blur(torch.rand(len(unique_classes), 1, 640, 640)**2, 3)**4).requires_grad_(True)
    plaus_loss = get_plaus_loss(targets, attribution_map=attr, 
                                                        opt=opt, 
                                                        debug=True,
                                                        only_loss=True)
    if opt.iou_loss_only:
        bbox_map = get_bbox_map(targets, attr)
        plaus_score = ((torch.sum((attr * bbox_map))) / (torch.sum(attr)))
        plaus_loss = (1.0 - plaus_score)
    # else:
    #     plaus_loss = get_plaus_loss(targets, attr, opt, only_loss = True)

    nsamples = 10
    rows = len(attr)  # Number of images
    cols = nsamples + 2  # Define the number of columns for subplots
    size = 3

    # Create a new figure for each i
    fig = plt.figure(figsize=(cols * size, rows * size))
    plt.tight_layout()
    
    for j in range(len(attr)):
        ax = fig.add_subplot(rows, cols, 2 + (j * cols))
        if j == 0:
            ax.set_title('Attr Step 0')
        if j == len(attr) - 1:
            ax.set_title(f'PGT Loss:\n{round(float(plaus_loss), 5)}')
        subfigimshow(attr[j], ax)  # Display the image
        ax.axis('off')
    for i in range(10):
        plaus_loss, (plaus_score, dist_reg, plaus_reg,), distance_map = get_plaus_loss(targets, attribution_map=attr, opt=opt, debug=True)
        if opt.iou_loss_only:
            bbox_map = get_bbox_map(targets, attr)
            plaus_score = ((torch.sum((attr * bbox_map))) / (torch.sum(attr)))
            plaus_loss = (1.0 - plaus_score)
            distance_map = bbox_map
            
        delta_attr = torch.autograd.grad(plaus_loss, attr, create_graph=True,)[0] 
        attr = attr - (delta_attr * opt.alpha) 
        # attr = attr.clamp(0, 1) 
        attr = normalize_batch(attr) 
        print(f'step: {i}, plaus_loss: {plaus_loss}, plaus_score: {plaus_score}, dist_reg: {dist_reg}, plaus_reg: {plaus_reg}') 
        for j in range(len(attr)): 
            # Add a subplot for each image 
            if i == 0: 
                ax = fig.add_subplot(rows, cols, 1 + (j * cols)) 
                ax.set_title(f'Distance Regularization Map {j}') 
                subfigimshow(distance_map[j], ax) 
                ax.axis('off') 
                # imshow(distance_map[j], save_path=f'figs/dist_grid{j}') 
            ax = fig.add_subplot(rows, cols, 3 + (j * cols) + i) 
            if j == 0: 
                ax.set_title(f'Attr Step {i + 1}') 
            if j == len(attr) - 1: 
                ax.set_title(f'PGT Loss:\n{round(float(plaus_loss), 5)}') 
            subfigimshow(attr[j], ax)  # Display the image
            ax.axis('off') 
            # imshow(attr[j], save_path=f'figs/test_map{j}_{i}')
    # Save the full figure
    save_path = 'figs/toy_problem_pgt'
    fig.savefig(f'{save_path}.png', bbox_inches='tight')
    fig.savefig(f'{save_path}.pdf', bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory
    print(f'plaus_loss: {plaus_loss}, plaus_score: {plaus_score}, dist_reg: {dist_reg}, plaus_reg: {plaus_reg}')
    print(f'saved as: {save_path}.png')
# for i in range(10):
#     delta_attr = torch.autograd.grad(plaus_loss, attr, retain_graph=True,)[0]
#     or j in range(len(attr)):
#         imshow(attr[j], save_path=f'figs/test_map{j}_{i}')