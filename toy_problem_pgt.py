import torch

from plaus_functs import get_center_coords, get_distance_grids, get_plaus_loss
from plot_functs import imshow
from torchvision.transforms.functional import gaussian_blur
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--focus_coeff', type=float, default=0.5, help='focus_coeff')
    parser.add_argument('--dist_coeff', type=float, default=100.0, help='dist_coeff')
    parser.add_argument('--dist_reg_only', type=bool, default=False, help='dist_reg_only')
    parser.add_argument('--pgt_coeff', type=float, default=1.0, help='pgt_coeff')
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha')
    parser.add_argument('--iou_coeff', type=float, default=0.075, help='iou_coeff')
    opt = parser.parse_args() 
    print(opt) 

    targets = torch.tensor([[0, 1, 0.4, 0.6, 0.05, 0.07],
                           [1, 0, 0.25, 0.2, 0.04, 0.05],])
    attr = (gaussian_blur(torch.rand(2, 1, 100, 100)**12, 3)**4).requires_grad_(True)
    for i in range(50):
        plaus_loss, (plaus_score, dist_reg, plaus_reg,), distance_map = get_plaus_loss(targets, attribution_map=attr, opt=opt, debug=True)
        
        delta_attr = torch.autograd.grad(plaus_loss, attr, retain_graph=True,)[0]
        attr = attr - (delta_attr * opt.alpha)
        attr = attr.clamp(0, 1)
        print(f'step: {i}, plaus_loss: {plaus_loss}, plaus_score: {plaus_score}, dist_reg: {dist_reg}, plaus_reg: {plaus_reg}')
        for j in range(len(attr)):
            if i == 0:
                imshow(distance_map[j], save_path=f'figs/dist_grid{j}')
            imshow(attr[j], save_path=f'figs/test_map{j}_{i}')

    print(f'plaus_loss: {plaus_loss}, plaus_score: {plaus_score}, dist_reg: {dist_reg}, plaus_reg: {plaus_reg}')
