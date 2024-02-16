import torch

from plaus_functs import get_center_coords, get_distance_grids
from plot_functs import imshow
from torchvision.transforms.functional import gaussian_blur

attr = (gaussian_blur(torch.rand(2, 1, 100, 100)**12, 3)**4)

# print(attr)
dist_grid = get_distance_grids(attr, coords=torch.tensor([[5, 3],[2, 1]]))

imshow(dist_grid, save_path='figs/dist_grid')
imshow(attr[0], save_path='figs/test_map')

co = get_center_coords(attr)
