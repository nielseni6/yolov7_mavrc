from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline

coco = COCO("/data/nielseni6/coco/annotations/instances_val2017.json")
img_dir = "/data/nielseni6/coco/images/val2017/"
image_id = 37777
img = coco.imgs[image_id]

image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
# plt.imshow(image, interpolation='nearest')
# plt.show()
# plt.savefig('figs/test.png')

# plt.imshow(image)
cat_ids = coco.getCatIds()
anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
anns = coco.loadAnns(anns_ids[0:1])
# coco.showAnns(anns)
# plt.savefig('figs/test.png')

# mask = coco.annToMask(anns[0])
# for i in range(len(anns)):
#     mask += coco.annToMask(anns[i])
# plt.imshow(mask)
# plt.savefig('figs/test.png')

# from matplotlib.collections import PatchCollection
# from matplotlib.patches import Polygon


seg=anns[0]['segmentation'][0]



# p = PatchCollection([Polygon(poly),], facecolor=[[1.0,1.0,1.0]], linewidths=-1)
# ax = plt.gca()
# ax.set_autoscale_on(False)

# print(p)

import matplotlib.path as mplPath
import torch

# Let's assume poly is your numpy array containing polygon points
# poly = np.array([[x1, y1], [x2, y2], ..., [xn, yn]])

poly = np.array(seg).reshape((int(len(seg)/2), 2))
height, width, channels = image.shape
img = np.zeros((height, width, channels))
poly_path = mplPath.Path(poly)
y, x = np.mgrid[:height, :width]
inside = poly_path.contains_points(np.vstack((x.flatten(), y.flatten())).T)
inside = inside.reshape((height, width))
img[inside] = 1
plt.imshow(img)
plt.savefig('figs/test.png')
print(img)