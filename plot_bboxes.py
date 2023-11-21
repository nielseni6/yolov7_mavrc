from utils.plots import plot_one_box, plot_one_box_PIL
from PIL import Image
import numpy as np
import cv2
from utils.general import xywh2xyxy, xywhn2xyxy, xyxy2xywh
import matplotlib.pyplot as plt
import random
# path="../../../data/Koutsoubn8/hex_data/cave/train/"
path='/data/Koutsoubn8/ijcnn_v7data/Real_world_test/'
data_name= "V_AIRPLANE_001_13"
img_path= path +"images/" + data_name + ".jpg"
label = path +"labels/" + data_name + ".txt"

img=np.array(Image.open(img_path))
img=cv2.resize(img,(480,480))
# img=Image.fromarray(img
# print(type(img))
# exit()
# print(img)
# print(img)
# exit()
gtruth=open(label,"r")
print(gtruth)
gtruth_info=gtruth.read()
print(gtruth_info)

def center_xy_coords(xywh):
    x,y,w,h = xywh
    center_x, center_y = (x+x+w)/2,(y+y+h)/2
    return np.array([center_x, center_y,w,h])

# exit()
gtruth_info=gtruth_info.replace('\n',"").split(" ")
coords_list=gtruth_info[1:]
coords=np.float32(coords_list)
print("coords",coords)
# coords=coords*0.25
# print("coords",coords)

# coords[1]=coords[1] * .25
# coords[3]=coords[3] * .25
# exit()

# coords=coords[[1,0,3,2]]
# coords[0]=coords[0]*1280
# coords[1]=coords[1]*720
# coords[2]=coords[2] *1280
# coords[3]=coords[3]*720
coords=coords*480


print("xywh",coords)
# exit()
# coords[0]=400   #y
# coords[1]=242   #x
# coords[2]=463   #w
# coords[3]=253   #h
# print("manual xywh", coords)




# [638,373,383,235]
# print(coords.shape)
# print(coords)

# coords[0]=coords[0]/1920
# coords[1]=coords[1]/1080
# coords[2]=coords[2]/1920
# coords[3]=coords[3]/1080


# coords=center_xy_coords(coords)
print(coords)
coords=coords.reshape((1,4))

# exit()
coords=xywh2xyxy((coords))
#
print("xyxy",coords)
coords=coords.reshape((4))
# exit()


plot_one_box(coords, img, color = (0,255,0) ,line_thickness=3)
img=Image.fromarray(img)
plt.imshow(img)
plt.savefig(("hex_coords.jpg"))
# img.save("hex_coords.jpg")

def plot_one_box_seg(x, img, color=None, label=None, line_thickness=-1, center_coords = True):
    def corners_coords(center_xywh):
        center_x, center_y, w, h = center_xywh
        x = center_x - w/2
        y = center_y - h/2
        return np.array([x, y, x+w, y+h])

    if center_coords:
        x = corners_coords(x) * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # if label:
    #     tf = max(tl - 1, 1)  # font thickness
    #     t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    #     c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    #     cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_one_box_PIL_(img, color=(255,255,255), img_size = 480, 
                     xyxy=None, line_thickness=-1):
    
    # gtruth_info=label.read()
    # print(gtruth_info)

    def center_xy_coords(xywh):
        x,y,w,h = xywh
        center_x, center_y = (x+x+w)/2,(y+y+h)/2
        return np.array([center_x, center_y,w,h])
    
    # exit()
    # gtruth_info=gtruth_info.replace('\n',"").split(" ")
    coords_list=xyxy[0][2:]
    coords = center_xy_coords(coords_list)
    # coords=np.float32(coords_list)
    # if coords.shape[0]==0:
    #     coords = coords_list.numpy()
    print("coords",coords)
    
    coords=coords*img_size
    print("xywh",coords)
    print(coords)
    try:
        coords=coords.reshape((1,4))
    except:
        print("coords has size 0 \n coords:",coords)
        print("coords_list:",coords_list)
        print("label:",label)
        return
        # exit()
        # break

    # exit()
    coords=xywh2xyxy((coords))
    #
    print("xyxy",coords)
    coords=coords.reshape((4))
    # exit()

    img = np.asarray(img, dtype=np.float32)
    plot_one_box_filled(coords, img, color = color, line_thickness=line_thickness)
    # img=Image.fromarray(img)
    # plt.imshow(img)
    # plt.savefig(("hex_coords.jpg"))
    
    return img


def plot_one_box_filled(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
