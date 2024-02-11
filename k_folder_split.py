import os
import numpy as np
import shutil

root_dir = '/data/nielseni6/drone_data/'

for i in range(10):
    if not os.path.isdir(root_dir + f'k_fold{i}'):
        os.makedirs(root_dir + f'k_fold{i}')
    if not os.path.isdir(root_dir + f'k_fold{i}/images'):
        os.makedirs(root_dir + f'k_fold{i}/images')
    if not os.path.isdir(root_dir + f'k_fold{i}/labels'):
        os.makedirs(root_dir + f'k_fold{i}/labels')

srcIm = root_dir+"Real_world_drone_data/images" # Folder to copy images from
srcLb = root_dir+"Real_world_drone_data/labels" # Folder to copy labels from

allFileNamesIm = os.listdir(srcIm)
allFileNamesLb = os.listdir(srcLb)

FileNamesAllIm = np.split(np.array(allFileNamesIm),
                    [int(len(allFileNamesIm)*0.1*i) for i in range(1,10)])
# FileNamesAllLb = np.split(np.array(allFileNamesLb),
#                     [int(len(allFileNamesLb)*0.1*i) for i in range(1,10)])

for i, FileNamesIm in enumerate(FileNamesAllIm):
    print(f'Total images in k_fold{i}: ', len(FileNamesIm))
    FileNamesIm = [srcIm+'/'+ name for name in FileNamesIm.tolist()]
    FileNamesLb = allFileNamesLb # FileNamesAllLb[i]
    FileNamesLb = [srcLb+'/'+ name for name in FileNamesLb]
    for name in FileNamesIm:
        shutil.copy(name, root_dir + f'k_fold{i}/images')
    for name in FileNamesLb:
        shutil.copy(name, root_dir + f'k_fold{i}/labels')

