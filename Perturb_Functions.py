# -*- coding: utf-8 -*-
# VDP
"""
Created on Mon Apr  6 03:07:41 2020

@author: ianni
"""

import torch
#import torchvision
#from torchvision import transforms
#import torchvision.datasets as datasets
#import matplotlib.pyplot as plt
#from model import MNIST_Model
import numpy as np
#import SmoothGrad as sg

#from captum.attr import (
#    GradientShap,
#    DeepLift,
#    DeepLiftShap,
#    IntegratedGradients,
#    LayerConductance,
#    NeuronConductance,
#    NoiseTunnel,
#    GuidedBackprop,
#)

#criterion = torch.nn.CrossEntropyLoss()
#model = MNIST_Model()
#
#model.load_state_dict(torch.load("model.dth"))
#model.eval()

def attack_data_format(img, attribution, n_steps, epsilon = 0.1, add_or_subtract_attr=True, normalize=True, tensor=True):
    
    attribution_numpy = attribution.clone().detach().cpu().numpy()
    if normalize:
        vmin = np.min(attribution_numpy)
        image_2d = attribution_numpy - vmin
        vmax = np.max(image_2d)
        attribution_numpy = (image_2d / vmax)
        mean_attr = np.mean(attribution_numpy, axis=None)
        attribution_numpy = attribution_numpy - 0.5#mean_attr
    
    pert = attribution_numpy
    #pert = np.sign(attribution_numpy)
    
    perturb = pert * epsilon
    
    formatted_image = img.clone().detach().cpu().numpy()
    vmin = np.min(formatted_image)
    vmax = np.max(formatted_image)    
    
    if add_or_subtract_attr == False:
        perturb = -perturb
    
    for i in range(n_steps):
        formatted_image = formatted_image + perturb
    
    #formatted_image = np.clip(formatted_image, a_min = vmin, a_max = vmax)
    #vmin = np.min(formatted_image)
    #image_2d = formatted_image - vmin
    #vmax = np.max(image_2d)
    #formatted_image = (image_2d / vmax)
    
    if tensor == True:
        formatted_image = torch.tensor(formatted_image)
    
    return formatted_image

def selective_faith_format(img, attribution, t, epsilon = 0.1, add_or_subtract_attr=True, Remove_High_or_Low_Pixels = True, Deletion_or_Insertion = True, normalize=True, tensor=True):
    
    Remain_Pixels, Remove_Pixels = list(), list()
    # t = percentage of pixels modified
    image = img.clone().detach().cpu().numpy()
    attribution_numpy = attribution.cpu().clone().detach().numpy()
    if normalize:
        vmin = np.min(attribution_numpy)
        image_2d = attribution_numpy - vmin
        vmax = np.max(image_2d)
        attribution_numpy = (image_2d / vmax)
        mean_attr = np.mean(attribution_numpy, axis=None)
        attribution_numpy = attribution_numpy - 0.5
    
    for i in range(attribution.size()[0]):
        r, g, b = image[i][0], image[i][1], image[i][2]
        r_mean, g_mean, b_mean = np.mean(r,axis=None), np.mean(g,axis=None), np.mean(b,axis=None)
        total_mean = [r_mean, g_mean, b_mean]
        # Normalize Attribution Map between 0 and 1
        attribution_numpy_norm1 = attribution[i].cpu().clone().detach().numpy()#.cpu().numpy()
        vmin = np.min(attribution_numpy_norm1)
        vmax_ = np.max(attribution_numpy_norm1)
        #print('Max: ', vmax_)
        #print('Min: ', vmin)
        attribution_numpy_norm = abs(attribution_numpy_norm1)
        
        # Sort attribution map from least to most significant pixels
        sorted_ranking = np.argsort(attribution_numpy_norm, axis=None)
        sorted_attributions = np.sort(attribution_numpy_norm, axis=None)
        if Remove_High_or_Low_Pixels == False:
            if Deletion_or_Insertion:
                replacement_spots = sorted_ranking[:round(len(sorted_attributions)*(t))]
            else:
                replacement_spots = sorted_ranking[:round(len(sorted_attributions)*(1-t))]
        else:
            if Deletion_or_Insertion:
                replacement_spots = sorted_ranking[round(len(sorted_attributions)*(1-t)):]
            else:
                replacement_spots = sorted_ranking[round(len(sorted_attributions)*(t)):]
        t_mean = 0.5#(np.mean(image[i],axis=None))
        mean_loc = 0
        
        f_image = image
        remaining_pixels = np.zeros_like(attribution_numpy_norm).astype(np.float)#image[0]
        removed_pixels = np.ones_like(attribution_numpy_norm).astype(np.float)
        for irt, rs in enumerate(replacement_spots):
            size = np.shape(attribution_numpy_norm)[1] # Size of image rows
            ch = int(rs / (size * size)) # channel
            rs = rs - (ch * (size * size))
            x = int(np.floor(rs / size))
            y = int(rs - (x * size))
            removed_pixels[ch][x][y] = 0.0
            remaining_pixels[ch][x][y] = 1.0
            if add_or_subtract_attr == False:
                f_image[0][ch][x][y] = f_image[0][ch][x][y] - (attribution_numpy[0][ch][x][y] * epsilon)
            else:
                f_image[0][ch][x][y] = f_image[0][ch][x][y] + (attribution_numpy[0][ch][x][y] * epsilon)
            
            ## Should be changed to use np.clip(in_array, a_min = 0, a_max = 1) ##
            if remaining_pixels[ch][x][y] >= 1:
                remaining_pixels[ch][x][y] = 1.0
            if remaining_pixels[ch][x][y] <= 0:
                remaining_pixels[ch][x][y] = 0.0
            ######################################################################
        Remain_Pixels.append(remaining_pixels.astype(np.float))
        Remove_Pixels.append(removed_pixels.astype(np.float))
    
    #formatted_image = f_image
    rem_pix = np.array([remaining_pixels.astype(np.float)])
    formatted_attr = rem_pix * (attribution_numpy * epsilon)
    
################################################################################
    
#    #pert = np.sign(attribution_numpy)

    if add_or_subtract_attr == False:
        formatted_attr = -formatted_attr
    
    formatted_image = image + formatted_attr

#    if t >= 0.99:
#        pert = attribution_numpy
#        perturb = pert * epsilon
#        formatted_image = img.clone().detach().cpu().numpy()
#        if add_or_subtract_attr == False:
#            perturb = -perturb
#        
#        formatted_image = formatted_image + perturb
    
    if tensor == True:
        formatted_image = torch.tensor(formatted_image)
    
    return formatted_image

def perturb_data_format(img, 
                     attribution,
                     t, 
                     snr, model, 
                     background = 'gaussian', # or 'saltandpepper' or 'black' or 'white'
                     criterion = torch.nn.CrossEntropyLoss(), Deletion_or_Insertion = True, Remove_High_or_Low_Pixels = True, remove_or_add_noise = True, replace_w_mean_or_noise=False, tensor=True,
                     ):
    
    Remain_Pixels, Remove_Pixels = list(), list()
    # t = percentage of pixels modified
    var = torch.linalg.norm(img).cpu() / snr
    image = img.cpu().detach().numpy()
    attribution_numpy = attribution.cpu().clone().detach().numpy()
    
    for i in range(attribution.size()[0]):
        r, g, b = image[i][0], image[i][1], image[i][2]
        r_mean, g_mean, b_mean = np.mean(r,axis=None), np.mean(g,axis=None), np.mean(b,axis=None)
        total_mean = [r_mean, g_mean, b_mean]
        # Normalize Attribution Map between 0 and 1
        attribution_numpy = attribution[i].cpu().clone().detach().numpy()#.cpu().numpy()
        vmin = np.min(attribution_numpy)
        image_2d = attribution_numpy - vmin
        vmax = np.max(image_2d)
        attribution_numpy = (image_2d / vmax)
        
        # Sort attribution map from least to most significant pixels
        sorted_ranking = np.argsort(attribution_numpy, axis=None)
        sorted_attributions = np.sort(attribution_numpy, axis=None)
        #print(sorted_attributions)
        if Remove_High_or_Low_Pixels == False:
            if Deletion_or_Insertion:
                replacement_spots = sorted_ranking[:round(len(sorted_attributions)*(t))]
            else:
                replacement_spots = sorted_ranking[:round(len(sorted_attributions)*(1-t))]
        else:
            if Deletion_or_Insertion:
                replacement_spots = sorted_ranking[round(len(sorted_attributions)*(1-t)):]
            else:
                replacement_spots = sorted_ranking[round(len(sorted_attributions)*(t)):]
        t_mean = 0.5#(np.mean(image[i],axis=None))
        mean_loc = 0
        if remove_or_add_noise:
            mean_loc = t_mean
        gaussian_noise = np.random.normal(loc = mean_loc, scale = var, size = np.shape(attribution_numpy))
        if background == 'saltandpepper':
            gaussian_noise = np.where(gaussian_noise > mean_loc, 1, 0)
        
        remaining_pixels = np.zeros_like(attribution_numpy)
        removed_pixels = np.ones_like(attribution_numpy)
        for irt, rs in enumerate(replacement_spots):
            size = np.shape(attribution_numpy)[1] # Size of image rows
            ch = int(rs / (size * size)) # channel
            rs = rs - (ch * (size * size))
            x = int(np.floor(rs / size))
            y = int(rs - (x * size))
            if remove_or_add_noise:
                removed_pixels[ch][x][y] = 0
                if replace_w_mean_or_noise:
                    remaining_pixels[ch][x][y] = total_mean[ch]
                else:
                    remaining_pixels[ch][x][y] = gaussian_noise[ch][x][y]
            else:
                remaining_pixels[ch][x][y] = remaining_pixels[ch][x][y] + gaussian_noise[ch][x][y]
            ## Should be changed to use np.clip(in_array, a_min = 0, a_max = 1) ##
            #if remaining_pixels[ch][x][y] >= 1:
            #    remaining_pixels[ch][x][y] = 1
            #if remaining_pixels[ch][x][y] <= 0:
            #    remaining_pixels[ch][x][y] = 0
            ######################################################################
        Remain_Pixels.append(remaining_pixels)
        Remove_Pixels.append(removed_pixels)
    
    formatted_image = image * Remove_Pixels
    formatted_image = formatted_image + Remain_Pixels

    if tensor == True:
        formatted_image = torch.tensor(formatted_image)
#        
    return formatted_image
    
def pixel_masks(img, 
                attribution,
                t,
                tensor=True,
                ):
    
    Remain_Pixels, Remove_Pixels = list(), list()
    
    # t = percentage of pixels modified
    image = img.clone().detach().cpu().numpy()
    for i in range(attribution.size()[0]):
        # Normalize Attribution Map between 0 and 1
        attribution_numpy = attribution[i].clone().numpy()
        vmin = np.min(attribution_numpy)
        image_2d = attribution_numpy - vmin
        vmax = np.max(image_2d)
        attribution_numpy = (image_2d / vmax)
        
        # Sort attribution map from least to most significant pixels
        sorted_attributions = np.sort(attribution_numpy, axis=None)
        t_mean = np.mean(image[i],axis=None)
        t_max = sorted_attributions[round(len(sorted_attributions)*(1-t))]
        remaining_pixels = np.zeros_like(attribution_numpy)
        remaining_pixels[attribution_numpy > t_max] = t_mean
        removed_pixels = np.ones_like(attribution_numpy)
        removed_pixels[attribution_numpy > t_max] = 0
        Remain_Pixels.append(remaining_pixels)
        Remove_Pixels.append(removed_pixels)
    
    if tensor == True:
        Remain_Pixels = torch.tensor(Remain_Pixels)
        Remove_Pixels = torch.tensor(Remove_Pixels)
    
    return Remain_Pixels, Remove_Pixels

def returnGradArray(img1):
    
    nsamples = 3
    stdev_spread=0.5
    magnitude = True
    classification = list()
    
    img = img1.detach().requires_grad_(True).cpu()
    pred = model(img)
    for i, p in enumerate(pred):
        classification.append(torch.max(p.data, 0)[1])
#        print('i: ', int(i))
    predictions = torch.tensor(classification)
    loss = criterion(pred, predictions)
    loss.backward()
    
    Sc_dx = img.grad
#    x_np = img1.numpy()
#    stdev = stdev_spread * (np.max(x_np) - np.min(x_np))
#    
#    total_gradients = torch.tensor(np.zeros_like(img1))
#    for i in range(nsamples):
#        noise = np.random.normal(0, stdev, img1.shape)
#        x_plus_noise = x_np + noise
#        x_noise_tensor = torch.tensor(x_plus_noise, dtype = torch.float32)
#        
#        x_noise_tensor.requires_grad_(True)
#        pred = model(x_noise_tensor)
#        loss = criterion(pred, predictions)
#        loss.backward()
#        gradient = x_noise_tensor.grad
#        
#        if magnitude:
#            total_gradients += (gradient * gradient)
#        else:
#            total_gradients += gradient
#    
#    Sc_dx = total_gradients / nsamples
    
    # If error switch gradient and prediction
    return Sc_dx#, predictions

def returnRandMaskArray(img1):

#    image = img1.detach()
    sz = img1.size()
    random_mask = torch.tensor(np.random.rand(sz[0],sz[1],sz[2],sz[3]),dtype=torch.float32)# + (image * 0)
    
    return random_mask

def returnGradPred(img):
    
    img.requires_grad_(True)
    pred = model(img)
    loss = criterion(pred, torch.tensor([int(torch.max(pred[0], 0)[1])]))
    loss.backward()
    
#    S_c = torch.max(pred[0].data, 0)[0]
    Sc_dx = img.grad
    
    return Sc_dx, pred

def returnGradPredIndex(img, index):
    
    img.requires_grad_(True)
    pred = model(img)
    loss = criterion(pred, torch.tensor([int(index)]))
    loss.backward()
    
#    S_c = torch.max(pred[0].data, 0)[0]
    Sc_dx = img.grad
    
    return Sc_dx, pred

def returnGrad(img):
    
    img.requires_grad_(True)
    pred = model(img)
    loss = criterion(pred, torch.tensor([int(torch.max(pred[0], 0)[1])]))
    loss.backward()
    
#    S_c = torch.max(pred[0].data, 0)[0]
    Sc_dx = img.grad
    
    return Sc_dx