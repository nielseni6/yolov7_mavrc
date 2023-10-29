# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 20:21:48 2023

@author: ianni
"""

import pandas as pd
import matplotlib.pyplot as plt

# file_path = './csv/evalattai_eval1_AddAttr__inc1_nsamples3_grayscale__robust__img_num9018_random_fullgrad_grad_eigen_eigengrad'
# file_path = './csv/plausibility_eval1_AddAttr__inc1_nsamples3_grayscale__robust__img_num9018_random_fullgrad_grad_eigen_eigengrad'
# file_path = '/home/nielseni6/PythonScripts/yolov7_mavrc/runs/test4/plausibility_eval1_AddAttr__inc1_nsamples1_grayscale__robust__img_num9018_random_vanilla_grad/plausibility_eval1_AddAttr__inc1_nsamples1_grayscale__robust__img_num9018_random_vanilla_grad'
# file_path = '/home/nielseni6/PythonScripts/yolov7_mavrc/runs/test1/evalattai_eval1_AddAttr__inc1_nsamples2_grayscale__robust__img_num9018_random_grad'
# file_path = '/home/nielseni6/PythonScripts/yolov7_mavrc/runs/test4/evalattai_eval1_AddAttr__inc1_nsamples1_grayscale__robust__img_num9018_random_vanilla_grad/evalattai_eval1_AddAttr__inc1_nsamples1_grayscale__robust__img_num9018_random_vanilla_grad'
# file_path = '/home/nielseni6/PythonScripts/yolov7_mavrc/runs/test4/plausibility_eval1_AddAttr__inc1_nsamples25_grayscale__robust__img_num9018_random_fullgrad_gradcam_eigen_eigengrad_vanilla_grad/plausibility_eval1_AddAttr__inc1_nsamples25_grayscale__robust__img_num9018_random_fullgrad_gradcam_eigen_eigengrad_vanilla_grad'
file_path = '/home/nielseni6/PythonScripts/yolov7_mavrc/runs/test4/evalattai_eval1_AddAttr__inc1_nsamples10_grayscale__robust__img_num9018_random_fullgrad_gradcam_eigen_eigengrad_vanilla_grad/evalattai_eval1_AddAttr__inc1_nsamples10_grayscale__robust__img_num9018_random_fullgrad_gradcam_eigen_eigengrad_vanilla_grad'


def plot_plaus_faith(file_path):
    # Load the CSV file
    data = pd.read_csv(str(file_path + '.csv'), header=None)
    data_CI = pd.read_csv(str(file_path + '_CI.csv'), header=None)
    
    print(file_path)
    # Set the x values (iterations)
    x = range(1, 11)
    
    # Create a figure and axis
    fig, ax = plt.subplots()
    
    with open(str(file_path + '.txt'), 'r') as f:
        method_names = [line.strip() for line in f]
    print(method_names)
    
    # method_names = ['random', 'fullgrad', 'grad', 'eigen', 'eigengrad',]
    
    if 'evalattai' in file_path:

        # print('len(method_names):', len(method_names))
        # print('data.iloc', data.iloc)
        # Plot each method
        for i in range(len(method_names)):
            # print('i:', i)
            ax.plot(x, data.iloc[i], label=method_names[i])
            ax.fill_between(x, data.iloc[i] - float(data_CI.iloc[i]), data.iloc[i] + float(data_CI.iloc[i]), alpha=0.4)
            # ax.fill_between(x, data.iloc[i] - 0.5, data.iloc[i] + 0.5, alpha=0.4)
        
        # Set labels and title
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.set_title('Loss vs Iterations for Different Methods')
        
        # Add a legend
        ax.legend()
        
        # Show the plot
        # plt.show()
        plt.savefig(str(str(file_path) + ".png"))
        
    elif 'plausibility' in file_path:
        # Replace all zeros with NaN
        data = data.replace(0, pd.NaT)
        
        # Drop all NaN values
        data = data.dropna(axis=1)
        
        data.index = method_names
        data_CI.index = method_names
        # data.columns = ['Attribution Methods']
        # data_CI.columns = ['Attribution Methods']
        data.columns = [' ']
        data_CI.columns = [' ']
        
        data = data.transpose()
        data_CI = data_CI.transpose()
        # # Reset the index
        # data.reset_index(inplace=True)
        # data_CI.reset_index(inplace=True)
        
        print(data)
        print(data_CI)
        
        # Create a bar plot
        ax = data.plot(kind='bar', yerr=(data_CI), capsize=5)
        # plt.bar(data.index, data, yerr=data_CI)
        
        # Add labels and title
        plt.xlabel('Method')
        plt.ylabel('IoU Average')
        plt.title('Plausibility Bar Plot')
        
        # print(data.columns)
        # Set x-tick labels to column names
        # ax.set_xticklabels(method_names, rotation=45, horizontalalignment='right')
        
        # Show the plot
        # plt.show()
        plt.savefig(str(str(file_path) + ".png"))
        
    else:
        print('File neither evalattai nor plausibility')

# plot_plaus_faith(file_path)