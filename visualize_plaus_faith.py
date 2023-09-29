# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 20:21:48 2023

@author: ianni
"""

import pandas as pd
import matplotlib.pyplot as plt

# file_path = './csv/evalattai_eval1_AddAttr__inc1_nsamples3_grayscale__robust__img_num9018_random_fullgrad_grad_eigen_eigengrad'
# file_path = './csv/plausibility_eval1_AddAttr__inc1_nsamples3_grayscale__robust__img_num9018_random_fullgrad_grad_eigen_eigengrad'
# file_path = '/home/nielseni6/PythonScripts/yolov7_mavrc/runs/test1/evalattai_eval1_AddAttr__inc1_nsamples15_grayscale__robust__img_num9018_random_fullgrad_grad_eigen_eigengrad'
file_path = '/home/nielseni6/PythonScripts/yolov7_mavrc/runs/test1/evalattai_eval1_AddAttr__inc1_nsamples2_grayscale__robust__img_num9018_random_grad'

def plot_plaus_faith(file_path):
    # Load the CSV file
    data = pd.read_csv(str(file_path + '.csv'), header=None)
    print(data)
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
        
        data = data.transpose()
        
        data.columns = method_names
        
        # Create a bar plot
        data.plot(kind='bar')
        
        # Add labels and title
        plt.xlabel('Method')
        plt.ylabel('IoU Average')
        plt.title('Plausibility Bar Plot')
        
        # Show the plot
        # plt.show()
        plt.savefig(str(str(file_path) + ".png"))
        
    else:
        print('File neither evalattai nor plausibility')

# plot_plaus_faith(file_path)