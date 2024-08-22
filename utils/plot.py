import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt

import re

def sort_key(path):
    match = re.search(r'_batch_(\d+)', path)
    return int(match.group(1)) if match else float('inf')


def plot_matrix_size(file_paths, images_path, ylabel):

    data_matrix = []

    for file_path in file_paths:
        df = pd.read_csv(file_path, header=None)
        data_matrix.append(df.values.flatten())

    if ylabel == 'Memory (GB)':
        data_matrix = (1/1024*np.array(data_matrix)).round(2)
    else:
        data_matrix = np.array(data_matrix).round(2)
    print(data_matrix)


    fig, ax = plt.subplots()

    x = [1, 2, 3, 4, 5]


    ax.scatter(x, data_matrix[0], color='#1f77b4', marker='o', s=40, label='batch size = 8')
    ax.scatter(x, data_matrix[1], color='#ff7f0e', marker='s', s=40, label='batch size = 16')
    ax.scatter(x, data_matrix[2], color='#2ca02c', marker='^', s=40, label='batch size = 32')
    ax.scatter(x, data_matrix[3], color='#d62728', marker='v', s=40, label='batch size = 64')
    ax.scatter(x, data_matrix[4], color='#9467bd', marker='D', s=40, label='batch size = 128')
    ax.scatter(x, data_matrix[5], color='#8c564b', marker='x', s=40, label='batch size = 256')
    ax.scatter(x, data_matrix[6], color='#e377c2', marker='+', s=40, label='batch size = 512')

    ax.set_xlabel("Matrix Dimension")
    ax.set_ylabel(ylabel)

    ax.set_xticks([1, 2, 3, 4, 5])  
    ax.set_xticklabels(['400x400', '800x400', '1200x400', '1600x400', '2000x400'])  
    ax.grid(True, color='gray', linewidth=0.5)
    ax.legend()


    for spine in ax.spines.values():
        spine.set_edgecolor('#4D4D4D')  
        spine.set_linewidth(1)  

    plt.savefig(f"{images_path}.png")

def plot_batch_size(file_paths, images_path, ylabel):

    data_matrix = []

    for file_path in file_paths:
        df = pd.read_csv(file_path, header=None)
        data_matrix.append(df.values.flatten())

    if ylabel == 'Memory (GB)':
        data_matrix = (1/1024*np.array(data_matrix)).round(2)
    else:
        data_matrix = np.array(data_matrix).round(2)
    print(data_matrix)


    fig, ax = plt.subplots(figsize=(10, 6))

    x = [1, 2, 4, 8, 16, 32, 64]


    ax.scatter(x, data_matrix[:,0], color='#1f77b4', marker='o', s=40, label='matrix size = 400x400')
    ax.scatter(x, data_matrix[:,1], color='#ff7f0e', marker='s', s=40, label='matrix size = 800x400')
    ax.scatter(x, data_matrix[:,2], color='#2ca02c', marker='^', s=40, label='matrix size = 1200x400')
    ax.scatter(x, data_matrix[:,3], color='#d62728', marker='v', s=40, label='matrix size = 1600x400')
    ax.scatter(x, data_matrix[:,4], color='#9467bd', marker='D', s=40, label='matrix size = 2000x400')

    ax.set_xlabel("Batch Size")
    ax.set_ylabel(ylabel)

    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])  
    ax.set_xticklabels(['8', '16', '32', '64', '128', '256', '512'])  
    ax.grid(True, color='gray', linewidth=0.5)
    ax.legend()

    for spine in ax.spines.values():
        spine.set_edgecolor('#4D4D4D')  
        spine.set_linewidth(1)  

    plt.savefig(f"{images_path}.png")


if __name__ == '__main__':


    time_file_paths = sorted(glob.glob("results/matrix_size/time_batch_*.csv"), key=sort_key)
    memory_file_paths = sorted(glob.glob("results/matrix_size/memory_batch_*.csv"), key=sort_key)

    time_matrix_images = 'results/images/time_matrix_size'
    time_batch_images = 'results/images/time_batch_size'
    memory_matrix_images = 'results/images/memory_matrix_size'
    memory_batch_images = 'results/images/memory_batch_size'

    plot_batch_size(time_file_paths, time_batch_images, ylabel='Running time (s)')
    plot_batch_size(memory_file_paths, memory_batch_images, ylabel='Memory (GB)')

    plot_matrix_size(time_file_paths, time_matrix_images, ylabel='Running time (s)')
    plot_matrix_size(memory_file_paths, memory_matrix_images, ylabel='Memory (GB)')

    

    
