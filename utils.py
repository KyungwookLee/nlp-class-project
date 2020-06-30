import csv

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_values(train_values, val_values, title):
    x = list(range(1, len(train_values)+1))
    plt.figure()
    plt.title(title)
    plt.plot(x, train_values, marker='o', label='Training')
    plt.plot(x, val_values, marker='x', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.legend()
    plt.savefig(title + '.png')

