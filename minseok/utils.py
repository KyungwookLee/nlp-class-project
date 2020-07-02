import csv

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


def decode_label(label):
    label_dict = {
        0: "ISTJ", 1: "ISFJ", 2: "INFJ", 3: "INTJ",
        4: "ISTP", 5: "ISFP", 6: "INFP", 7: "INTP",
        8: "ESTP", 9: "ESFP", 10: "ENFP", 11: "ENTP",
        12: "ESTJ", 13: "ESFJ", 14: "ENFJ", 15: "ENTJ"
    }
    return label_dict[label]


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

