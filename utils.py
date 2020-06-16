import csv

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


def encode_label(label):
    label_dict = {
        "ISTJ": 0, "ISFJ": 1, "INFJ": 2, "INTJ": 3,
        "ISTP": 4, "ISFP": 5, "INFP": 6, "INTP": 7,
        "ESTP": 8, "ESFP": 9, "ENFP": 10, "ENTP": 11,
        "ESTJ": 12, "ESFJ": 13, "ENFJ": 14, "ENTJ": 15
    }
    return label_dict[label]


def load_tsv(tsv_file, tokenizer):
    posts, masks, labels = [], [], []
    max_len = 0
    with open(tsv_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for label, post in tqdm(reader, desc="Reading "+tsv_file):
            encoded_post = tokenizer.encode_plus(
                                post,
                                add_special_tokens=True,    # add [CLS] and [SEP] tokens 
                                max_length=128,             # pad & truncate to T=128
                                pad_to_max_length=True,
                                return_attention_mask=True, # construct attention masks
                                return_tensors='pt',        # return pytorch tensors
                           )
            posts.append(encoded_post['input_ids'])
            masks.append(encoded_post['attention_mask'])
            labels.append(encode_label(label))
    posts = torch.cat(posts, dim=0)
    masks = torch.cat(masks, dim=0)
    labels = torch.tensor(labels)
    return posts, masks, labels


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

