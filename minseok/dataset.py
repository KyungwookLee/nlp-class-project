import csv
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler

from tqdm import tqdm, trange


class MBTIDataset(object):

    def __init__(self, tsv_file, tokenizer):

        self.posts = []
        self.labels = []
        self.tokenizer = tokenizer
        max_length = 512

        with open(tsv_file, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for label, posts in tqdm(reader, desc="Reading " + tsv_file):
                if len(posts) < 3:
                    continue
                posts_i = ['[CLS]']
                posts = posts.split(' sep ')    # varies on dataset
                for post in posts:
                    posts_i += self.tokenizer.tokenize(post) + ['[SEP]']
                posts_i = self.tokenizer.convert_tokens_to_ids(posts_i)[:max_length]

                # don't just truncate to max_length. Truncate to the last SEP token.
                if len(posts_i) == max_length and posts_i[-1] != 102:   # 102 is reserved for SEP token id
                    last_sep_idx = next(i for i in reversed(range(len(posts_i))) if posts_i[i] == 102)
                    posts_i = posts_i[:last_sep_idx+1]

                self.posts.append(posts_i)
                self.labels.append(encode_label(label))


    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (self.posts[index], self.labels[index])


def encode_label(label):
    label_dict = {
        "ISTJ": 0, "ISFJ": 1, "INFJ": 2, "INTJ": 3,
        "ISTP": 4, "ISFP": 5, "INFP": 6, "INTP": 7,
        "ESTP": 8, "ESFP": 9, "ENFP": 10, "ENTP": 11,
        "ESTJ": 12, "ESFJ": 13, "ENFJ": 14, "ENTJ": 15
    }
    return label_dict[label]


def round_to_nearest_multiple(num, divisor):
    diff = num % divisor
    if diff != 0:
        return num - diff + divisor
    return num - diff


def mbti_collate_fn(samples):
    
    input_ids, labels = zip(*samples)
    attention_mask = [[1] * len(input_id) for input_id in input_ids]

    input_ids = pad_sequence([torch.Tensor(input_id).to(torch.long) for input_id in input_ids], \
                              padding_value=0, batch_first=True)
    attention_mask = pad_sequence([torch.Tensor(mask).to(torch.long) for mask in attention_mask], \
                                   padding_value=0, batch_first=True)

    # batch_size, dim = input_ids.size()
    # extra_len = 256 - (dim % 256)     # 256 is default for reformer
    # if extra_len != 0:              # set length to multiples of 256 (required for transformer)
    #     input_ids = torch.cat([input_ids, torch.zeros([batch_size, extra_len]).to(torch.long)], dim=1)
    #     attention_mask = torch.cat([attention_mask, torch.zeros([batch_size, extra_len]).to(torch.long)], dim=1)

    labels = torch.Tensor(labels).to(torch.long)

    return input_ids, attention_mask, labels


class MBTIBatchSampler(Sampler):

    def __init__(self, dataset: MBTIDataset, batch_size, shuffle=False):
        super().__init__(dataset)
        self.shuffle = shuffle

        _, indices = zip(*sorted((len(input_ids), index) for index, (input_ids, _) in enumerate(tqdm(dataset, desc="Bucketing"))))
        self.batched_indices = [indices[index: index+batch_size] for index in range(0, len(indices), batch_size)]
    
    def __len__(self):
        return len(self.batched_indices)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batched_indices)

        for batch in self.batched_indices:
            yield batch