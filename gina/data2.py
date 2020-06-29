import torch
import csv
from transformers import BertTokenizerFast

def encode_label(label):
    label_dict = {
        "E": 1, "I": 0, "N": 0, "S": 1,
        "T": 1, "F": 0, "P": 1, "J": 0
    }
    return label_dict[label]


def dataloader(file, tokenizer):
    with open(file) as f:
        reader = csv.reader(f, delimiter='\t')
        read = list(reader)

    label, text, mask = [], [], []
    for r in read:
        binary = [c for c in r[0]]
        for b in binary:
            label.append(encode_label(b))
        sen = '[CLS]'*3 + r[1]
        token = tokenizer.encode_plus(sen,max_length=512,pad_to_max_length=True, return_attention_mask=True)
        text.append(token['input_ids'])
        mask.append(token['attention_mask'])

    text = torch.tensor(text)
    label = torch.tensor(label).view(-1,4)
    mask = torch.tensor(mask)
    # print(label.size(),mask.size(),text.size())

    return label, text, mask
