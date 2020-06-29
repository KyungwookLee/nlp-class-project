import torch
import csv

def encode_label(label):
    label_dict = {
        "ISTJ": 0, "ISFJ": 1, "INFJ": 2, "INTJ": 3,
        "ISTP": 4, "ISFP": 5, "INFP": 6, "INTP": 7,
        "ESTP": 8, "ESFP": 9, "ENFP": 10, "ENTP": 11,
        "ESTJ": 12, "ESFJ": 13, "ENFJ": 14, "ENTJ": 15
    }
    return label_dict[label]


def dataloader(file, tokenizer):
    with open(file) as f:
        reader = csv.reader(f, delimiter='\t')
        read = list(reader)
        # test = read[0][1]
        # print(read)

    label, text, mask = [], [], []
    for r in read:
        # print(r[0])
        label.append(encode_label(r[0]))
        token = tokenizer.encode_plus(r[1],max_length=512,pad_to_max_length=True, return_attention_mask=True)
        # print(token['attention_mask'])
        text.append(token['input_ids'])
        mask.append(token['attention_mask'])

    text = torch.tensor(text)
    label = torch.tensor(label)
    mask = torch.tensor(mask)

    return label, text, mask
