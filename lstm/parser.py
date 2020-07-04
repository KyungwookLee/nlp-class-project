import os
import random
import csv
import re
from tqdm import tqdm

for split in ['train', 'val', 'test']:
    f1 = open("lstm.data/post_" + split + ".label", 'w')
    f2 = open("lstm.data/post_" + split + ".input", 'w')
    with open("../preprocess_new/split_" + split + "/mbti.tsv", newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in tqdm(reader):
            if row[1] != '':
                f1.write(row[0] + '\n')
                f2.write(row[1] + '\n')
    f1.close()
    f2.close()
    
    f1 = open("lstm.data/person_" + split + ".label", 'w')
    f2 = open("lstm.data/person_" + split + ".input", 'w')
    if split == 'val':
        with open("../gina/personal_data_valid.tsv", newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in tqdm(reader):
                if row[1] != '':
                    f1.write(row[0] + '\n')
                    f2.write(row[1] + '\n')
    else:
        with open("../gina/personal_data_" + split + ".tsv", newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in tqdm(reader):
                if row[1] != '':
                    f1.write(row[0] + '\n')
                    f2.write(row[1] + '\n')
    f1.close()
    f2.close()
    
    f1 = open("lstm.data/person_aug_" + split + ".label", 'w')
    f2 = open("lstm.data/person_aug_" + split + ".input", 'w')
    if split == 'val':
        with open("../gina/personal_data_aug_valid.tsv", newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in tqdm(reader):
                if row[1] != '':
                    f1.write(row[0] + '\n')
                    f2.write(row[1] + '\n')
    else:
        with open("../gina/personal_data_aug_" + split + ".tsv", newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in tqdm(reader):
                if row[1] != '':
                    f1.write(row[0] + '\n')
                    f2.write(row[1] + '\n')
    f1.close()
    f2.close()


