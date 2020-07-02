import csv
import re

from tqdm import tqdm

with open('data/100speeches.csv', 'r', encoding='utf-8', errors='ignore') as f:
    reader = csv.reader(f)
    next(reader)
    for label, speech in tqdm(reader, desc="Preprocessing..."):
        if label == '':
            continue
        speech = speech.replace("\n", "")
        speech = speech.replace(":", ": ")
        speech = speech.replace(";", "; ")
        speech = re.sub('[.?!]', ' sep ', speech)
        speech = re.sub(' +', ' ', speech).lower().strip()
        prefix, suffix = 'sep ', ' sep'
        if speech.startswith(prefix):
            speech = speech[len(prefix):]
        if speech.endswith(suffix):
            speech = speech[:-len(suffix)]
        print(f'{label}\t{speech}')
