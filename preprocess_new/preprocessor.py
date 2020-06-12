import os
import random
import re
import csv
from tqdm import tqdm
from TweetNormalizer import normalizeTweet

def tokenize_emoji(post):
    return re.sub(": ([^:|]+) :", r":\1:", normalizeTweet(post))

random.seed(0)

# Normalize tokens
o_samples = []
s_samples_train, s_samples_val, s_samples_test = [], [], []
with open("../data/mbti.csv", newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in tqdm(reader, desc="Normalizing"):
        # Original posts
        o_samples.append((row[0], tokenize_emoji(row[1].replace('|||', ' SEP '))))
        
        # Split posts
        posts = row[1].split('|||')
        random.shuffle(posts)
        num_posts = len(posts)
        for post_train in posts[:int(0.8*num_posts)]:
            s_samples_train.append((row[0], tokenize_emoji(post_train)))
        for post_val in posts[int(0.8*num_posts):int(0.9*num_posts)]:
            s_samples_val.append((row[0], tokenize_emoji(post_val)))
        for post_test in posts[int(0.9*num_posts):]:
            s_samples_test.append((row[0], tokenize_emoji(post_test)))

num_samples = len(o_samples)
random.shuffle(o_samples)
random.shuffle(s_samples_train)
random.shuffle(s_samples_val)
random.shuffle(s_samples_test)

samples = {'original': {'all': o_samples, 
                        'train': o_samples[:int(0.8*num_samples)], 
                        'val': o_samples[int(0.8*num_samples):int(0.9*num_samples)], 
                        'test': o_samples[int(0.9*num_samples):]}, 
           'split': {'all': s_samples_train + s_samples_val + s_samples_test, 
                     'train': s_samples_train, 
                     'val': s_samples_val, 
                     'test': s_samples_test}
           }

# Write pre-processed data
for post_split in ['original', 'split']:
    for split in ['all', 'train', 'val', 'test']:
        os.makedirs(f'./{post_split}_{split}', exist_ok=True)
        f1 = open(os.path.join(f'./{post_split}_{split}/', 'mbti.tsv'), 'w')
        f2 = open(os.path.join(f'./{post_split}_{split}/', 'mbti_ei.tsv'), 'w')
        f3 = open(os.path.join(f'./{post_split}_{split}/', 'mbti_ns.tsv'), 'w')
        f4 = open(os.path.join(f'./{post_split}_{split}/', 'mbti_ft.tsv'), 'w')
        f5 = open(os.path.join(f'./{post_split}_{split}/', 'mbti_pj.tsv'), 'w')
        for sample in tqdm(samples[post_split][split], desc=f"Writing {post_split}_{split}_mbti files"):
            f1.write(f'{sample[0]}\t{sample[1]}' + '\n')
            f2.write(f'{sample[0][0]}\t{sample[1]}' + '\n')   
            f3.write(f'{sample[0][1]}\t{sample[1]}' + '\n')   
            f4.write(f'{sample[0][2]}\t{sample[1]}' + '\n') 
            f5.write(f'{sample[0][3]}\t{sample[1]}' + '\n')     
        f1.close()
        f2.close()
        f3.close()
        f4.close()
        f5.close()