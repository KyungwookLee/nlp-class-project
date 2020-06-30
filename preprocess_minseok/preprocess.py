import sys
import re
import csv
from itertools import groupby
from tqdm import tqdm

def preprocess_posts(posts):
    posts = posts.strip("'")
    posts = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', posts)
    posts = posts.replace('|||', ' sep ')
    posts = re.sub('[^a-zA-Z?!\'\" -]', '', posts)             # remove non-alphabets
    posts = re.sub(' +', ' ', posts).lower().strip()    # remove space > 1
    posts = ' '.join([i[0] for i in groupby(posts.split())])
    # posts = posts.replace(' sep ', '. ')                # run this line for reformer (reformer has no sep token)
    prefix, suffix = 'sep ', ' sep'
    if posts.startswith(prefix):
        posts = posts[len(prefix):]
    if posts.endswith(suffix):
        posts = posts[:-len(suffix)]
    return posts

# per post
# with open("../data/mbti_1.csv", "r") as f:
#     reader = csv.reader(f)
#     next(reader)
#     for label, post_data in reader:
#         posts = post_data.split('|||')
#         for post in posts:
#             preprocessed = preprocess_post(post)
#             if len(preprocessed) == 0:
#                 continue
#             print(f'{label}\t{preprocessed}')

# per person
with open("../data/mbti_1.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)
    for label, post_data in tqdm(reader, desc="Preprocessing..."):
        preprocessed = preprocess_posts(post_data)
        if len(preprocessed) == 0:
            continue
        print(f'{label}\t{preprocessed}')

