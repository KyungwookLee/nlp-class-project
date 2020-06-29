import sys
import re
import csv

def preprocess_posts(posts):
    posts = posts.lower()
    posts = re.sub(r"^'", '', posts)
    posts = re.sub(r'http[^ ]*( |$)', '__URL__ ', posts)
    posts = re.sub(r'([\.~!\?\(\)\[\]:;])', r' \1 ', posts)
    posts = re.sub(r'(\|{3,})',r' \1',posts)
    posts = re.sub(r' +', ' ', posts)
    posts = re.sub(r"''", '', posts)

    return posts

augmentation = True

if not augmentation:
    with open(sys.argv[1]) as f:
        reader = csv.reader(f)
        next(reader)

        for i, row in enumerate(reader):
            # print(row)
            label = row[0]
            posts = row[1][0:-1]
            preprocessed = preprocess_posts(posts)
            preprocessed = re.sub(r'\|{3,}','[SEP] ', preprocessed)
            # preprocessed = '[CLS] '+preprocessed
            print(f'{label}\t{preprocessed}')


#data augmentation
if augmentation:
    with open(sys.argv[1]) as f:
        reader = csv.reader(f)
        next(reader)

        for i, row in enumerate(reader):
            # print(row)
            label = row[0]
            posts = row[1][0:-1]
            preprocessed = preprocess_posts(posts)
            preprocessed = re.sub(r'\|{3,}','[SEP] ', preprocessed)
            sub = preprocessed.split('[SEP]')
            sub_l = len(sub)
            for j in range(4):
                preprocessed = '[SEP]'.join(sub[j*sub_l//4:(j+1)*sub_l//4])
                print(f'{label}\t{preprocessed}')
