import csv
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange

with open('./mbti.tsv', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    labels = []
    posts = []
    for label, post in tqdm(reader, desc="Reading File"):
        labels.append(label)
        posts.append(post)

posts_train, posts_val, labels_train, labels_val = train_test_split(posts, labels, test_size=0.2, random_state=0)

with open('./mbti_train.tsv', 'w') as train_file:
    writer = csv.writer(train_file, delimiter='\t')
    for i in trange(len(posts_train), desc="Writing to Train File"):
        writer.writerow([labels_train[i], posts_train[i]])

with open('./mbti_val.tsv', 'w') as val_file:
    writer = csv.writer(val_file, delimiter='\t')
    for i in trange(len(posts_val), desc="Writing to Validation File"):
        writer.writerow([labels_val[i], posts_val[i]])
