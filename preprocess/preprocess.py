import sys
import re
import csv

def preprocess_post(post):
    post = post.lower()
    post = re.sub(r"'$", '', post)
    post = re.sub(r"^'", '', post)
    post = re.sub(r'http[^ ]*( |$)', '__URL__ ', post)
    post = re.sub(r'([\.~!\(\)\[\]:])', r' \1 ', post)
    post = re.sub(r'([a-z])\.', r' \1 .', post)

    post = re.sub(r' +', ' ', post)
    post = re.sub(r' $', '', post)

    return post
    
with open(sys.argv[1]) as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        label = row[0]
        post_str = row[1]
        posts = post_str.split('|||')
        for post in posts:
            preprocessed = preprocess_post(post)
            print(f'{label}\t{preprocessed}')

