# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
#import plotly.tools as tls
from time import time
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import cross_validate, StratifiedKFold, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier


start_load = time()
print("Data Loading.......")

### Person data
"""
train = np.loadtxt('../gina/personal_data_train.tsv', dtype=str, delimiter='\t')
#train = np.concatenate((train, np.array([cls[0] for cls in train[:,0]])[:,None], np.array([cls[1] for cls in train[:,0]])[:,None], np.array([cls[2] for cls in train[:,0]])[:,None], np.array([cls[3] for cls in train[:,0]])[:,None]), axis=1)
train = pd.DataFrame(train, columns=['type', 'post']) #, 'EI', 'NS', 'FT', 'PJ'])
val = np.loadtxt('../gina/personal_data_valid.tsv', dtype=str, delimiter='\t')
#val = np.concatenate((val, np.array([cls[0] for cls in val[:,0]])[:,None], np.array([cls[1] for cls in val[:,0]])[:,None], np.array([cls[2] for cls in val[:,0]])[:,None], np.array([cls[3] for cls in val[:,0]])[:,None]), axis=1)
val = pd.DataFrame(val, columns=['type', 'post']) #, 'EI', 'NS', 'FT', 'PJ'])
test = np.loadtxt('../gina/personal_data_test.tsv', dtype=str, delimiter='\t')
#test = np.concatenate((test, np.array([cls[0] for cls in test[:,0]])[:,None], np.array([cls[1] for cls in test[:,0]])[:,None], np.array([cls[2] for cls in test[:,0]])[:,None], np.array([cls[3] for cls in test[:,0]])[:,None]), axis=1)
test = pd.DataFrame(test, columns=['type', 'post']) #, 'EI', 'NS', 'FT', 'PJ'])
"""

### Person-augmented data
"""
train = np.loadtxt('../gina/personal_data_aug_train.tsv', dtype=str, delimiter='\t')
train = pd.DataFrame(train, columns=['type', 'post']) #, 'EI', 'NS', 'FT', 'PJ'])
val = np.loadtxt('../gina/personal_data_aug_valid.tsv', dtype=str, delimiter='\t')
val = pd.DataFrame(val, columns=['type', 'post']) #, 'EI', 'NS', 'FT', 'PJ'])
test = np.loadtxt('../gina/personal_data_aug_test.tsv', dtype=str, delimiter='\t')
test = pd.DataFrame(test, columns=['type', 'post']) #, 'EI', 'NS', 'FT', 'PJ'])
"""

### Post data
train = np.loadtxt('../preprocess_new/split_train/mbti.tsv', dtype=str, delimiter='\t')
train = pd.DataFrame(train, columns=['type', 'post']) #, 'EI', 'NS', 'FT', 'PJ'])
val = np.loadtxt('../preprocess_new/split_val/mbti.tsv', dtype=str, delimiter='\t')
val = pd.DataFrame(val, columns=['type', 'post']) #, 'EI', 'NS', 'FT', 'PJ'])
test = np.loadtxt('../preprocess_new/split_test/mbti.tsv', dtype=str, delimiter='\t')
test = pd.DataFrame(test, columns=['type', 'post']) #, 'EI', 'NS', 'FT', 'PJ'])
"""
train = pd.concat([pd.read_csv('lstm.data/post_train.label', names=['type']), 
                   pd.read_csv('lstm.data/post_train.input', names=['post']), axis=1)
val = pd.concat([pd.read_csv('lstm.data/post_val.label', names=['type']), 
                 pd.read_csv('lstm.data/post_val.input', names=['post']), axis=1)
test = pd.concat([pd.read_csv('lstm.data/post_test.label', names=['type']), 
                  pd.read_csv('lstm.data/post_test.input', names=['post']), axis=1)
                              
train.type, train.post = train.type.astype(str), train.post.astype(str)
val.type, val.post = val.type.astype(str), val.post.astype(str)
test.type, test.post = test.type.astype(str), test.post.astype(str)
"""

mbti_classes = ['type'] #, 'EI', 'NS', 'FT', 'PJ']

print(f"Data loading finished: {time()-start_load}sec")
"""
kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
scoring = {'acc': 'accuracy',
               'neg_log_loss': 'neg_log_loss',
               'f1_micro': 'f1_micro',
               'f1_macro': 'f1_macro',
               'f1_weighted': 'f1_weighted',
               'bal_acc': 'balanced_accuracy'}
"""

def plot_stat():
    cnt_srs = train['type'].value_counts()
    
    plt.figure(figsize=(12,4))
    sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Types', fontsize=12)
    plt.draw()
    #plt.show()
    
def scoring(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    return acc, bal_acc, macro_f1

"""### ExtraTreesClassifier with SVD """
def etc(train, test):
    print("-------ExtraTreesClassifier with SVD-------")
    start = time()

    etc = ExtraTreesClassifier(n_estimators = 20, max_depth=4, n_jobs = -1)
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
    tsvd = TruncatedSVD(n_components=10)
    
    #np.random.seed(1)
    #results = cross_validate(model, train['post'], train['type'], cv=cv, scoring=scoring, n_jobs=-1)
    for mbti_class in mbti_classes:
        model = Pipeline([('tfidf1', tfidf), ('tsvd1', tsvd), ('etc', etc)]).fit(train['post'], train[mbti_class])
        pred = model.predict(test['post'])
        acc, bal_acc, macro_f1 = scoring(test[mbti_class], pred)
        print("{} - Acc: {:0.6f}, Bal Acc: {:0.6f}, Macro F1: {:0.6f}".format(mbti_class, acc, bal_acc, macro_f1))
    print(f"Time: {time() - start}sec")


"""### Naive Bayes"""
def nb(train, test):
    print("-------Naive Bayes-------")
    start = time()
    #np.random.seed(1)
    tfidf2 = CountVectorizer(ngram_range=(1, 1), 
                             stop_words='english',
                             lowercase = True, 
                             max_features = 5000)
    
    #results_nb = cross_validate(model_nb, train['post'], train['type'], cv=cv, scoring=scoring, n_jobs=-1)
    for mbti_class in mbti_classes:
        model = Pipeline([('tfidf1', tfidf2), ('nb', MultinomialNB())]).fit(train['post'], train[mbti_class])
        pred = model.predict(test['post'])
        acc, bal_acc, macro_f1 = scoring(test[mbti_class], pred)
        print("{} - Acc: {:0.6f}, Bal Acc: {:0.6f}, Macro F1: {:0.6f}".format(mbti_class, acc, bal_acc, macro_f1))

    print(f"Time: {time() - start}sec")

"""### Logistic Regression"""
def log_reg(train, test):
    print("-------Logistic Regression-------")
    start = time()
    #np.random.seed(1)
    tfidf2 = CountVectorizer(ngram_range=(1, 1), stop_words='english',
                                                     lowercase = True, max_features = 5000)
    #results_lr = cross_validate(model_lr, train['post'], train['type'], cv=cv, 
    #                          scoring=scoring, n_jobs=-1)
    for mbti_class in mbti_classes:
        model = Pipeline([('tfidf1', tfidf2), ('lr', LogisticRegression(class_weight="balanced", C=0.005, max_iter=500))]).fit(train['post'], train[mbti_class])
        pred = model.predict(test['post'])
        acc, bal_acc, macro_f1 = scoring(test[mbti_class], pred)
        print("{} - Acc: {:0.6f}, Bal Acc: {:0.6f}, Macro F1: {:0.6f}".format(mbti_class, acc, bal_acc, macro_f1))
        
    print(f"Time: {time() - start}sec")

"""### KNN """
def knn(train, test):
    print("-------K-Nearest Neighbors-------")
    start = time()
    #np.random.seed(1)
    tfidf2 = CountVectorizer(ngram_range=(1, 1), stop_words='english',
                                                     lowercase = True, max_features = 5000)
    #results_lr = cross_validate(model_lr, train['post'], train['type'], cv=cv, 
    #                          scoring=scoring, n_jobs=-1)
    for mbti_class in mbti_classes:
        model = Pipeline([('tfidf1', tfidf2), ('lr', KNeighborsClassifier())]).fit(train['post'], train[mbti_class])
        pred = model.predict(test['post'])
        acc, bal_acc, macro_f1 = scoring(test[mbti_class], pred)
        print("{} - Acc: {:0.6f}, Bal Acc: {:0.6f}, Macro F1: {:0.6f}".format(mbti_class, acc, bal_acc, macro_f1))
        
    print(f"Time: {time() - start}sec")

"""### XGBoost"""
def xgb(train, test):
    print("-------XGBoost-------")
    start = time()
    #np.random.seed(1)
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english').fit(train['post'])
    tsvd = TruncatedSVD(n_components=10)
    tfidf2 = CountVectorizer(ngram_range=(1, 1), stop_words='english', lowercase = True, max_features = 5000)
    #results_lr = cross_validate(model_lr, train['post'], train['type'], cv=cv, 
    #                          scoring=scoring, n_jobs=-1)
    model = XGBClassifier(booster='gbtree',
                        colsample_bylevel=0.9,
                        colsample_bytree=0.8,
                        gamma=0,
                        max_depth=8,
                        min_child_weight=3,
                        n_estimators=300,
                        nthread=4,
                        objective='multi:softmax',
                        random_state=2)
    model_binary = XGBClassifier(booster='gbtree',
                        colsample_bylevel=0.9,
                        colsample_bytree=0.8,
                        gamma=0,
                        max_depth=8,
                        min_child_weight=3,
                        n_estimators=300,
                        nthread=4,
                        objective='binary:logistic',
                        random_state=2)
                        
    for mbti_class in mbti_classes:
        if mbti_class == 'type':
            model.fit(tfidf.transform(train['post']), train[mbti_class], eval_set=[(tfidf.transform(val['post']), val[mbti_class])], early_stopping_rounds=25, verbose=100)
        else:
            model_binary.fit(tfidf.transform(train['post']), train[mbti_class], eval_set=[(tfidf.transform(val['post']), val[mbti_class])], early_stopping_rounds=25, verbose=100)
        pred = model.predict(tfidf.transform(test['post'])) if mbti_class == 'type' else model_binary.predict(tfidf.transform(test['post']))
        acc, bal_acc, macro_f1 = scoring(test[mbti_class], pred)
        print("{} - Acc: {:0.6f}, Bal Acc: {:0.6f}, Macro F1: {:0.6f}".format(mbti_class, acc, bal_acc, macro_f1))
        
    print(f"Time: {time() - start}sec")

"""### Support Vector Machine"""
def svc(train, test):
    print("-------Support Vector Machine-------")
    start = time()
    #np.random.seed(1)
    tfidf2 = CountVectorizer(ngram_range=(1, 1), stop_words='english',
                                                     lowercase = True, max_features = 5000)
    #results_lr = cross_validate(model_lr, train['post'], train['type'], cv=cv, 
    #                          scoring=scoring, n_jobs=-1)
    for mbti_class in mbti_classes:
        model = Pipeline([('tfidf1', tfidf2), ('lr', OneVsRestClassifier(SVC(kernel='linear')))]).fit(train['post'], train[mbti_class])
        pred = model.predict(test['post'])
        acc, bal_acc, macro_f1 = scoring(test[mbti_class], pred)
        print("{} - Acc: {:0.4f}, Bal Acc: {:0.4f}, Macro F1: {:0.4f}".format(mbti_class, acc, bal_acc, macro_f1))
        
    print(f"Time: {time() - start}sec")


if __name__ == "__main__":
    np.random.seed(1)
    plot_stat()
    
    etc(train, test)
    nb(train, test)
    log_reg(train, test)
    knn(train, test)
    xgb(train, test)
    #svc(train, test)