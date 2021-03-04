# Your Personality Speaks: An Exploration to Explainability of Personality Classification

This is the **PyTorch implementation** for the *Deep Learning for Natural Language Process* class project. Detailed information of our work is described in the [report pdf file](https://github.com/l1905kw/nlp-class-project/blob/master/report.pdf).

## Introduction
We explore various neural network models to predict people’s MBTI types with their short written text. We formulated the task as a text classification problem having 16 MBTI types as target classes. We use several machine learning algorithms for the baseline of the work. Then, we implement CNN and RNN-based methods to compare the performance. We also experiment on a self-attention-based encoder – namely, BERT  – for representing the input text.
We also investigate various ways to visualize how the model predicts people’s MBTI types on the model’s inference. Attention mechanism usually allows the model to focus on the relevant expressions of the input text as needed. We adopt the attention mechanism to show which part of the text plays an important role in predicting the type.




## Datasets
We employ the MBTI dataset available from [Kaggle](https://www.kaggle.com/datasnaek/mbti-type). The dataset is comprised of over 8,600 rows of data in which each row represents each person’s four-letter MBTI code, as well as the last 50 things that they have posted on the forum.

 - **Per post** - To increase the training samples of the dataset, we split each row of the data by the number of posts it contains
 - **Per person** - Original dataset
 - **Per person augmented** - 

You can get preprocessed post data from ``preprocess_new/split_[train/test/val]`` and personal data from ``gina``.

## Results
Accuracies, balanced accuracies, and F1-scores of each model for each task:
<img align="middle" width="700" src="https://github.com/l1905kw/tree/master/imgs/result.PNG">

Top 20 important words for classifying MBTI types:
<img align="middle" width="700" src="https://github.com/l1905kw/tree/master/imgs/important_words.PNG">
