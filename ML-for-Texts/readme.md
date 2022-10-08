# Machine Learning for Texts Project
The Film Junky Union, a new edgy community for classic movie enthusiasts, is developing a system for filtering and categorizing movie reviews. 
## Objective

The goal is to train a model to automatically detect negative reviews. Use the dataset of IMDB movie reviews with polarity labelling to build a model for classifying positive and negative reviews. Try to attain an F1 score of at least 0.85.

## Data Description
 * The data was provided by Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

### Description of the selected fields
 * review: the review text
 * pos: the target, '0' for negative and '1' for positive
 * ds_part: 'train'/'test' for the train/test part of dataset, correspondingly

##  Libraries Used
 * Pandas
 * Matplotlib.pyplot, matplotlib.dates
 * scipy.stats
 * numpy
 * math
 * re
 * sklearn
 * nltk
 * spacy
 * lightgbm
 * xgboost
 * torch
 * transformers
 * tqdm

##  Models Evaluated
 * LogisticRegression
 * NLTK, TF-IDF, LR 
 * spaCy, TF-IDF, LR
 * spaCy, TF-IDF, XGB
 * spaCy, TF-IDF, LGBM
 * BERT, LR

## Background

This project is part of the Data Scientist training program from Practicum by Yandex. More info in link below:

https://practicum.yandex.com/data-scientist
