# <p align='center'> IMDB Sentiment Analysis (NLP)  </p>
<p align='center'>
  <img src="https://github.com/giova22i/Data-Science-Portfolio/blob/main/images/moviecover.jpg" />
    </p>

## Objective

The goal is to train a model to automatically detect negative reviews. Use the dataset of IMDB movie reviews with polarity labelling to build a model for classifying positive and negative reviews. Try to attain an F1 score of at least 0.85.

## Data Description
The data was provided by Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

Here's the description of the selected fields:

 * review: the review text
 * pos: the target, '0' for negative and '1' for positive
 * ds_part: 'train'/'test' for the train/test part of dataset, correspondingly


##  Models Evaluated
      LogisticRegression
      NLTK, TF-IDF, LR 
      spaCy, TF-IDF, LR
      spaCy, TF-IDF, XGB
      spaCy, TF-IDF, LGBM
      BERT, LR

## Insights
* Showcased the model's ability to accurately identify negative reviews, highlighting its potential for improving review analysis and decision-making processes.
  
##  Libraries Used
      Pandas
      Matplotlib.pyplot, matplotlib.dates
      scipy.stats
      numpy
      math
      re
      sklearn
      nltk
      spacy
      lightgbm
      xgboost
      torch
      transformers
      tqdm


## Background

This project is part of the Data Scientist training program from Practicum by Yandex. More info in link below:

https://practicum.yandex.com/data-scientist
