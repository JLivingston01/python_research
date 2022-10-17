# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 23:01:49 2019

@author: jliv
"""

import oauth2

import datetime as dt
import re
import json
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.stem.porter import *
import nltk
import nltk.sentiment
import numpy as np
np.random.seed(2018)


import pandas as pd

corpuslist = listdir("C://Users/jliv/Downloads/tweets/Musicians/")
corpusdf = pd.read_csv("C://Users/jliv/Downloads/tweets/Musicians/"+corpuslist[0])
for i in corpuslist[1:]:
    corpusdf = corpusdf.append(pd.read_csv("C://Users/jliv/Downloads/tweets/Musicians/"+i))
corpusdf.columns.values

corpusdf = corpusdf[corpusdf['label'] != 'Jason Livingston']
corpusdf.fillna("",inplace = True)
text = list(corpusdf['text'])
final_tweets = list(corpusdf['final_tweets'])
labels = list(corpusdf['label'])

sid = nltk.sentiment.vader.SentimentIntensityAnalyzer()
compound = []
neutral = []
negative = []
positive = []

for i,j in zip(final_tweets,text):
    ss = sid.polarity_scores(i)
    comp = ss['compound']
    neg = ss['neg']
    neu = ss['neu']
    pos = ss['pos']
    compound.append(comp)
    neutral.append(neu)
    negative.append(neg)
    positive.append(pos)



tweet_df = pd.DataFrame()
tweet_df['text'] = text
tweet_df['final_tweets'] = final_tweets
tweet_df['label'] = labels
tweet_df['compound'] = compound
tweet_df['positive'] = positive
tweet_df['neutral'] = neutral
tweet_df['negative'] = negative

scores = pd.pivot_table(data = tweet_df, index = ['label'], values = ['compound','positive','neutral','negative'], aggfunc = 'mean')
cnt = pd.pivot_table(data = tweet_df, index = ['label'], values = ['compound'], aggfunc = 'count').rename(mapper={'compound':'count'})
scores['count'] = cnt['compound']



tweet_df[tweet_df['label']=='JK'].sample(15)['final_tweets']

np.mean(tweet_df['compound'])

import matplotlib.pyplot as plt

plt.scatter(scores['count'],scores['compound'])
for i in list(scores.index):
    plt.annotate(s = i, xy = (scores['count'][i]+20,scores['compound'][i]),size = 14)
plt.show()


def normalize(x):
    return (x-min(x))/(max(x)-min(x))

plt.figure(figsize = (6,6))
plt.scatter(normalize(scores['count']),normalize(scores['compound']))
for i in list(scores.index):
    plt.annotate(s = i, xy = (normalize(scores['count'])[i],normalize(scores['compound'])[i]),size = 14)
plt.show()

scores['compound norm'] = normalize(scores['compound'])
scores['count norm'] = normalize(scores['count'])

scores['dist'] = np.sqrt(scores['compound norm']**2+scores['count norm']**2)
scores['prod'] = scores['compound norm']*scores['count norm']


plt.scatter(scores['dist'],scores['prod'])
for i in list(scores.index):
    plt.annotate(s = i, xy = (scores['dist'][i],scores['prod'][i]),size = 14)
plt.show()
