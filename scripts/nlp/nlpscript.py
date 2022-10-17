# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 16:04:14 2019

@author: jliv
"""
#PACKAGES

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



from wordcloud import get_single_color_func

from wordcloud import WordCloud

import tensorflow

from tensorflow import keras
from tensorflow import losses

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder


import pickle
#49-545, api pull, clean for 1 day of data


stopwords = nltk.corpus.stopwords  
stop_words = set(stopwords.words("english"))
word_tokenize = nltk.tokenize.word_tokenize

query = 'https://api.twitter.com/1.1/search/tweets.json?l=en&q="Fox%20News"%20-Congrats%20-Stand%20-Laura%20-Why%20since%3A2019-01-29%20until%3A2019-01-30&result_type=recent&count=1000&tweet_mode=extended'

def req(query):
    consumer = oauth2.Consumer(key='kq5bb4YfBfoLUXd90vCwq4RWX'.encode('utf-8'), secret='JzIDVyTToHGRpoSX61zQHr1QyXNVyOM7DHDLFrfIoK4q3XlMcA'.encode('utf-8'))
    token = oauth2.Token(key='1047557115156602880-Lg0ZzFAXdRBE3MWvIjDoosgbrmqbFd', secret='XtYHGlsPUBxb2cc4O48NqXmrtVzEiplawEO3illlAHmKz')
    client = oauth2.Client(consumer, token)
    resp, content = client.request( query, method="GET", body=bytes("", "utf-8"), headers=None )
    return content

#home_timeline = req(query)
    
#consumer = oauth2.Consumer(key='kq5bb4YfBfoLUXd90vCwq4RWX'.encode('utf-8'), secret='JzIDVyTToHGRpoSX61zQHr1QyXNVyOM7DHDLFrfIoK4q3XlMcA'.encode('utf-8'))
#token = oauth2.Token(key='1047557115156602880-Lg0ZzFAXdRBE3MWvIjDoosgbrmqbFd', secret='XtYHGlsPUBxb2cc4O48NqXmrtVzEiplawEO3illlAHmKz')
#client = oauth2.Client(consumer, token)
#resp, content = client.request( query, method="GET", body=bytes("", "utf-8"), headers=None )

def oauth_req(url, token, secret, http_method="GET", post_body="", http_headers=None):
    consumer = oauth2.Consumer(key='kq5bb4YfBfoLUXd90vCwq4RWX '.encode('utf-8'), secret='JzIDVyTToHGRpoSX61zQHr1QyXNVyOM7DHDLFrfIoK4q3XlMcA '.encode('utf-8'))
    token = oauth2.Token(key=token, secret=secret)
    client = oauth2.Client(consumer, token)
    resp, content = client.request( url, method=http_method, body=bytes(post_body, "utf-8"), headers=http_headers )
    return content

searchterm = '"Lyft"'
terma = searchterm.replace('"',"")
terma = terma.replace(" ","")
language = 'en'

startdate = dt.datetime.now()
to_date = startdate + dt.timedelta(1)
startdate = dt.datetime.strftime(startdate,"%Y-%m-%d")
to_date = dt.datetime.strftime(to_date,"%Y-%m-%d")

#startdate = "2019-02-05"
#to_date = "2019-02-06"
max_tweets = 1000
appendix = "v1"
#exclude = ['-Congrats','-Stand','-Laura','-Why']
exclude = ['']
#How = mixed, recent or popular
how = 'mixed'
searchterm = searchterm.split()
searchterm = "%20".join(searchterm)

enddate = dt.datetime.strftime(dt.datetime.strptime(startdate,"%Y-%m-%d") +dt.timedelta(1),"%Y-%m-%d")
days = dt.datetime.strptime(to_date,"%Y-%m-%d") - dt.datetime.strptime(startdate,"%Y-%m-%d")
days = days.days
exclude = "%20".join(exclude)



parameters = (language,searchterm,startdate,enddate)
#raw_query="l={}&q={}%20{}%20since%3A{}%20until%3A{}&result_type=mixed&count=1000".format(language,searchterm,exclude,startdate,enddate)

   
times = []
date = []
text = []
retweet_cnt = []
fvrt_cnt = []
user = []
user_flwrs=[]
user_statuses = []
timezone = []

'''len(text)
lengths = []
for i in text:
    lengths.append(len(i))'''
#raw_query="l={}&q={}%20{}%20since%3A{}%20until%3A{}&result_type=mixed&tweet_mode=extended&count=1000".format(language,searchterm,exclude,startdate,enddate)
#query = 'https://api.twitter.com/1.1/search/tweets.json?'+raw_query
#home_timeline = oauth_req(query, '986743245127503872-ePHRirA1hxJsMVPjogWbFSeZFmo4V5Q'.encode('utf-8'), 'N4PqSMhHGqjlZ2yqmLnPB8cFJgPXfMsj7PbzSrk55ageO'.encode('utf-8') )
raw_query="lang={}&q={}%20{}%20since%3A{}%20until%3A{}&result_type={}&count=1000&tweet_mode=extended".format(language,searchterm,exclude,startdate,enddate,how)
query = 'https://api.twitter.com/1.1/search/tweets.json?'+raw_query
home_timeline = req(query)
home_timeline = home_timeline.decode("utf-8") 
home_timeline = json.loads(home_timeline)
statuses = home_timeline['statuses']
print(len(statuses))

for i in range(len(statuses)):
    
    times.append(statuses[i]['created_at'])
    try:
        text.append(statuses[i]['retweeted_status']['full_text'])
    except:
        text.append(statuses[i]['full_text'])
    fvrt_cnt.append(statuses[i]['favorite_count'])
    retweet_cnt.append(statuses[i]['retweet_count'])
    user.append(statuses[i]['user']['name'])
    user_flwrs.append(statuses[i]['user']['followers_count'])
    user_statuses.append(statuses[i]['user']['statuses_count'])
    timezone.append(statuses[i]['user']['time_zone'])


    
    

emojis = pd.read_csv('C://Users/jliv/Downloads/emojis.txt',sep = '\t', encoding = 'utf-8')
#Map of Unicode and Names
emoji_map = pd.DataFrame()
emoji_map['name'] = emojis['Name(s)']
emoji_map['code'] = emojis['Escaped Unicode']
#Map of Emojis and names
emoji_map1 = pd.DataFrame()
emoji_map1['name'] = emojis['Name(s)']
emoji_map1['Emoji'] = emojis['Emoji']
#Handle escape characters in unicode
codes = [] 
for i in list(emojis['Escaped Unicode']):
    x = i.replace("\\","\\")
    codes.append(x)
    

emojislist = emoji_map1['Emoji']
#Convert CSVs of mappings to dict mappings
emoji_map.index = codes
emoji_dict = emoji_map.to_dict()
emoji_dict = emoji_dict['name']

emoji_map1.index = emojislist
emoji_dict1 = emoji_map1.to_dict()
emoji_dict1 = emoji_dict1['name']

#Replace tweet emojis and unicode with descriptions of characters
emoji_clean = []
for i in text:
    x = i
    for k,v in emoji_dict1.items():
        x = x.replace(k, v)
    for k,v in emoji_dict.items():
        x = x.replace(k, v)
    emoji_clean.append(x)


tweetvector_clean = []
for i in emoji_clean:
    x = re.sub(r"^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$"," ", i)
    x = re.sub(r"htt\S+"," ", x) #x = x.decode('utf-8')
    x = re.sub(r"pic.twit\S+"," ", x)
    x = re.sub(r"www.\S+"," ", x)
    x = re.sub(r"www.\S+"," ", x)
    x = re.sub(r"@\S+"," ", x)
    x = re.sub(r"\xa0"," ", x)
    x = re.sub(r"\\u\S+"," ", x)
    x = x.replace('#',' ')
    x = x.replace('amp;','&')
    x = x.replace('gt;',' ')
    x = x.replace('\\n',' ')
    y = x.replace('$',' ')
    y = y.replace('(',' ')
    y = y.replace('–',' ')
    y = y.replace('‘',' ')
    y = y.replace('“',' ')
    y = y.replace('”',' ')
    y = y.replace('`',' ')
    y = y.replace(']',' ')
    y = y.replace('[',' ')
    y = y.replace(';',' ')
    y = y.replace(')',' ')
    y = y.replace('/',' ')
    y = y.replace('*',' ')
    y = y.replace(',',' ')
    y = y.replace('’','')
    y = y.replace('.','')
    y = y.replace('-',' ')
    y = y.replace("'",'')
    y = y.replace(':',' ')
    y = y.replace('@',' ')
    y = y.replace('!',' ')
    y = y.replace('…',' ')
    y = y.replace('?',' ')
    y = y.replace('>',' ')
    y = y.replace('&',' ')
    y = y.replace("\\","")
    y = y.replace("\\u2066","")
    tweetvector_clean.append(y)


tweetvector_tokenized = []
for i in tweetvector_clean:
    x = word_tokenize(i)
    tweetvector_tokenized.append(x)
tweetvector_stopped = []
for i in tweetvector_tokenized:
    newstatement = [j for j in i if j not in stop_words]
    tweetvector_stopped.append(newstatement)
####
    
final_tweets = []
for i in tweetvector_stopped:
    x = " ".join(i)
    final_tweets.append(x)



sid = nltk.sentiment.vader.SentimentIntensityAnalyzer()
compound = []
neutral = []
negative = []
positive = []
for i in final_tweets:
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
tweet_df['datetime'] = times
tweet_df['text'] = text
tweet_df['retweet_cnt'] = retweet_cnt
tweet_df['fvrt_cnt'] = fvrt_cnt
tweet_df['final_tweets'] = final_tweets
tweet_df['compound'] = compound
tweet_df['positive'] = positive
tweet_df['neutral'] = neutral
tweet_df['negative'] = negative


date = []
time = []
for i in list(times):
    x = dt.datetime.strptime(i,"%a %b %d %H:%M:%S %z %Y")
    d = dt.datetime.strftime(x,"%Y-%m-%d")
    t = dt.datetime.strftime(x,"%H:%M:%S")
    date.append(d)
    time.append(t)
    
tweet_df['date'] = date  
tweet_df['time'] = time   

tweet_non_neutral = tweet_df[tweet_df['compound'] != 0]
tweet_neutral = tweet_df[tweet_df['compound'] == 0 ]


tweet_summary = pd.pivot_table(tweet_non_neutral, values = ['compound'],index = ['date'], aggfunc = 'mean')
tweet_count = pd.pivot_table(tweet_non_neutral, values = ['compound'],index = ['date'], aggfunc = 'count')
tweet_all_count = pd.pivot_table(tweet_df, values = ['compound'],index = ['date'], aggfunc = 'count')
tweet_summary.reset_index(inplace = True, drop = False)
tweet_count.reset_index(inplace = True, drop = False)
tweet_all_count.reset_index(inplace = True, drop = False)

tweet_summary = pd.merge(tweet_summary,tweet_count, on = 'date', how = 'left')
tweet_summary = pd.merge(tweet_summary,tweet_all_count, on = 'date', how = 'left')
tweet_summary = tweet_summary.rename(index = str, columns = {'compound_x':'mean_compound','compound_y':'non_neutral_tweets','compound':'total_tweets'})
now = dt.datetime.strftime(dt.datetime.now(),"%Y-%m-%d")

tweet_df.to_csv("C://Users/jliv/Downloads/tweets/tweettest/"+terma+"tweets"+startdate+".csv")

tweet_summary.to_csv("C://Users/jliv/Downloads/tweets/"+terma+"scores"+startdate+".csv")


searchterm = '"Uber"'
terma = searchterm.replace('"',"")
terma = terma.replace(" ","")
language = 'en'

startdate = dt.datetime.now()
to_date = startdate + dt.timedelta(1)
startdate = dt.datetime.strftime(startdate,"%Y-%m-%d")
to_date = dt.datetime.strftime(to_date,"%Y-%m-%d")

#startdate = "2019-02-05"
#to_date = "2019-02-06"
max_tweets = 1000
appendix = "v1"
#exclude = ['-Congrats','-Stand','-Laura','-Why']
exclude = ['']
#How = mixed, recent or popular
how = 'mixed'
searchterm = searchterm.split()
searchterm = "%20".join(searchterm)

enddate = dt.datetime.strftime(dt.datetime.strptime(startdate,"%Y-%m-%d") +dt.timedelta(1),"%Y-%m-%d")
days = dt.datetime.strptime(to_date,"%Y-%m-%d") - dt.datetime.strptime(startdate,"%Y-%m-%d")
days = days.days
exclude = "%20".join(exclude)



parameters = (language,searchterm,startdate,enddate)
#raw_query="l={}&q={}%20{}%20since%3A{}%20until%3A{}&result_type=mixed&count=1000".format(language,searchterm,exclude,startdate,enddate)

   
times = []
date = []
text = []
retweet_cnt = []
fvrt_cnt = []
user = []
user_flwrs=[]
user_statuses = []
timezone = []

'''len(text)
lengths = []
for i in text:
    lengths.append(len(i))'''
#raw_query="l={}&q={}%20{}%20since%3A{}%20until%3A{}&result_type=mixed&tweet_mode=extended&count=1000".format(language,searchterm,exclude,startdate,enddate)
#query = 'https://api.twitter.com/1.1/search/tweets.json?'+raw_query
#home_timeline = oauth_req(query, '986743245127503872-ePHRirA1hxJsMVPjogWbFSeZFmo4V5Q'.encode('utf-8'), 'N4PqSMhHGqjlZ2yqmLnPB8cFJgPXfMsj7PbzSrk55ageO'.encode('utf-8') )
raw_query="lang={}&q={}%20{}%20since%3A{}%20until%3A{}&result_type={}&count=1000&tweet_mode=extended".format(language,searchterm,exclude,startdate,enddate,how)
query = 'https://api.twitter.com/1.1/search/tweets.json?'+raw_query
home_timeline = req(query)
home_timeline = home_timeline.decode("utf-8") 
home_timeline = json.loads(home_timeline)
statuses = home_timeline['statuses']
print(len(statuses))

for i in range(len(statuses)):
    
    times.append(statuses[i]['created_at'])
    try:
        text.append(statuses[i]['retweeted_status']['full_text'])
    except:
        text.append(statuses[i]['full_text'])
    fvrt_cnt.append(statuses[i]['favorite_count'])
    retweet_cnt.append(statuses[i]['retweet_count'])
    user.append(statuses[i]['user']['name'])
    user_flwrs.append(statuses[i]['user']['followers_count'])
    user_statuses.append(statuses[i]['user']['statuses_count'])
    timezone.append(statuses[i]['user']['time_zone'])


    
    

emojis = pd.read_csv('C://Users/jliv/Downloads/emojis.txt',sep = '\t', encoding = 'utf-8')
#Map of Unicode and Names
emoji_map = pd.DataFrame()
emoji_map['name'] = emojis['Name(s)']
emoji_map['code'] = emojis['Escaped Unicode']
#Map of Emojis and names
emoji_map1 = pd.DataFrame()
emoji_map1['name'] = emojis['Name(s)']
emoji_map1['Emoji'] = emojis['Emoji']
#Handle escape characters in unicode
codes = [] 
for i in list(emojis['Escaped Unicode']):
    x = i.replace("\\","\\")
    codes.append(x)
    

emojislist = emoji_map1['Emoji']
#Convert CSVs of mappings to dict mappings
emoji_map.index = codes
emoji_dict = emoji_map.to_dict()
emoji_dict = emoji_dict['name']

emoji_map1.index = emojislist
emoji_dict1 = emoji_map1.to_dict()
emoji_dict1 = emoji_dict1['name']

#Replace tweet emojis and unicode with descriptions of characters
emoji_clean = []
for i in text:
    x = i
    for k,v in emoji_dict1.items():
        x = x.replace(k, v)
    for k,v in emoji_dict.items():
        x = x.replace(k, v)
    emoji_clean.append(x)


tweetvector_clean = []
for i in emoji_clean:
    x = re.sub(r"^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$"," ", i)
    x = re.sub(r"htt\S+"," ", x) #x = x.decode('utf-8')
    x = re.sub(r"pic.twit\S+"," ", x)
    x = re.sub(r"www.\S+"," ", x)
    x = re.sub(r"www.\S+"," ", x)
    x = re.sub(r"@\S+"," ", x)
    x = re.sub(r"\xa0"," ", x)
    x = re.sub(r"\\u\S+"," ", x)
    x = x.replace('#',' ')
    x = x.replace('amp;','&')
    x = x.replace('gt;',' ')
    x = x.replace('\\n',' ')
    y = x.replace('$',' ')
    y = y.replace('(',' ')
    y = y.replace('–',' ')
    y = y.replace('‘',' ')
    y = y.replace('“',' ')
    y = y.replace('”',' ')
    y = y.replace('`',' ')
    y = y.replace(']',' ')
    y = y.replace('[',' ')
    y = y.replace(';',' ')
    y = y.replace(')',' ')
    y = y.replace('/',' ')
    y = y.replace('*',' ')
    y = y.replace(',',' ')
    y = y.replace('’','')
    y = y.replace('.','')
    y = y.replace('-',' ')
    y = y.replace("'",'')
    y = y.replace(':',' ')
    y = y.replace('@',' ')
    y = y.replace('!',' ')
    y = y.replace('…',' ')
    y = y.replace('?',' ')
    y = y.replace('>',' ')
    y = y.replace('&',' ')
    y = y.replace("\\","")
    y = y.replace("\\u2066","")
    tweetvector_clean.append(y)


tweetvector_tokenized = []
for i in tweetvector_clean:
    x = word_tokenize(i)
    tweetvector_tokenized.append(x)
tweetvector_stopped = []
for i in tweetvector_tokenized:
    newstatement = [j for j in i if j not in stop_words]
    tweetvector_stopped.append(newstatement)
####
    
final_tweets = []
for i in tweetvector_stopped:
    x = " ".join(i)
    final_tweets.append(x)



sid = nltk.sentiment.vader.SentimentIntensityAnalyzer()
compound = []
neutral = []
negative = []
positive = []
for i in final_tweets:
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
tweet_df['datetime'] = times
tweet_df['text'] = text
tweet_df['retweet_cnt'] = retweet_cnt
tweet_df['fvrt_cnt'] = fvrt_cnt
tweet_df['final_tweets'] = final_tweets
tweet_df['compound'] = compound
tweet_df['positive'] = positive
tweet_df['neutral'] = neutral
tweet_df['negative'] = negative


date = []
time = []
for i in list(times):
    x = dt.datetime.strptime(i,"%a %b %d %H:%M:%S %z %Y")
    d = dt.datetime.strftime(x,"%Y-%m-%d")
    t = dt.datetime.strftime(x,"%H:%M:%S")
    date.append(d)
    time.append(t)
    
tweet_df['date'] = date  
tweet_df['time'] = time   

tweet_non_neutral = tweet_df[tweet_df['compound'] != 0]
tweet_neutral = tweet_df[tweet_df['compound'] == 0 ]


tweet_summary = pd.pivot_table(tweet_non_neutral, values = ['compound'],index = ['date'], aggfunc = 'mean')
tweet_count = pd.pivot_table(tweet_non_neutral, values = ['compound'],index = ['date'], aggfunc = 'count')
tweet_all_count = pd.pivot_table(tweet_df, values = ['compound'],index = ['date'], aggfunc = 'count')
tweet_summary.reset_index(inplace = True, drop = False)
tweet_count.reset_index(inplace = True, drop = False)
tweet_all_count.reset_index(inplace = True, drop = False)

tweet_summary = pd.merge(tweet_summary,tweet_count, on = 'date', how = 'left')
tweet_summary = pd.merge(tweet_summary,tweet_all_count, on = 'date', how = 'left')
tweet_summary = tweet_summary.rename(index = str, columns = {'compound_x':'mean_compound','compound_y':'non_neutral_tweets','compound':'total_tweets'})
now = dt.datetime.strftime(dt.datetime.now(),"%Y-%m-%d")

tweet_df.to_csv("C://Users/jliv/Downloads/tweets/tweettest/"+terma+"tweets"+startdate+".csv")

tweet_summary.to_csv("C://Users/jliv/Downloads/tweets/"+terma+"scores"+startdate+".csv")







#Assemble all collected data for each brand
#560 - 631





imlist = listdir("C://Users/jliv/Downloads/tweets/tweettest/")
len(imlist)

lyftlist = [x for x in imlist if 'Lyfttweets' in x]

lyftdf = pd.read_csv('C://Users/jliv/Downloads/tweets/tweettest/'+lyftlist[0])

for i in lyftlist[1:]:
    temp = pd.read_csv('C://Users/jliv/Downloads/tweets/tweettest/'+i)
    lyftdf = lyftdf.append(temp)

lyftdf.reset_index(inplace= True, drop = True)


imlist = listdir("C://Users/jliv/Downloads/tweets/tweettest/")
len(imlist)

uberlist = [x for x in imlist if 'Ubertweets' in x]
uberdf = pd.read_csv('C://Users/jliv/Downloads/tweets/tweettest/'+uberlist[0])

for i in uberlist[1:]:
    temp = pd.read_csv('C://Users/jliv/Downloads/tweets/tweettest/'+i)
    uberdf = uberdf.append(temp)

uberdf.reset_index(inplace= True, drop = True)



lyftdf['comp'] = 'lyft'
uberdf['comp'] = 'uber'

nltk.download('wordnet')
stemmer = SnowballStemmer('english')

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result
    
    


docdf = uberdf.append(lyftdf)
docdf.reset_index(inplace = True, drop = True)

docdf['final_tweets'] = docdf['final_tweets'].fillna(" ")
docs = list(docdf['final_tweets'])
docs1 = []
for i in docs:
    x = i.lower().replace('lyft',"").replace('uber',"")
    docs1.append(x)
    
docdf['final_tweets2'] = docs1

docdf2 = pd.pivot_table(data = docdf, index = ['final_tweets2'], values=['compound'], aggfunc = 'count')
docdf2.reset_index(inplace = True, drop = False)


#processed_docs = docdf2['final_tweets2'].map(preprocess)
processed_docs = docdf['final_tweets2'].map(preprocess)
merged = []
for i in processed_docs:
    merged.append(" ".join(i))
docdf['final_tweets2_stemmed'] = merged

docdf.to_csv('C://Users/jliv/Downloads/tweets/tweets_collected.csv')



#Lyft Wordcloud
#641 - 794




tweetdf = pd.read_csv("C://Users/jliv/Downloads/tweets/tweets_collected.csv")

tweetdf = tweetdf[tweetdf['comp']=='lyft']
tweetdf.reset_index(inplace = True, drop = True)
tweetdf['final_tweets2_stemmed'] = tweetdf['final_tweets2_stemmed'].fillna(' ')
#CREATE WORDCLOUD WITH LABELED TWEETS

#Create df of each occurrence of word with scores of tweet
tweetlist = list(tweetdf['final_tweets2_stemmed'])


tokenized = []
compound = []
negative = []
neutral = []
positive = []
date = []
for i in range(len(tweetlist)):
    tokenized.append(tweetlist[i].split())
    compound.append(tweetdf['compound'][i])
    negative.append(tweetdf['negative'][i])
    neutral.append(tweetdf['neutral'][i])
    positive.append(tweetdf['positive'][i])
    date.append(tweetdf['date'][i])
    
words = []
compound2 = []
negative2 = []
neutral2 = []
positive2 = []
date2 = []
for i in range(len(tokenized)):
    for j in tokenized[i]:
        words.append(j.lower())
        compound2.append(compound[i])
        negative2.append(negative[i])
        neutral2.append(neutral[i])
        positive2.append(positive[i])
        date2.append(date[i])
        
wordsdf = pd.DataFrame()
wordsdf['date'] = date2
wordsdf['words'] = words
wordsdf['compound'] = compound2
wordsdf['negative'] = negative2
wordsdf['neutral'] = neutral2
wordsdf['positive'] = positive2


#DFs of unique words with average score when used
wordssent = pd.pivot_table(data = wordsdf, values = ['compound','negative','neutral','positive'], index = ['words'], aggfunc = 'mean')
wordscount = pd.pivot_table(data = wordsdf, values = ['compound'], index = ['words'], aggfunc = 'count')

wordssent['count'] = wordscount['compound']

#Sorted by Use
wordssent.sort_values(by = 'count', ascending = False, inplace = True)


#Sorted by Negative Sentiment
words_negative_sent = wordssent.copy()

words_negative_sent.sort_values(by = 'compound', ascending = True, inplace = True)

words_negative_sent[words_negative_sent['count'] > 10]
#Sorted by Positive Sentiment
words_positive_sent = wordssent.copy()

words_positive_sent.sort_values(by = 'compound', ascending = False, inplace = True)


time_df = pd.pivot_table(tweetdf, index = 'date',values = 'compound', aggfunc = 'mean')


wordssent.reset_index(drop = False, inplace = True)
 


#Wordcloud wordcloud



class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping
       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.
       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)



compmin = min(wordssent['compound'])
compmax = max(wordssent['compound'])


compmin = -1
compmax = 1
n = 30

wordrepeats = []
wordrepeats_sent = []
for i,j,l in zip(list(wordssent[wordssent['count']>n]['words']),list(wordssent[wordssent['count']>n]['count']),list(wordssent[wordssent['count']>n]['compound'])): 
    for k in range(j):
        wordrepeats.append(i.lower())
        wordrepeats_sent.append(l)

text = " ".join(wordrepeats)

UW = []
color = []
for i,j in zip(list(wordssent[wordssent['count']>n]['words']),list(wordssent[wordssent['count']>n]['compound'])):
    UW.append(i)
    color.append('rgb('+str(int(255*(1- (j-compmin)/(compmax-compmin))))+','+str(int(155*(j-compmin)/(compmax-compmin)))+', 0)')
 
colorset = list(set(color))
color_to_words = {}
for i in colorset:
    words_by_color = []
    for j in range(len(UW)):
        if color[j] == i:
            words_by_color.append(UW[j])
        else:
            pass
    color_to_words[i] = words_by_color
        
    
grouped_color_func = SimpleGroupedColorFunc(color_to_words, 'grey')


wordcloud = WordCloud(collocations = False,width = 800, height = 500,background_color = "black").generate(text)
wordcloud.recolor(color_func=grouped_color_func)
# Display the generated image:
# the matplotlib way:

plt.figure( figsize=(8,6) )
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Lyft Word Cloud")
plt.savefig("C://Users/jliv/Documents/GitHub/JLivingston01.github.io/images/lyft_wordcloud.png")


#Uber Wordcloud
#800 - 963


#tweetdf = pd.read_csv('C://users/jliv/downloads/tweets/lda_tweets.csv')


tweetdf = pd.read_csv("C://Users/jliv/Downloads/tweets/tweets_collected.csv")

tweetdf = tweetdf[tweetdf['comp']=='uber']
tweetdf.reset_index(inplace = True, drop = True)
tweetdf['final_tweets2_stemmed'] = tweetdf['final_tweets2_stemmed'].fillna(' ')
#CREATE WORDCLOUD WITH LABELED TWEETS

#Create df of each occurrence of word with scores of tweet
tweetlist = list(tweetdf['final_tweets2_stemmed'])


tokenized = []
compound = []
negative = []
neutral = []
positive = []
date = []
for i in range(len(tweetlist)):
    tokenized.append(tweetlist[i].split())
    compound.append(tweetdf['compound'][i])
    negative.append(tweetdf['negative'][i])
    neutral.append(tweetdf['neutral'][i])
    positive.append(tweetdf['positive'][i])
    date.append(tweetdf['date'][i])
    
words = []
compound2 = []
negative2 = []
neutral2 = []
positive2 = []
date2 = []
for i in range(len(tokenized)):
    for j in tokenized[i]:
        words.append(j.lower())
        compound2.append(compound[i])
        negative2.append(negative[i])
        neutral2.append(neutral[i])
        positive2.append(positive[i])
        date2.append(date[i])
        
wordsdf = pd.DataFrame()
wordsdf['date'] = date2
wordsdf['words'] = words
wordsdf['compound'] = compound2
wordsdf['negative'] = negative2
wordsdf['neutral'] = neutral2
wordsdf['positive'] = positive2


#DFs of unique words with average score when used
wordssent = pd.pivot_table(data = wordsdf, values = ['compound','negative','neutral','positive'], index = ['words'], aggfunc = 'mean')
wordscount = pd.pivot_table(data = wordsdf, values = ['compound'], index = ['words'], aggfunc = 'count')

wordssent['count'] = wordscount['compound']

#Sorted by Use
wordssent.sort_values(by = 'count', ascending = False, inplace = True)


#Sorted by Negative Sentiment
words_negative_sent = wordssent.copy()

words_negative_sent.sort_values(by = 'compound', ascending = True, inplace = True)

words_negative_sent[words_negative_sent['count'] > 10]
#Sorted by Positive Sentiment
words_positive_sent = wordssent.copy()

words_positive_sent.sort_values(by = 'compound', ascending = False, inplace = True)


time_df = pd.pivot_table(tweetdf, index = 'date',values = 'compound', aggfunc = 'mean')


#from PIL import Image, ImageDraw, ImageFont
#import math
# create Image object with the input image
 
#image = Image.open('background.png')
wordssent.reset_index(drop = False, inplace = True)
 
#Wordcloud wordcloud



class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping
       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.
       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)



compmin = min(wordssent['compound'])
compmax = max(wordssent['compound'])

compmin = -1
compmax = 1
n = 30

wordrepeats = []
wordrepeats_sent = []
for i,j,l in zip(list(wordssent[wordssent['count']>n]['words']),list(wordssent[wordssent['count']>n]['count']),list(wordssent[wordssent['count']>n]['compound'])): 
    for k in range(j):
        wordrepeats.append(i.lower())
        wordrepeats_sent.append(l)

text = " ".join(wordrepeats)

UW = []
color = []
for i,j in zip(list(wordssent[wordssent['count']>n]['words']),list(wordssent[wordssent['count']>n]['compound'])):
    UW.append(i)
    color.append('rgb('+str(int(255*(1- (j-compmin)/(compmax-compmin))))+','+str(int(155*(j-compmin)/(compmax-compmin)))+', 0)')
 
colorset = list(set(color))
color_to_words = {}
for i in colorset:
    words_by_color = []
    for j in range(len(UW)):
        if color[j] == i:
            words_by_color.append(UW[j])
        else:
            pass
    color_to_words[i] = words_by_color
        
    
grouped_color_func = SimpleGroupedColorFunc(color_to_words, 'grey')


wordcloud = WordCloud(collocations = False,width = 800, height = 500,background_color = "black").generate(text)
wordcloud.recolor(color_func=grouped_color_func)
# Display the generated image:
# the matplotlib way:

plt.figure( figsize=(8,6) )
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Uber Word Cloud")
plt.savefig("C://Users/jliv/Documents/GitHub/JLivingston01.github.io/images/uber_wordcloud.png")





#Topic Modeling TFIDF and LDA
#967 - 1088
tweetdf = pd.read_csv("C://Users/jliv/Downloads/tweets/tweets_collected.csv")

tweetdf['final_tweets2'] = tweetdf['final_tweets2'].fillna(" ")
processed_docs = tweetdf['final_tweets2'].map(preprocess)

dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 100:
        break


    
dictionary.filter_extremes(no_below=2, no_above=0.09, keep_n=1000)
#len(dictionary)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]



#Method 1 BAG OF WORDS LDA
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=4, id2word=dictionary, passes=2, workers=2)


for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
    
#Method 2   TDIDF lda
#Fit Model
tfidf = models.TfidfModel(bow_corpus)
#Apply Model
corpus_tfidf = tfidf[bow_corpus]


lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=4, id2word=dictionary, passes=2, workers=4)


model_info = []
for idx, topic in lda_model_tfidf.print_topics(-1):
    model_info.append('Topic: {} Word: {}'.format(idx, topic))
    


filename = 'C://Users/jliv/downloads/tweets/lda_tdidf.mod'
pickle.dump(lda_model_tfidf, open(filename, 'wb'))


    
for index, score in sorted(lda_model_tfidf[bow_corpus[500]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 5)))
    

topic = []
for i in list(tweetdf['final_tweets2']):
    unseen_document = i
    bow_vector = dictionary.doc2bow(preprocess(unseen_document))
    bv2 = tfidf[bow_vector]
    topic.append(sorted(lda_model_tfidf[bv2], key=lambda tup: -1*tup[1])[0][0])
    #topic.append(sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1])[0][0])
    #topic.append(sorted(lda_model[bow_corpus], key=lambda tup: -1*tup[1])[0][0])

#unseen_document = 'my driver was terrible'
#bow_vector = dictionary.doc2bow(preprocess(unseen_document))
#for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1])[0][0]:
#    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))

tweetdf['ldatopic'] = topic

tfidflist_tweetlevel = []

for i in range(len(processed_docs)):
    tfidflist_tweetlevel.append(corpus_tfidf[i])

tweetdf['tfidf'] = tfidflist_tweetlevel
tweetdf['processed_docs'] = processed_docs

tfidflen = []
doclen = []
for i in range(len(corpus_tfidf)):
    tfidflen.append(len(corpus_tfidf[i]))
    doclen.append(len(processed_docs[i]))
tweetdf['tfidflen'] = tfidflist_tweetlevel
tweetdf['doclen'] = processed_docs

tweetdf['final_tweets2_stemmed'] = tweetdf['final_tweets2_stemmed'].fillna(" ")

docss = len(tweetdf)
tfidflist_jl = []
tfidflist_jl_norm = []
for i in range(len(tweetdf)):
    twt = processed_docs[i]
    tmptfidf = []
    for j in twt:
        worddocs = len(tweetdf[tweetdf['final_tweets2_stemmed'].str.contains(j)])
        idf = np.log(docss/worddocs)
        tf = sum(1 for k in twt if k == j)/len(twt)
        tfidfres = tf*idf
        tmptfidf.append(tfidfres)
    try:
        tmptfidf_n = tmptfidf/max(tmptfidf)
    except:
        tmptfidf_n = [1]
    tfidflist_jl_norm.append(tmptfidf_n)
    tfidflist_jl.append(tmptfidf)

tfidfsums = []
for i in tfidflist_jl:
    tfidfsums.append(sum(i))    
    
tfidfvect = []
for i in tfidflist_jl:
    for j in i:
        tfidfvect.append(j)
        
plt.hist(tfidfvect, bins = 40)
plt.show()
tweetdf['tfidf_jl'] = tfidflist_jl
tweetdf['tfidflist_jl_norm'] = tfidflist_jl_norm

tweetdf.to_csv("C://users/jliv/downloads/tweets/tweets_collected_lda_tfidf.csv")


#Naive Bayes Classifier
#1095 - 1167
tweetdf = pd.read_csv("C://users/jliv/downloads/tweets/tweets_collected_lda_tfidf.csv")


corpuslist = listdir("C://Users/jliv/Downloads/tweets/corpus/")
corpusdf = pd.read_csv("C://Users/jliv/Downloads/tweets/corpus/"+corpuslist[0])
for i in corpuslist[1:]:
    corpusdf = corpusdf.append(pd.read_csv("C://Users/jliv/Downloads/tweets/corpus/"+i))
    
    


def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

corpusdf['count'] = 1
corpusdf2 = pd.pivot_table(data = corpusdf, index = ['final_tweets','label'], values = ['count'], aggfunc = 'sum')
corpusdf2.reset_index(inplace=True, drop = False)
corpusdf2['final_tweets'] = corpusdf2['final_tweets'].fillna(" ")
rand_tweets_labels = list(corpusdf2['label'])
unique_tweets = list(corpusdf2['final_tweets'])



tweet_doc = []
for i in unique_tweets:
    x = i.lower()
    tweet_doc.append(x.split())
    

unique_words = []
for i in unique_tweets:
    x = i.split()
    for j in x:
        unique_words.append(j.lower())   

       
UW = pd.DataFrame()
UW['unique_words'] = unique_words
UW['count'] = 1
UW_piv = pd.pivot_table(data = UW, values = 'count', index = 'unique_words', aggfunc = 'sum')
UW_piv = UW_piv.sort_values(by = 'count', ascending = False)
unique_words2 = list(UW_piv.index)
dropping = ['i','the','``',"''"]
unique_words3 = [i for i in unique_words2 if i not in dropping]
word_features =unique_words3[:300]

ww = UW_piv.copy()

ww.reset_index(inplace= True)
ww = list(ww[(ww['count'] >= 20)&(ww['count'] <= 30)]['unique_words'])
word_features = list(word_features)+ww

#word_features =unique_words3


from random import shuffle
tweet_featset = [(document_features(d), c) for (d,c) in zip(tweet_doc,rand_tweets_labels)]   
tweet_featset2 = tweet_featset
shuffle(tweet_featset2)

train_set, test_set = tweet_featset[:len(tweet_featset)], tweet_featset[:len(tweet_featset)]
#train_set, test_set = tweet_featset2[:int(len(tweet_featset2)*.75)], tweet_featset2[int(len(tweet_featset2)*.75):]

classifier = nltk.NaiveBayesClassifier.train(train_set)
classifier.show_most_informative_features(30)
nltk.classify.accuracy(classifier, test_set)



unique_tweets = tweetdf['final_tweets']
tweets_split = []
for i in unique_tweets:
    tweets_split.append(i.split())
    
new_labels = []
probs = []
for i in tweets_split:
    test_features = [(document_features(i), 'test')]
    new_labels.append(classifier.classify(test_features[0][0]))
    dist = classifier.prob_classify(test_features[0][0])
    probs.append(dist.prob(classifier.classify(test_features[0][0])))

tweetdf['NB_label'] = new_labels
tweetdf['NB_label_prob'] = probs

tweetdf.to_csv("C://users/jliv/downloads/tweets/tweets_NBC_Labels.csv")
len(tweetdf[tweetdf['NB_label']=='promotional']['text'])


#Time Sentiment and Topic Analysis by Naive Bayes Label
#1171 - 1321
    
tweetdf = pd.read_csv("C://users/jliv/downloads/tweets/tweets_NBC_Labels.csv")

lyftdf1 = tweetdf[tweetdf['comp']=='lyft']
uberdf1 = tweetdf[tweetdf['comp']=='uber']

ubermean = np.mean(uberdf1['compound'])
lyftmean = np.mean(lyftdf1['compound']) 

lyftscoresdf = pd.pivot_table(data = lyftdf1, index = ['NB_label'], values = ['compound'], aggfunc = 'mean' )
uberscoresdf = pd.pivot_table(data = uberdf1, index = ['NB_label'], values = ['compound'], aggfunc = 'mean' )

lyftscoresdf.reset_index(inplace = True, drop = False)
uberscoresdf.reset_index(inplace = True, drop = False)
fig = plt.figure(figsize = (10.5,8))
plt.plot(lyftscoresdf['compound'], label = 'Lyft Average Sentiment')
plt.plot(uberscoresdf['compound'], label = 'Uber Average Sentiment')
plt.title('Naive Bayes Classifier Topics: Lyft Promo Tweets are more positive, Otherwise Similar Sentiment')
plt.legend(loc = 2)
plt.xticks(range(4),list(lyftscoresdf['NB_label']), rotation = 45)
plt.savefig("C://Users/jliv/Documents/GitHub/JLivingston01.github.io/images/Topics_Sentiment_NBC.png")
plt.show()

lyftscoresdf = pd.pivot_table(data = lyftdf1, index = ['date'], values = ['compound'], aggfunc = 'mean' )
uberscoresdf = pd.pivot_table(data = uberdf1, index = ['date'], values = ['compound'], aggfunc = 'mean' )

lyftscoresdf.reset_index(inplace = True, drop = False)
uberscoresdf.reset_index(inplace = True, drop = False)
    

fig = plt.figure(figsize = (10,8))
plt.plot(lyftscoresdf['compound'], label = 'Lyft Average Sentiment')
plt.plot(uberscoresdf['compound'], label = 'Uber Average Sentiment')
plt.xticks(range(17),list(lyftscoresdf['date']), rotation = 45)
plt.legend(loc = 2)
plt.title('NBC Topic ALL: Few meaningful differences, lower dives for Uber Sentiment')
plt.savefig("C://Users/jliv/Documents/GitHub/JLivingston01.github.io/images/Time_Sentiment_NBC.png")
plt.show()





topic = 'financial'

lyftdf2 = lyftdf1[lyftdf1['NB_label']==topic]
uberdf2 = uberdf1[uberdf1['NB_label']==topic]

#lyftdf2 = lyftdf
#uberdf2 = uberdf

lyftscoresdf = pd.pivot_table(data = lyftdf2, index = ['date'], values = ['compound'], aggfunc = 'mean' )
uberscoresdf = pd.pivot_table(data = uberdf2, index = ['date'], values = ['compound'], aggfunc = 'mean' )

lyftscoresdf.reset_index(inplace = True, drop = False)
uberscoresdf.reset_index(inplace = True, drop = False)
    

fig = plt.figure(figsize = (10,8))
plt.plot(lyftscoresdf['compound'], label = 'Lyft Average Sentiment')
plt.plot(uberscoresdf['compound'], label = 'Uber Average Sentiment')
plt.xticks(range(17),list(lyftscoresdf['date']), rotation = 45)
plt.legend(loc = 2)
plt.title('NBC Topic '+str(topic)+': Highly Correlated, Not Meaningfully Different')
plt.savefig("C://Users/jliv/Documents/GitHub/JLivingston01.github.io/images/Time_Sentiment_Financial_NBC.png")
plt.show()




topic = 'service'

lyftdf2 = lyftdf1[lyftdf1['NB_label']==topic]
uberdf2 = uberdf1[uberdf1['NB_label']==topic]

#lyftdf2 = lyftdf
#uberdf2 = uberdf

lyftscoresdf = pd.pivot_table(data = lyftdf2, index = ['date'], values = ['compound'], aggfunc = 'mean' )
uberscoresdf = pd.pivot_table(data = uberdf2, index = ['date'], values = ['compound'], aggfunc = 'mean' )

lyftscoresdf.reset_index(inplace = True, drop = False)
uberscoresdf.reset_index(inplace = True, drop = False)
    

fig = plt.figure(figsize = (10,8))
plt.plot(lyftscoresdf['compound'], label = 'Lyft Average Sentiment')
plt.plot(uberscoresdf['compound'], label = 'Uber Average Sentiment')
plt.xticks(range(17),list(lyftscoresdf['date']), rotation = 45)
plt.legend(loc = 1)
plt.title('NBC Topic '+str(topic)+': Highly Correlated, Not Meaningfully Different')
plt.savefig("C://Users/jliv/Documents/GitHub/JLivingston01.github.io/images/Time_Sentiment_Service_NBC.png")
plt.show()





topic = 'promotional'

lyftdf2 = lyftdf1[lyftdf1['NB_label']==topic]
uberdf2 = uberdf1[uberdf1['NB_label']==topic]

#lyftdf2 = lyftdf
#uberdf2 = uberdf

lyftscoresdf = pd.pivot_table(data = lyftdf2, index = ['date'], values = ['compound'], aggfunc = 'mean' )
uberscoresdf = pd.pivot_table(data = uberdf2, index = ['date'], values = ['compound'], aggfunc = 'mean' )

lyftscoresdf.reset_index(inplace = True, drop = False)
uberscoresdf.reset_index(inplace = True, drop = False)

lumerge = pd.merge(left = lyftscoresdf, right = uberscoresdf, on = 'date', how = 'left')
    

fig = plt.figure(figsize = (10,8))
plt.plot(lumerge['compound_x'], label = 'Lyft Average Sentiment')
plt.plot(lumerge['compound_y'], label = 'Uber Average Sentiment')
plt.xticks(range(17),list(lyftscoresdf['date']), rotation = 45)
plt.ylim((-.2,1.2))
plt.legend(loc = 2)
plt.title('NBC Topic '+str(topic)+': Few Uber Tweets Classified Promotional')
plt.savefig("C://Users/jliv/Documents/GitHub/JLivingston01.github.io/images/Time_Sentiment_Promotional_NBC.png")
plt.show()



topic = 'news'

lyftdf2 = lyftdf1[lyftdf1['NB_label']==topic]
uberdf2 = uberdf1[uberdf1['NB_label']==topic]

#lyftdf2 = lyftdf
#uberdf2 = uberdf

lyftscoresdf = pd.pivot_table(data = lyftdf2, index = ['date'], values = ['compound'], aggfunc = 'mean' )
uberscoresdf = pd.pivot_table(data = uberdf2, index = ['date'], values = ['compound'], aggfunc = 'mean' )

lyftscoresdf.reset_index(inplace = True, drop = False)
uberscoresdf.reset_index(inplace = True, drop = False)
    

fig = plt.figure(figsize = (10,8))
plt.plot(lyftscoresdf['compound'], label = 'Lyft Average Sentiment')
plt.plot(uberscoresdf['compound'], label = 'Uber Average Sentiment')
plt.xticks(range(17),list(lyftscoresdf['date']), rotation = 45)
plt.legend(loc = 2)
plt.title('Topic '+str(topic)+': Not Meaningfully Different')
plt.savefig("C://Users/jliv/Documents/GitHub/JLivingston01.github.io/images/Time_Sentiment_News_NBC.png")
plt.show()


##Classification with Tensor Flow and Sentence Convolution
#1327 - 1437
#Total Corpus of Words
txt_twt = tweetdf['processed_docs']
corpusdf = pd.read_csv("C://Users/jliv/Downloads/tweets/corpus/"+corpuslist[0])
for i in corpuslist[1:]:
    corpusdf = corpusdf.append(pd.read_csv("C://Users/jliv/Downloads/tweets/corpus/"+i))

corpusdf['final_tweets'] = corpusdf['final_tweets'].fillna(" ")
corpusdf['processed_docs'] = corpusdf['final_tweets'].map(preprocess)
twtcrp = corpusdf['processed_docs']

#txt_twt = txt_twt.fillna([" "])
#twtcrp  = twtcrp.fillna([" "])
wds = []
for i in txt_twt:
    for j in i:
        try:
            wds.append(j)
        except:
            pass
        
for i in twtcrp:
    for j in i:
        try:
            wds.append(j)
        except:
            pass
        
dat = pd.DataFrame()
dat['wds'] = wds
dat['cnt'] = 1
UWs = pd.pivot_table(data = dat, index = ['wds'],values = ['cnt'], aggfunc = 'sum' )
UWs['rng'] = list(range(len(UWs)))
UWs.drop(['cnt'], inplace = True, axis = 1)

uwdict = UWs.to_dict()
uwdict = uwdict['rng']
twtcrp.reset_index(inplace= True, drop = True)
ls = []
for i in twtcrp:
    ls.append(len(i))
maxes = max(ls)


nndocs = []
for i in twtcrp:
    tmp = []
    for j in i:
        try:
            tmp.append(uwdict[j])
        except:
            tmp.append(-1)
    lentemp = len(tmp)
    for k in range(maxes-lentemp):
        tmp.append(-1)
    nndocs.append(np.array(tmp))
    

nndf = pd.DataFrame(nndocs)
nndocs2 = np.array(nndf)
nndocs2.shape


labs = np.array(corpusdf['label'])
labs.shape

#nndocs = nndocs.transpose()

encoder2 = LabelEncoder()
encoder2.fit(labs)
encoded_Ytrain = encoder2.transform(labs)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_labs = np_utils.to_categorical(encoded_Ytrain)
#dummy_labs = dummy_labs.reshape(13898, 4,1)
dummy_labs.shape

nndocs2= nndocs2.reshape(13898,1,38)
dummy_labs= dummy_labs.reshape(13898,1,4)
nndocs2[0].shape
# kernel_size=(4,1)


nndocs2= nndocs2.reshape(13898,1,1,1,38)
dummy_labs= dummy_labs.reshape(13898,1,1,1,4)

nndocs2= nndocs2.reshape(13898,1,38)
dummy_labs= dummy_labs.reshape(13898,1,4)
IS = nndocs2[0].shape
model = keras.Sequential()
'''model.add(keras.layers.Conv1D(40,kernel_size = (4),activation='sigmoid',input_shape=(None,787), padding='same'))
model.add(keras.layers.Dense(299, activation=tensorflow.nn.relu))
model.add(keras.layers.Dense(15, activation=tensorflow.nn.relu))
model.add(keras.layers.Dense(4, activation=tensorflow.nn.relu))'''
'''model.add(keras.layers.Conv1D(40,kernel_size = (6),strides = 1,activation='relu',input_shape=(None,38), padding='same'))
model.add(keras.layers.Dense(38, activation=tensorflow.nn.relu))
model.add(keras.layers.Conv1D(40,kernel_size = (6),strides = 1,activation='relu',input_shape=(None,38), padding='same'))
model.add(keras.layers.Dense(38, activation=tensorflow.nn.relu))
model.add(keras.layers.Conv1D(40,kernel_size = (6),strides = 1,activation='relu',input_shape=(None,38), padding='same'))
model.add(keras.layers.Dense(38, activation=tensorflow.nn.relu))
model.add(keras.layers.Conv1D(40,kernel_size = (6),strides = 1,activation='relu',input_shape=(None,38), padding='same'))
model.add(keras.layers.Dense(38, activation=tensorflow.nn.relu))
model.add(keras.layers.Dense(4, activation=tensorflow.nn.sigmoid))'''
#model.add(keras.layers.ConvLSTM2D(120,input_shape=(None,None,None,38), kernel_size = (6), strides=(1), padding='same', data_format=None, dilation_rate=1, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=True, go_backwards=True, stateful=False, dropout=0.0, recurrent_dropout=0.0))
model.add(keras.layers.LSTM(120, activation='tanh', recurrent_activation='hard_sigmoid', \
                            use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
                            bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, 
                            recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
                            kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, 
                            dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=True, 
                            return_state=False, go_backwards=False, stateful=False, unroll=False))
model.add(keras.layers.Dense(120, activation=tensorflow.nn.relu))
model.add(keras.layers.Dense(38, activation=tensorflow.nn.relu))
model.add(keras.layers.Dense(4, activation=tensorflow.nn.sigmoid))
model.compile(optimizer=tensorflow.train.AdamOptimizer(), 
              loss=losses.mean_squared_error,
              metrics=['accuracy'])
#model.summary()

model.fit(nndocs2, dummy_labs, epochs=300, verbose = 1)



predresult = model.predict(nndocs2)


result_vecttrain = []

for i in predresult:
    result_vecttrain.append(np.argmax(i))
        
ylabnum = []
for i in dummy_labs:
    ylabnum.append(np.argmax(i))
    
resdf = pd.DataFrame()
resdf['ylab'] = ylabnum
resdf['pred'] = result_vecttrain
resdf['res'] = np.where(resdf['ylab']==resdf['pred'],1,0)


#This model has no predictive power.. redoing with common word TF matrix
#np.mean(resdf['res'] )

#model.summary()



#NN with TF word mapping
#1449 - 1749
txt_twt = list(tweetdf['processed_docs'])
corpusdf = pd.read_csv("C://Users/jliv/Downloads/tweets/corpus/"+corpuslist[0])
for i in corpuslist[1:]:
    corpusdf = corpusdf.append(pd.read_csv("C://Users/jliv/Downloads/tweets/corpus/"+i))

corpusdf['final_tweets'] = corpusdf['final_tweets'].fillna(" ")
corpusdf['processed_docs'] = corpusdf['final_tweets'].map(preprocess)
twtcrp = list(corpusdf['processed_docs'])
twtcrp_fin = list(corpusdf['final_tweets'])
#txt_twt = txt_twt.fillna([" "])
#twtcrp  = twtcrp.fillna([" "])
wds = []
for i in txt_twt:
    for j in i:
        try:
            wds.append(j)
        except:
            pass
wds = []        
for i in twtcrp:
    for j in i:
        try:
            wds.append(j)
        except:
            pass
        
dat = pd.DataFrame()
dat['wds'] = wds
dat['cnt'] = 1
UWs = pd.pivot_table(data = dat, index = ['wds'],values = ['cnt'], aggfunc = 'sum' )
UWs = UWs.sort_values(by = 'cnt', ascending = False)
UWs.reset_index(inplace= True, drop = False)
UWs2 = UWs[:300]

ww = UWs.copy()

ww.reset_index(inplace= True)
ww = list(ww[(ww['cnt'] >= 20)&(ww['cnt'] <= 30)]['wds'])
WL = list(UWs2['wds'])+ww


twtcrp = list(twtcrp)
#==============================================================================
# Xdf = pd.DataFrame()
# for i in WL:
#     Xdf[i] = [0]
#==============================================================================

Xdf = pd.DataFrame()
Xdf['twtcrp'] = twtcrp
Xdf['twtcrp_fin'] = twtcrp_fin
for i in WL:
    Xdf[i] = np.where(Xdf['twtcrp_fin'].str.contains(i),1,0)

#Xdf.drop(['twtcrp_fin','twtcrp','y'], inplace= True, axis = 1)
Xdf.drop(['twtcrp_fin','twtcrp'], inplace= True, axis = 1)
Xarray = np.array(Xdf)



labs = np.array(corpusdf['label'])
labs.shape


#nndocs = nndocs.transpose()

encoder2 = LabelEncoder()
encoder2.fit(labs)
encoded_Ytrain = encoder2.transform(labs)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_labs = np_utils.to_categorical(encoded_Ytrain)
#dummy_labs = dummy_labs.reshape(13898, 4,1)
dummy_labs.shape
Xarray.shape
Xarray= Xarray.reshape(13898,1,787)
dummy_labs= dummy_labs.reshape(13898,1,4)
Xarray[0].shape
# kernel_size=(4,1)
IS = nndocs2[0].shape

model = keras.Sequential()
'''model.add(keras.layers.Conv1D(40,kernel_size = (4),activation='sigmoid',input_shape=(None,787), padding='same'))
model.add(keras.layers.Dense(299, activation=tensorflow.nn.relu))
model.add(keras.layers.Dense(15, activation=tensorflow.nn.relu))
model.add(keras.layers.Dense(4, activation=tensorflow.nn.relu))'''
model.add(keras.layers.Dense(787, activation=tensorflow.nn.relu))
model.add(keras.layers.Dense(300, activation=tensorflow.nn.sigmoid))
model.add(keras.layers.Dense(100, activation=tensorflow.nn.sigmoid))
model.add(keras.layers.Dense(25, activation=tensorflow.nn.sigmoid))
model.add(keras.layers.Dense(4, activation=tensorflow.nn.sigmoid))
#model.add(keras.layers.Dense(3, activation=tensorflow.keras.activations.linear))
model.compile(optimizer=tensorflow.train.AdamOptimizer(), 
              loss=losses.mean_squared_error,
              metrics=['accuracy'])
#model.summary()

model.fit(Xarray, dummy_labs, epochs=100, verbose = 1)



predresult = model.predict(Xarray)


result_vecttrain = []

for i in predresult:
    result_vecttrain.append(np.argmax(i))
        
ylabnum = []
for i in dummy_labs:
    ylabnum.append(np.argmax(i))
    
resdf = pd.DataFrame()
resdf['ylab'] = ylabnum
resdf['pred'] = result_vecttrain
resdf['res'] = np.where(resdf['ylab']==resdf['pred'],1,0)
np.mean(resdf['res'])


ylabs = pd.DataFrame()
ylabs['Y'] = labs
ylabs['num'] = ylabnum
ylabs = pd.pivot_table(data = ylabs, index = ['Y'], values = ['num'], aggfunc = 'mean')
ylabs.reset_index(inplace= True, drop = False)
ylabs.set_index('num', inplace=True)
labelmap = ylabs.to_dict()['Y']




txt_twt = list(tweetdf['processed_docs'])
txt_twt_fin = list(tweetdf['final_tweets'])
Xdf = pd.DataFrame()
Xdf['txt_twt'] = txt_twt
Xdf['txt_twt_fin'] = txt_twt_fin
for i in WL:
    Xdf[i] = np.where(Xdf['txt_twt_fin'].str.contains(i),1,0)

#Xdf.drop(['txt_twt_fin','txt_twt','y'], inplace= True, axis = 1)
Xdf.drop(['txt_twt_fin','txt_twt'], inplace= True, axis = 1)
Xtest = np.array(Xdf)


Xtest= Xtest.reshape(3400,1,787)

predresult = model.predict(Xtest)


result_vecttrain = []

for i in predresult:
    result_vecttrain.append(np.argmax(i))
        
    
resdf = pd.DataFrame()
resdf['pred'] = result_vecttrain


tweetdf['nn_label'] = resdf['pred']

tweetdf['nn_label_val'] = tweetdf['nn_label'].map(labelmap)


pd.pivot_table(data = tweetdf, index = ['NB_label'], values = ['nn_label_lstm'], aggfunc = 'count')
pd.pivot_table(data = tweetdf, index = ['nn_label_val'], values = ['nn_label'], aggfunc = 'count')

tweetdf.to_csv("C://Users/jliv/downloads/tweets/tweets_final_models.csv")


#Graph classified tweets sentiment 
tweetdf = pd.read_csv("C://Users/jliv/downloads/tweets/tweets_final_models.csv")
lyftdf1 = tweetdf[tweetdf['comp']=='lyft']
uberdf1 = tweetdf[tweetdf['comp']=='uber']


topic = 'financial'

lyftdf2 = lyftdf1[lyftdf1['nn_label_val']==topic]
uberdf2 = uberdf1[uberdf1['nn_label_val']==topic]

#lyftdf2 = lyftdf
#uberdf2 = uberdf

lyftscoresdf = pd.pivot_table(data = lyftdf2, index = ['date'], values = ['compound'], aggfunc = 'mean' )
uberscoresdf = pd.pivot_table(data = uberdf2, index = ['date'], values = ['compound'], aggfunc = 'mean' )

lyftscoresdf.reset_index(inplace = True, drop = False)
uberscoresdf.reset_index(inplace = True, drop = False)
    

fig = plt.figure(figsize = (10.5,7.5))
plt.plot(lyftscoresdf['compound'], label = 'Lyft Average Sentiment')
plt.plot(uberscoresdf['compound'], label = 'Uber Average Sentiment')
plt.xticks(range(17),list(lyftscoresdf['date']), rotation = 45)
plt.legend(loc = 2)
plt.title('NN Topic '+str(topic)+': Not Meaningfully Different')
plt.savefig("C://Users/jliv/Documents/GitHub/JLivingston01.github.io/images/Time_Sentiment_Financial_NN.png")
plt.show()




topic = 'service'

lyftdf2 = lyftdf1[lyftdf1['nn_label_val']==topic]
uberdf2 = uberdf1[uberdf1['nn_label_val']==topic]

#lyftdf2 = lyftdf
#uberdf2 = uberdf

lyftscoresdf = pd.pivot_table(data = lyftdf2, index = ['date'], values = ['compound'], aggfunc = 'mean' )
uberscoresdf = pd.pivot_table(data = uberdf2, index = ['date'], values = ['compound'], aggfunc = 'mean' )

lyftscoresdf.reset_index(inplace = True, drop = False)
uberscoresdf.reset_index(inplace = True, drop = False)
    

fig = plt.figure(figsize = (10.5,7.5))
plt.plot(lyftscoresdf['compound'], label = 'Lyft Average Sentiment')
plt.plot(uberscoresdf['compound'], label = 'Uber Average Sentiment')
plt.xticks(range(17),list(lyftscoresdf['date']), rotation = 45)
plt.legend(loc = 2)
plt.title('NN Topic '+str(topic)+': Not Meaningfully Different')
plt.savefig("C://Users/jliv/Documents/GitHub/JLivingston01.github.io/images/Time_Sentiment_service_NN.png")
plt.show()





topic = 'promotional'

lyftdf2 = lyftdf1[lyftdf1['nn_label_val']==topic]
uberdf2 = uberdf1[uberdf1['nn_label_val']==topic]

#lyftdf2 = lyftdf
#uberdf2 = uberdf

lyftscoresdf = pd.pivot_table(data = lyftdf2, index = ['date'], values = ['compound'], aggfunc = 'mean' )
uberscoresdf = pd.pivot_table(data = uberdf2, index = ['date'], values = ['compound'], aggfunc = 'mean' )

lyftscoresdf.reset_index(inplace = True, drop = False)
uberscoresdf.reset_index(inplace = True, drop = False)
    

lumerge = pd.merge(left = lyftscoresdf, right = uberscoresdf, on = 'date', how = 'left')
    

fig = plt.figure(figsize = (10.5,7.5))
plt.plot(lumerge['compound_x'], label = 'Lyft Average Sentiment')
plt.plot(lumerge['compound_y'], label = 'Uber Average Sentiment')
plt.xticks(range(17),list(lyftscoresdf['date']), rotation = 45)
plt.ylim((-.2,1.2))
plt.legend(loc = 2)
plt.title('NN Topic '+str(topic)+': Lyft with Stronger Sustained Sentiment, Uber not as Promotional')
plt.savefig("C://Users/jliv/Documents/GitHub/JLivingston01.github.io/images/Time_Sentiment_promotional_NN.png")
plt.show()



topic = 'news'

lyftdf2 = lyftdf1[lyftdf1['nn_label_val']==topic]
uberdf2 = uberdf1[uberdf1['nn_label_val']==topic]

#lyftdf2 = lyftdf
#uberdf2 = uberdf

lyftscoresdf = pd.pivot_table(data = lyftdf2, index = ['date'], values = ['compound'], aggfunc = 'mean' )
uberscoresdf = pd.pivot_table(data = uberdf2, index = ['date'], values = ['compound'], aggfunc = 'mean' )

lyftscoresdf.reset_index(inplace = True, drop = False)
uberscoresdf.reset_index(inplace = True, drop = False)
    

fig = plt.figure(figsize = (10.5,7.5))
plt.plot(lyftscoresdf['compound'], label = 'Lyft Average Sentiment')
plt.plot(uberscoresdf['compound'], label = 'Uber Average Sentiment')
plt.xticks(range(17),list(lyftscoresdf['date']), rotation = 45)
plt.legend(loc = 2)
plt.title('NN Topic '+str(topic)+': Not Meaningfully Different')
plt.savefig("C://Users/jliv/Documents/GitHub/JLivingston01.github.io/images/Time_Sentiment_news_NN.png")
plt.show()







lyftscoresdf = pd.pivot_table(data = lyftdf1, index = ['nn_label_val'], values = ['compound'], aggfunc = 'mean' )
uberscoresdf = pd.pivot_table(data = uberdf1, index = ['nn_label_val'], values = ['compound'], aggfunc = 'mean' )

lyftscoresdf.reset_index(inplace = True, drop = False)
uberscoresdf.reset_index(inplace = True, drop = False)
    

fig = plt.figure(figsize = (10.5,7.5))
plt.plot(lyftscoresdf['compound'], label = 'Lyft Average Sentiment')
plt.plot(uberscoresdf['compound'], label = 'Uber Average Sentiment')
plt.xticks(range(4),list(lyftscoresdf['nn_label_val']), rotation = 45)
plt.legend(loc = 2)
plt.title('NN Topic Model: Lyft stronger in Financial, Promotional and Service Sentiment')
plt.savefig("C://Users/jliv/Documents/GitHub/JLivingston01.github.io/images/Sentiment_Topics_NN.png")
plt.show()



#lyftdf2 = lyftdf
#uberdf2 = uberdf

lyftscoresdf = pd.pivot_table(data = lyftdf1, index = ['date'], values = ['compound'], aggfunc = 'mean' )
uberscoresdf = pd.pivot_table(data = uberdf1, index = ['date'], values = ['compound'], aggfunc = 'mean' )

lyftscoresdf.reset_index(inplace = True, drop = False)
uberscoresdf.reset_index(inplace = True, drop = False)
    

fig = plt.figure(figsize = (10.5,7.5))
plt.plot(lyftscoresdf['compound'], label = 'Lyft Average Sentiment')
plt.plot(uberscoresdf['compound'], label = 'Uber Average Sentiment')
plt.xticks(range(17),list(lyftscoresdf['date']), rotation = 45)
plt.legend(loc = 2)
plt.title('NN Topic All: Not Meaningfully Different')
plt.savefig("C://Users/jliv/Documents/GitHub/JLivingston01.github.io/images/Time_Sentiment_NN.png")
plt.show()