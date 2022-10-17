
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 150)

import gensim
import gensim.corpora as corpora

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


filename = "c:/users/jliv/downloads/spy_merge2.csv"
dat = pd.read_csv(filename)

dat = dat[['name','short_description']].copy()
dat = dat[~dat['short_description'].isna()].copy()

alnum = 'qwertyuiopasdfghjklzxcvbnm -'
def clean(x):
    x = ''.join([i.lower() for i in x if i.lower() in alnum])
    return x
    
dat['clean']=dat['short_description'].apply(clean)
dat['clean'] = dat['clean'].str.replace('-',' ',regex=False)

all_text = dat['clean'].str.cat(sep=' ')

all_text = pd.DataFrame({'word':all_text.split()})
all_text['cnt']=1
all_text = all_text.groupby(['word']).agg({'cnt':'count'}).reset_index()
all_text = all_text[all_text['cnt']>3].copy()


all_words = list(all_text['word'])

len(all_words)

def filter_words(x):
    x = [i for i in x if i in all_words]
    return x



stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use','game',
                   'play','player','get'
                   ])

def remove_stopwords(x):
    x = [i for i in x if i not in stop_words]
    return x




dat['tokenized']=dat['clean'].str.split()
#dat['tokenized'] = dat['tokenized'].apply(filter_words)
dat['tokenized'] = dat['tokenized'].apply(remove_stopwords)

id2word = corpora.Dictionary(list(dat['tokenized']))


corpus = [id2word.doc2bow(text) for text in dat['tokenized']]


# number of topics
num_topics = 10
# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)


print(lda_model.print_topics())

def predict_topics(x):    
    return lda_model[x]

topics = pd.DataFrame({'corpus':corpus})
topics['topics'] = topics['corpus'].apply(predict_topics)
topics['topics'] = topics['topics'].apply(dict)

for i in range(0,num_topics):
    def get_topics(x):
        try:
            return x[i]
        except:
            return 0
    
    topics[i] = topics['topics'].apply(get_topics)

topics['topic_chosen'] = np.argmax(
    np.array(
        topics[list(range(0,num_topics))]
        ),axis=1)

definitions = dict(lda_model.print_topics())

dat.reset_index(inplace=True)
dat['topic_chosen']=topics['topic_chosen']
dat['topic_definition'] = dat['topic_chosen'].map(definitions)

dat.to_csv("c:/users/jliv/downloads/game_topics.csv",index=False)

dat[['name','short_description','topic_chosen','topic_definition']
    ].drop_duplicates()

def inspect(n):
    print(dat['short_description'][n] , dat['topic_definition'][n])
    
inspect(100)