# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 10:32:54 2019

@author: jliv
"""
import bs4
import urllib
import re
import numpy as np
from keras.preprocessing.text import one_hot
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from collections import Counter
import keras
import tensorflow

"""This program scrapes officequotes.net for office scripts, turns each script into arrays of hashed rolling 
5-grams and an array of the 6th word, and develops an LSTM text generator from the office language."""

'''Obtain paths to scripts'''
html_page = urllib.request.urlopen("http://officequotes.net/")
soup = bs4.BeautifulSoup(html_page)
links = soup.findAll("a")
hrefs = [i.get('href') for i in links]
hrefs = hrefs[hrefs.index('no1-01.php'):hrefs.index('webisodes.php')]

#Pull and process first script, initializing arrays
html_page = urllib.request.urlopen("http://officequotes.net/"+hrefs[0])
soup = bs4.BeautifulSoup(html_page)
    
quotes = soup.findAll('div', {"class": "quote"})

txt = [i.get_text() for i in quotes]
txt = " ".join(txt)

txt = re.sub(r'Deleted Scene [0-9]+', ' ', txt)
txt = txt.replace("\n&nbsp\n"," ")
txt = txt.replace("\n"," ")
txt = txt.replace("\\","")
txt = txt.replace("#"," ")
txt = txt.replace("&"," and ")
txt = txt.replace("â€¦"," ")
txt = txt.replace("â€™"," ")
txt = txt.replace("â€œ"," ")
txt = txt.replace("â€"," ")
txt = txt.replace(";"," cmm ")
txt = txt.replace("..."," pp ")
txt = txt.replace("."," pp ")
txt = txt.replace("!"," ee ")
txt = txt.replace("?"," qq ")
txt = txt.replace(":"," cc ")
txt = txt.replace(","," cmm ")
txt = txt.replace("-"," ")
txt = txt.replace("'","")
txt = txt.replace("["," bbA ")
txt = txt.replace("]"," bbB ")
txt = txt.replace("("," bbA ")
txt = txt.replace(")"," bbB ")
txt = txt.replace("$"," ds ")
txt = txt.replace('"'," qt ")
txt = txt.replace('/'," ")
txt = txt.replace('%'," percent ")
txt = txt.replace('‘',"")
txt = txt.replace('’',"")
txt = txt.replace('“'," ")
txt = txt.replace('”'," ")
txt = txt.replace('…'," pp ")
txt = txt.replace('—'," ")
txt = txt.replace('–'," ")
txt = txt.replace('é',"e")
txt = txt.replace('ñ',"n")
txt = txt.replace('ü',"u")
txt = txt.replace('{'," ")
txt = txt.replace('}'," ")
txt = txt.replace('_'," ")
txt = txt.replace('@'," ")
txt = txt.replace('='," ")
txt = txt.replace('+'," ")
txt = txt.replace('*'," ")
txt = txt.lower()
txt = txt.split()

len(txt)

txtjoined = " ".join(txt)
vocablen = 22034

txtjoined = txtjoined.replace('would you like some googi googi qq i have some very delicious googi cmm googi cmm only 99 cents plus tax pp try my googi cmm googi pp bba lowering voice bbb try my googi cmm googi pp bba high pitched voice bbb try my googi cmm googi pp try my pp '," ")



encoded_docs = one_hot(txtjoined, vocablen)


training = []
tw=[]
y = []
yw = []

for i in range(len(encoded_docs)-2):
    training.append([encoded_docs[i]])
    tw.append([txt[i]])
    y.append(encoded_docs[i+1])
    yw.append(txt[i+1])

training = np.array(training)
yw = np.array(yw)
    
#Load remaining scripts and append resulting arrays to training and yw
for k in hrefs[1:]:
    html_page = urllib.request.urlopen("http://officequotes.net/"+k)
    soup = bs4.BeautifulSoup(html_page)
    
    quotes = soup.findAll('div', {"class": "quote"})

    txt = [i.get_text() for i in quotes]
    txt = " ".join(txt)

    txt = re.sub(r'Deleted Scene [0-9]+', ' ', txt)
    txt = txt.replace("\n&nbsp\n"," ")
    txt = txt.replace("\n"," ")
    txt = txt.replace("\\","")
    txt = txt.replace("#"," ")
    txt = txt.replace("â€¦"," ")
    txt = txt.replace("â€œ"," ")
    txt = txt.replace("â€™"," ")
    txt = txt.replace("â€"," ")
    txt = txt.replace("&"," and ")
    txt = txt.replace(";"," cmm ")
    txt = txt.replace("..."," pp ")
    txt = txt.replace("."," pp ")
    txt = txt.replace("!"," ee ")
    txt = txt.replace("?"," qq ")
    txt = txt.replace(":"," cc ")
    txt = txt.replace(","," cmm ")
    txt = txt.replace("-"," ")
    txt = txt.replace("'","")
    txt = txt.replace("["," bbA ")
    txt = txt.replace("]"," bbB ")
    txt = txt.replace("("," bbA ")
    txt = txt.replace(")"," bbB ")
    txt = txt.replace("$"," ds ")
    txt = txt.replace('"'," qt ")
    txt = txt.replace('/'," ")
    txt = txt.replace('%'," percent ")
    txt = txt.replace('‘',"")
    txt = txt.replace('’',"")
    txt = txt.replace('“'," ")
    txt = txt.replace('”'," ")
    txt = txt.replace('…'," pp ")
    txt = txt.replace('—'," ")
    txt = txt.replace('–'," ")
    txt = txt.replace('é',"e")
    txt = txt.replace('ñ',"n")
    txt = txt.replace('ü',"u")
    txt = txt.replace('{'," ")
    txt = txt.replace('}'," ")
    txt = txt.replace('_'," ")
    txt = txt.replace('@'," ")
    txt = txt.replace('='," ")
    txt = txt.replace('+'," ")
    txt = txt.replace('*'," ")
    txt = txt.lower()

    txt = txt.split()



    #txt = txt[250000:325000]

    txtjoined = " ".join(txt)



    encoded_docs = one_hot(txtjoined, vocablen)


    training1 = []
    tw1=[]
    y1 = []
    yw1 = []

    for i in range(len(encoded_docs)-2):
        training1.append([encoded_docs[i]])
        tw1.append([txt[i]])
        y1.append(encoded_docs[i+1])
        yw1.append(txt[i+1])
    
    training1 = np.array(training1)
    yw1 = np.array(yw1)
    
    training = np.concatenate((training,training1),axis = 0)
    yw = np.concatenate((yw,yw1))
    
#training.shape
#yw.shape

labcounts = Counter(yw.tolist())
#labcounts['michael']
labcounts = pd.DataFrame.from_dict(labcounts,orient='index', columns = ['Counts'])
labcounts = labcounts.sort_values(by = 'Counts', ascending = False)

words10 = list(labcounts[labcounts['Counts'] > 30].index)
textsum = pd.DataFrame(training)
textsum['yw'] = yw
textsum = textsum[textsum['yw'].isin(words10)]

yw1 = list(textsum['yw'])
training1 = np.array(textsum.drop(['yw'], axis = 1))
training1.shape

labs = np.array(yw1)
encoder2 = LabelEncoder()
encoder2.fit(labs)
encoded_Ytrain = encoder2.transform(labs)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_labs = np_utils.to_categorical(encoded_Ytrain)

ylabnum = np.argmax(dummy_labs, axis = 1)
#ylabnum = []
#for i in dummy_labs:
#    ylabnum.append(np.argmax(i))
dummy_labs.shape
ylabnum.shape

ylabs = pd.DataFrame()
ylabs['Y'] = labs
ylabs['num'] = ylabnum
ylabs = pd.pivot_table(data = ylabs, index = ['Y'], values = ['num'], aggfunc = 'mean')
ylabs.reset_index(inplace= True, drop = False)
ylabs.set_index('num', inplace=True)
labelmap = ylabs.to_dict()['Y']

model = keras.Sequential()

#10 best
model.add(keras.layers.Embedding(vocablen, 200, input_length=1))
#model.add(keras.layers.LSTM(256, return_sequences=True))
#model.add(keras.layers.Bidirectional(keras.layers.LSTM(50, return_sequences=True)))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(200, return_sequences=True)))
#model.add(keras.layers.Dropout(0.2))
#model.add(keras.layers.LSTM(50,go_backwards=True))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(dummy_labs.shape[1], activation=tensorflow.nn.sigmoid))
#model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(dummy_labs.shape[1], activation=tensorflow.nn.softmax))
#model.add(keras.layers.Dense(4, activation=tensorflow.nn.sigmoid))

model.compile(optimizer=tensorflow.train.AdamOptimizer(), 
              #loss=losses.mean_squared_error,
              #loss=losses.log_loss,
              loss = 'categorical_crossentropy',
              #loss = losses.softmax_cross_entropy ,
              metrics=['accuracy'])
#model.summary()

#training2 = training1[275000:350000]
#dummy_labs2 = dummy_labs[275000:350000]
#training2 = training1[200000:275000]
#dummy_labs2 = dummy_labs[200000:275000]
model.fit(training1, dummy_labs, epochs=10, verbose = 1, batch_size = 512, shuffle = False)

model.save('c://users/jliv/downloads/mod3.h5')







p1='pam cc who wants to come over and make art after'

p11 = np.array(one_hot(p1,vocablen))

import random
p11 = np.array(random.sample(range(1, vocablen), 22))

nw = p1.split()
nw = []
for i in range(12):
    nw.append(labelmap[np.argmax(model.predict(p11[0+i:11+i].reshape(1,11)))])



          
for i in range(500):
    predwords = nw[i+1:i+12]
    vw = np.array(one_hot(" ".join(predwords),vocablen))  
    #a = labelmap[np.argmax(model.predict(vw.reshape(1,5)))]
    a = labelmap[np.argmax(model.predict(vw.reshape(1,11)))]
    nw.append(a)  
    
    
    
predtext = " ".join(nw)  
predtext = predtext.replace(" pp", ". ")
predtext = predtext.replace(" ee ","! ")
predtext = predtext.replace(" qq","? ")
predtext = predtext.replace(" cc ",": ")
predtext = predtext.replace(" cmm ",", ")
predtext = predtext.replace(" bba "," [ ")
predtext = predtext.replace(" bbb "," ] ")

print(predtext[11:])









