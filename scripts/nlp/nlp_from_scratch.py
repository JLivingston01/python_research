# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:50:46 2020

@author: jliv
"""

import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt
pd.set_option("display.max_columns",500)

s1 = "I love cats"
s2="I love hats"
s3="I do not like hats or cats"

s1 = s1.lower().split()
s2 = s2.lower().split()
s3 = s3.lower().split()

dl = [s1,s2,s3]
data = {0:s1,
        1:s2,
        2:s3}

corpus = list(set(s1+s2+s3))

df00 = np.zeros((len(corpus),len(dl)))
df=pd.DataFrame(df00,columns = list(range(len(data.keys()))),index=corpus)

dfcounts = pd.DataFrame(df00,columns = list(range(len(data.keys()))),index=corpus).copy()


for i in df.columns.values:
    df[i]=np.where(df.index.isin(data[i]),1,0)


dft = df.T
# LSA

svd=np.linalg.svd(dft)[0]

plt.scatter(svd.T[1],svd.T[0])

svdt = TruncatedSVD(n_components=2)
vectors = svdt.fit_transform(dft)



# TFIDF
words = np.array([len(i) for i in dl])
tf = dft/words.reshape(len(dl),1)

docs_w_word = dft.T
docs_w_word = np.count_nonzero(docs_w_word,axis=1)
idf = pd.Series(np.log(len(dl)/docs_w_word),index=corpus)

tfidf = pd.DataFrame()
for i in data.keys():
    tfidf[i] = tf.T[i]*idf






