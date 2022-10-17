

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import numpy as np
from os import listdir

#Multiple files of uber tweets I previously pulled from Twitter's API
imlist = listdir("C://Users/jliv/Downloads/tweets/tweettest/")

uberlist = [x for x in imlist if 'Ubertweets' in x]
uberdf = pd.read_csv('C://Users/jliv/Downloads/tweets/tweettest/'+uberlist[0])

for i in uberlist[1:]:
    temp = pd.read_csv('C://Users/jliv/Downloads/tweets/tweettest/'+i)
    uberdf = uberdf.append(temp)

uberdf.reset_index(inplace= True, drop = True)

#Tagging and alphaing docs for doc to vec
docs = list(uberdf.final_tweets)
docsx = []

for i in docs:
    x = i.lower()
    x = "".join([j for j in x if (j.isalpha())|(j==" ")])
    docsx.append(x)
    
    
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]

#Doc2Vec to embed the text semantic information
model = Doc2Vec(documents, vector_size=100, window=2, min_count=1, workers=4)
model.epochs = 2

for i in range(2000):
    print("Training epoch: "+str(i))
    model.train(documents,total_examples = model.corpus_count, epochs = model.epochs)

vector = []
for i in docsx:
    v = model.infer_vector(i)
    vector.append(v)
    
arr = np.array(vector)

#PCA on embeddings
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
pca.fit(arr) 


#Looking at embeddings, 
import matplotlib.pyplot as plt
pcacomps = pca.transform(arr)
plt.scatter(pcacomps[0],pcacomps[1])

pcacomps1 = pcacomps.transpose()

pca.components_.shape

from sklearn.cluster import KMeans

errs = []
for i in range(2,30):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(pcacomps)
    err = kmeans.inertia_
    errs.append(err)
    
    

m1 = pd.Series(errs).shift(1)-pd.Series(errs)
m2 = pd.Series(m1).shift(1)-pd.Series(m1)
    
    
plt.plot(errs)
plt.xticks(list(range(len(errs))),list(range(2,30)))
plt.show()
plt.plot(m1)
plt.xticks(list(range(len(errs))),list(range(2,30)))
plt.show()
plt.plot(m2)
plt.xticks(list(range(len(errs))),list(range(2,30)))
plt.show()

#Picking Best kmeans by moment 2
m1rm = m2.rolling(window=6,center=True,min_periods =6).mean()
m1rmsd = m1rm.rolling(window=6,center=True,min_periods =6).std()
choices = list(range(2,30))
df = pd.DataFrame(np.array([m1rmsd,choices]).transpose(),columns = ['sd','k'])

selection = df[df['sd']==np.nanmin(df['sd'])]['k'].values[0]
kmeans = KMeans(n_clusters =int(selection))
kmeans.fit(pcacomps)
    
labs = kmeans.labels_

#Looking at classified text by label
docsseries = pd.DataFrame()
docsseries['text']=docsx
docsseries['lab'] = labs

docsseries[docsseries['lab']==13]