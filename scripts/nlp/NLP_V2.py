#TEXT MINING V2



import requests
import bs4
import pandas as pd
import numpy as np
#SCRAPE SOME TEXT
url = 'http://officequotes.net/no1-01.php'

result = requests.get(url)

result.text

soup = bs4.BeautifulSoup(result.text)
    
quotes = soup.findAll('div', {"class": "quote"})



txt = [i.get_text() for i in quotes]
txt = " ".join(txt)


txt="".join([i for i in txt if (i.isalnum())|(i==" ")|(i=="'")|(i==".")|(i=="\n")])

txt = txt.split("\n")

txt = [i.replace(".","").lower() for i in txt]

txt = [i for i in txt if (i != " ")&(~("deleted" in i))&(i!="")&(i!= "nbsp")]
txtsplit = [[i.split()[0],i.split()[1:]] for i in txt]


#WORD COUNT

words = []
for i in txtsplit:
    words = words+i[1]
    

cnts = dict.fromkeys(words, 0) 

for i in words:
    cnts[i]+=1
    
cntsdf=pd.DataFrame(cnts,index = [0]).T    

cntsdf.sort_values(ascending=False,by=0).head(45)

#FREQUENCY ENCODE
df = pd.DataFrame(columns = cnts.keys())


for i in txtsplit:
    cc = dict.fromkeys(i[1], 0) 
    for j in i[1]:
        cc[j]+=1
    df=df.append(pd.DataFrame(cc,index=[0]))

df.fillna(0,inplace=True)

#ONE HOT ENCODE

onehot = np.where(df>0,1,0)

onehot = pd.DataFrame(onehot,columns=df.columns.values)

#TF-IDF WEIGHTING

from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

svd = TruncatedSVD(n_components=10, random_state=42)

svd.fit(onehot)


word_embeddings=pd.DataFrame(svd.components_)

document_embeddings_usig = pd.DataFrame(svd.fit_transform(onehot))

plt.scatter(document_embeddings_usig[0],document_embeddings_usig[1])
plt.show()

document_embeddings_u = document_embeddings_usig@np.linalg.pinv(np.diag(svd.singular_values_))


plt.scatter(document_embeddings_u[0],document_embeddings_u[1])
plt.show()

svd2 = TruncatedSVD(n_components=10, random_state=42)
svd2.fit(onehot.T)


lsa4=pd.DataFrame(svd2.components_)


plt.scatter(lsa4.T[0],lsa4.T[1])
plt.show()


##  SPEAKER SEQUENCING

speakers = [i[0] for i in txtsplit if i[0] != 'documentary']

s1=pd.Series(speakers)

s2 = s1.shift(-1)


speakers=pd.DataFrame([s1,s2]).T
speakers['cnt']=1

ps = {}
for i in speakers[0].unique():
    p =( pd.pivot_table(speakers[speakers[0]==i],
                        index=[1],
                        values=['cnt'],
                        aggfunc='sum').rename(mapper={'cnt':i},axis='columns')/len(speakers[speakers[0]==i])).to_dict()

    ps[i]=p[i]

i0='michael'
count = 0
names = [i0]
while count<270:
    nn=list(ps[i0].keys())
    pp=[ps[i0][i] for i in nn]
    i0=np.random.choice(nn,size=1,replace=True,p=np.array(pp)/sum(pp))[0]
    names.append(i0)
    count+=1


speakerssim = pd.DataFrame({'names':names})
speakerssim['cnt']=1


sim=pd.pivot_table(speakerssim,index=['names'],values=['cnt'],aggfunc='sum')

actual=pd.pivot_table(speakers,index=[0],values=['cnt'],aggfunc='sum')

simvsactual=pd.merge(actual,sim,left_index=True,right_index=True,how='outer')

#Words
wordsim = [["newline",i[0]]+i[1] for i in txtsplit]

wordsim2=[]
for i in wordsim:
    wordsim2=wordsim2+i



w1=pd.Series(wordsim2)

w2 = w1.shift(-1)


wordss=pd.DataFrame([w1,w2]).T
wordss['cnt']=1
wordss['tot']=1

wordss=wordss[(~wordss[1].isna())&(wordss[1]!='truth')]

wordscount=pd.pivot_table(wordss,index=[0,1],values='cnt',aggfunc='sum').reset_index()
wordstot=pd.DataFrame(wordss.groupby([0]).apply(lambda x: len(x)))

wordstot['words']=wordstot.index

wordscount=pd.merge(wordscount,wordstot,left_on=[0],right_on=['words'],how='left')

wordscount['p']=wordscount['cnt']/wordscount['0_y']





i0='michael'
count = 0
wordseries = [i0]
while count<4000:
    table=wordscount[wordscount['0_x']==i0][[1,'p']]
    nn=list(table[1])
    pp=list(table['p'])
    i0=np.random.choice(nn,size=1,replace=True,p=np.array(pp))[0]
    wordseries.append(i0)
    count+=1
    
    
" ".join(wordseries[:500])

    
    
    