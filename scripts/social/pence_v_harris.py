


import datetime as dt
import oauth2
import pandas as pd
import json
import time

#import pymysql

def pull_tweets(symbols):
    def req(query):
        consumer = oauth2.Consumer(key=''.encode('utf-8'), secret=''.encode('utf-8'))
        token = oauth2.Token(key='', secret='')
        client = oauth2.Client(consumer, token)
        resp, content = client.request( query, method="GET", body=bytes("", "utf-8"), headers=None )
        return content
    
    tdf2 = pd.DataFrame()
    for newstock in symbols[:]:
        language = 'en'
        
        startdates = [dt.datetime.strftime(dt.datetime.now()+dt.timedelta(-1+i),"%Y-%m-%d") for i in range(8)]
        enddates = [dt.datetime.strftime(dt.datetime.now()+dt.timedelta(-0+i),"%Y-%m-%d") for i in range(8)]
        #exclude = ['-Congrats','-Stand','-Laura','-Why']
        exclude = ['']
        #How = mixed, recent or popular
        how = 'mixed'
        
        searchterm = newstock
        
        
        exclude = "%20".join(exclude)
        
        
        
    
        times = []
        text = []
        retweet_cnt = []
        fvrt_cnt = []
        user = []
        user_flwrs=[]
        user_statuses = []
        timezone = []
        
        
        #home_timeline = req("https://api.twitter.com/1.1/application/rate_limit_status.json?resources=help,users,search,statuses")
        #home_timeline = home_timeline.decode("utf-8") 
        #home_timeline = json.loads(home_timeline)
        for startdate,enddate in zip(startdates[:],enddates[:]):
            raw_query="lang={}&q={}%20-RT%20{}%20since%3A{}%20until%3A{}&result_type={}&count=1000&tweet_mode=extended&-filter=retweets".format(language,searchterm,exclude,startdate,enddate,how)
            query = 'https://api.twitter.com/1.1/search/tweets.json?'+raw_query
            home_timeline = req(query)
            home_timeline = home_timeline.decode("utf-8") 
            home_timeline = json.loads(home_timeline)
            statuses = home_timeline['statuses']
            print(startdate,newstock,len(statuses))
            
            #time.sleep(7)
            
            
            
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
             
        tdf = pd.DataFrame({'time':times,'text':text,'retweets':retweet_cnt,'favorites':fvrt_cnt,
                     'user':user,'followers':user_flwrs,'user_statuses':user_statuses,'timezone':timezone})        
        tdf['ticker']=newstock
        
        tdf.text = tdf['text'].apply(lambda x: "".join([i for i in x if (i.isalpha())|(i==" ")] ))
        tdf['text']=tdf['text'].apply(lambda x: " ".join([i for i in x.split() if "http" not in i]))
        
        tdf['date'] =tdf.time.apply(lambda x: dt.datetime.strftime(dt.datetime.strptime(x,"%a %b %d %H:%M:%S %z %Y"),"%Y-%m-%d"))
        
        tdf2 = tdf2.append(tdf)
    return tdf2
    

pence = pull_tweets(['mike pence'])

harris = pull_tweets(['kamala harris'])


pd.set_option('display.max_columns',25)


pence = pence[pence['date']=='2020-10-08'].copy()
harris= harris[harris['date']=='2020-10-08'].copy()


from nltk.sentiment.vader import SentimentIntensityAnalyzer


sid = SentimentIntensityAnalyzer()


pense_t = list(pence['text'])
harris_t = list(harris['text'])

pense_neg = []
pense_neu = []
pense_pos = []
pense_comp = []


harris_neg = []
harris_neu = []
harris_pos = []
harris_comp = []

for i in pense_t:

    ss = sid.polarity_scores(i)
    neg,neu,pos,comp = ss['neg'],ss['neu'],ss['pos'],ss['compound']
    
    pense_neg.append(neg)
    pense_neu.append(neu)
    pense_pos.append(pos)
    pense_comp.append(comp)
    

for i in harris_t:

    ss = sid.polarity_scores(i)
    neg,neu,pos,comp = ss['neg'],ss['neu'],ss['pos'],ss['compound']
    
    harris_neg.append(neg)
    harris_neu.append(neu)
    harris_pos.append(pos)
    harris_comp.append(comp)



import numpy as np

np.mean(pense_neg)

np.mean(harris_neg)

import matplotlib.pyplot as plt

bins = np.linspace(-1,1,30)
bins2 = np.linspace(0,1,20)

plt.hist(harris_pos,label='harris',alpha=.5,bins=bins2)
plt.hist(pense_pos,label='pence',alpha=.5,bins=bins2)
plt.title("positive")
plt.legend()
plt.show()

plt.hist(harris_neu,label='harris',alpha=.5,bins=bins2)
plt.hist(pense_neu,label='pence',alpha=.5,bins=bins2)
plt.title("neutral")
plt.legend()
plt.show()


plt.hist(harris_neg,label='harris',alpha=.5,bins=bins2)
plt.hist(pense_neg,label='pence',alpha=.5,bins=bins2)
plt.title("negative")
plt.legend()
plt.show()


plt.hist(harris_comp,label='harris',alpha=.5,bins=bins)
plt.hist(pense_comp,label='pence',alpha=.5,bins=bins)
plt.title("compound")
plt.legend()
plt.show()




