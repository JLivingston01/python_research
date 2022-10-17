


import datetime as dt
import oauth2
import pandas as pd
import json
import time

def req(query):
    consumer = oauth2.Consumer(key=''.encode('utf-8'), 
        secret=''.encode('utf-8'))
    token = oauth2.Token(key='', 
        secret='')
    client = oauth2.Client(consumer, token)
    resp, content = client.request( query, method="GET", body=bytes("", "utf-8"), headers=None )
    return content


for newstock in ['psychotherapy']:
    language = 'en'
    
    startdates = [dt.datetime.strftime(dt.datetime.now()+dt.timedelta(-7+i),"%Y-%m-%d") for i in range(8)]
    enddates = [dt.datetime.strftime(dt.datetime.now()+dt.timedelta(-6+i),"%Y-%m-%d") for i in range(8)]
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
        
        time.sleep(7)
        
        
        
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



text[0]