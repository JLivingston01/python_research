

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup


dates = list(pd.date_range('2021-01-29','2022-03-01').astype(str))

date = '2021-01-29'
for date in dates:
    query = f"/entries-results/{date}"
    base_url = f"https://entries.horseracingnation.com{query}"
    
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    
    response = requests.get(base_url,headers = headers)
    
    
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    
    tables = soup.find_all('table')
    
    table=tables[0]
    rows = table.find_all('tr')
    
    elements = table.find_all('td')
    
    time = [elements[i].time.get("datetime") 
            for i in range(len(elements)) if i%7==0]
    
    link = [elements[i].a.get("href") 
            for i in range(len(elements)) if i%7==1]
    
    track = [elements[i].a.text.replace('\n','').strip()
            for i in range(len(elements)) if i%7==1]
    
    
    extracted_info = pd.DataFrame({
        'track':track,
        'time':time,
        'link':link
        })
    
    extracted_info['time']=pd.to_datetime(extracted_info['time']).dt.time
            
    all_races = pd.DataFrame()
    print(date)
    for query,trackname in zip(extracted_info['link'],extracted_info['track']):
        #query = extracted_info['link'].values[0]

        base_url = f"https://entries.horseracingnation.com{query}"
        
        track_response = requests.get(base_url,headers = headers)
        
        track_soup = BeautifulSoup(track_response.text, 'html.parser')
        
        
        entries_tables = track_soup.find_all('table',class_='table-entries')
            
        
        race_num = []
        for i in range(len(entries_tables)):

            horses_in_race = len(entries_tables[i].find_all('td',{'data-label':"Horse / Sire"}))
            
            ids = list(np.repeat(i,horses_in_race))
            
            race_num = race_num+ids
        
        
        horse_names_all = track_soup.find_all('td',{'data-label':"Horse / Sire"})
        mlodds_all = track_soup.find_all('td',{'data-label':"Morning Line Odds"})
        horses = [i.h4.text for i in horse_names_all]
        mlodds = [i.p.text for i in mlodds_all]
        
            
            
        races = pd.DataFrame({
            'name':horses,
            'odds':mlodds,
            'racenum':race_num
            })
        
        
        payouts_tables = track_soup.find_all('table',class_='table-payouts')
        
        results_df = pd.DataFrame()
        for pt in payouts_tables:
            #pt = payouts_tables[0]
            
            elements=pt.find_all('td')
            results = [i.text.replace('\n','').strip() for i in elements]
            names = [results[i] for i in range(len(results)) if i%5==0]
            tmp = pd.DataFrame()
            tmp['name']=names
            tmp['place']=range(1,len(tmp)+1)
            
            results_df=results_df.append(tmp)
        
        print(trackname,len(results_df))

        try:
            races=races.merge(results_df,how='left',on=['name'])
                
            races[['odds_num','odds_denom']]=races['odds'].str.split(
                "/",expand=True)#.astype(int)
            
            races['odds_num']=np.where(races['odds_num']=='',100,races['odds_num'])
            races['odds_num']=races['odds_num'].astype(int)
            races['odds_denom'].fillna(0,inplace=True)
            races['odds_denom']=races['odds_denom'].astype(int)
            
            races['prob_w'] = races['odds_denom']/(races['odds_num']+races['odds_denom'])
            races['track']=trackname
            
            all_races=all_races.append(races)

        except:
            print("no results found, pass")
    
    
    all_races.to_csv(f"data/results_{date}.csv",index=False)
