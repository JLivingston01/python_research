


import pandas as pd
pd.set_option("display.max_columns", 500)
import datetime as dt

dat1 = pd.read_csv('horsing_dat.csv')
today = dt.datetime.strftime(dt.date.today(),'%Y-%m-%d')

fromdate = dt.datetime.strftime(
    dt.datetime.strptime(max(dat1['DATE']),'%Y-%m-%d')+dt.timedelta(1),
    '%Y-%m-%d')
dates = list(pd.date_range(fromdate,today).astype(str))

#date='2021-08-13'

out = pd.DataFrame()

for date in dates:
    query = f"/entries-results/{date}"
    base_url = f"https://entries.horseracingnation.com{query}"
    
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    
    tables = pd.read_html(base_url)
    
    t = tables[0]
    
    tracks = list(t['Track'].str.lower().str.replace(
        " ","-").str.replace("---","-").str.replace("'","-")
        )
    tracks = [
        ''.join([i for i in j if (i.isalnum())|(i=="-")]) for j in tracks]
    
    #track = tracks[0]
    
    print(date)
    for track in tracks:
        try:
            query = f"/entries-results/{track}/{date}"
            base_url = f"https://entries.horseracingnation.com{query}"
        
            tables = pd.read_html(base_url)
        
            runner_tables = [i for i in tables if ('PP' in i.columns)&('ML' in i.columns)]
            for i in range(len(runner_tables)):
                runner_tables[i]['RACENUM']=i+1
                
            runners = pd.concat(runner_tables)
            runners['DATE'] = date
            runners['TRACK']=track

            
            runners[['Runner','Sire']]=runners['Horse / Sire'].str.split("  ",expand=True)
            runners[['Trainer','Jockey']]=runners['Trainer / Jockey'].str.split("  ",expand=True)
            
            fs = [i for i in tables if ('Runner' in i.columns)&('Win' in i.columns)]
            
            if len(fs)>0:
                finishers = pd.concat(fs)
                finishers=finishers[~finishers['Runner'].str.contains("WINNER by")].copy()
                finishers['PLACE']=finishers.index+1
                
                runners = runners[['Runner','ML','DATE','Sire',
                                   'PP','Trainer','Jockey','TRACK','RACENUM']
                                  ].merge(
                    finishers[['Runner','PLACE']],
                    on=['Runner'],
                    how='left'
                    ).fillna(5)
                
                print(track)
        
                out = out.append(runners)
        except: 
            pass
        
dat1.append(out).to_csv("horsing_dat.csv",index=False)
    
    
    
    
    
    
    
    
    
