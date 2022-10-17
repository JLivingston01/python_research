
import pandas as pd
import numpy as np


date = '2022-07-21'
outcomes = pd.read_csv(f"horse_predictions_{date}.csv")


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
out = pd.DataFrame()
for track in tracks:
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

outcomes = outcomes.merge(out[['Runner','PLACE']],
               on=['Runner'],
               how='left')

outcomes['CORRECT']=np.where(outcomes['PLACE']==1,1,0)

np.mean(outcomes[~outcomes['PLACE'].isna()]['CORRECT'])

outcomes['payout']=np.where(outcomes['CORRECT']==0,-1,
         outcomes['odds'])

sum(outcomes[~outcomes['PLACE'].isna()]['payout'])


