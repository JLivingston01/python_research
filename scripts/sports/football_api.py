

import requests
import pandas as pd

import time
import json

url = "https://elenasport-io1.p.rapidapi.com/v2/leagues"

querystring = {"name":"Premier","page":"1"}

headers = {
    'x-rapidapi-host': "elenasport-io1.p.rapidapi.com",
    'x-rapidapi-key': "c24ab38b1dmsh53b3e8f9d414ce7p1532cdjsnc967532986c4"
    }

response = requests.request("GET", url, headers=headers, params=querystring)

dct = json.loads(response.text)

epl = [i for i in dct['data'] if i['countryName']=='England']
epl[0]['id']


url = "https://elenasport-io1.p.rapidapi.com/v2/leagues/"+str(epl[0]['id'])+"/seasons"

headers = {
    'x-rapidapi-key': "c24ab38b1dmsh53b3e8f9d414ce7p1532cdjsnc967532986c4",
    'x-rapidapi-host': "elenasport-io1.p.rapidapi.com"
    }

response = requests.request("GET", url, headers=headers)
dct = json.loads(response.text)


seasons = [(i['id'],i['start']) for i in dct['data']]


i=seasons[0][0]

len(seasons)
#for  i in seasons[1:4]:
#for  i in seasons[4:7]:
#for  i in seasons[7:10]:
#for  i in seasons[10:13]:
#for  i in seasons[13:16]:
#for  i in seasons[16:19]:
#for  i in seasons[19:]:
for  i in seasons:
    page=1
    
    dat = pd.DataFrame()
    while True:
        url = "https://elenasport-io1.p.rapidapi.com/v2/seasons/"+str(i[0])+"/fixtures"
        
        querystring = {"page":str(page)}
        
        headers = {
            'x-rapidapi-key': "c24ab38b1dmsh53b3e8f9d414ce7p1532cdjsnc967532986c4",
            'x-rapidapi-host': "elenasport-io1.p.rapidapi.com"
            }
        
        response = requests.request("GET", url, headers=headers, params=querystring)
        
        dct = json.loads(response.text)
        
        dattemp = pd.DataFrame(dct['data'])
        dat=dat.append(dattemp)
        if dct['pagination']['hasNextPage']==True:
            page+=1
        else:
            break
        time.sleep(7)
    
    file = dat['seasonName'].unique()[0]

    dat[['homeName','awayName','seasonName','date','round','team_home_90min_goals','team_away_90min_goals']].sort_values(
            by='round').to_csv(file.replace("/","_").replace(" ","_").replace("-","_")+"_b.csv",index=False)

