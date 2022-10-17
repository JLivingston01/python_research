
import pandas as pd
import requests
from bs4 import BeautifulSoup

import sqlite3

pd.set_option('display.max_columns',500)
### CONSTANTS ###
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36',
}



def getLeagueHrefs(leagueUrl) -> list:
    leagueResponse = requests.get(leagueUrl,
                            headers = headers)
    
    leagueSoup = BeautifulSoup(leagueResponse.text,features="lxml")
    leagueTables = leagueSoup.find_all("table")
    leagueHrefs = list(set([a['href'] for a in leagueTables[3].find_all('a') if 'kader' in a['href']]))
    
    return leagueHrefs

def getTeamUrlValues(leagueHref) -> (list,list):
    url = f"""
    https://www.transfermarkt.us{leagueHref}
    """
    response = requests.get(url,
                            headers = headers)
    
    soup = BeautifulSoup(response.text,features="lxml")
    
    tables = soup.find_all("table")
    roster = tables[1].find_all("td",{'class':'hauptlink'})
    
    positions = pd.read_html(response.text)[1]
    positions = list(positions['player'])
    positions = [positions[i] for i in range(len(positions)) if (i+1)%3==0]
    
    values = [i.text for i in tables[1].find_all("td",{'class':'rechts hauptlink'})]
    
    roster = [roster[i] for i in range(len(roster)) if i%2==0]
    
    return roster,values,positions

def getPlayerStats(r,v,p) -> (pd.DataFrame,str):
    playerHref = r.a['href'].replace("profil","leistungsdatendetails")#r = roster[8]
    #playerHref = '/mikey-ambrose/leistungsdatendetails/spieler/255918'
    Url = f"https://www.transfermarkt.us{playerHref}/saison//verein/0/liga/0/wettbewerb//pos/0/trainer_id/0/plus/1"
    
    playerResponse = requests.get(Url,
                            headers = headers)
    
    position = p
    
    playerSoup = BeautifulSoup(playerResponse.text,features="lxml")
    table = playerSoup.find_all('table')[1]
    imgs = table.find_all('td',{'class':'hauptlink no-border-rechts zentriert'})
    clubs = [i.find("img")['alt'] for i in imgs ]
    
    playerStats = pd.read_html(playerResponse.text)[1]
    
    gkCols = ['SEASON','drop_1','COMPETITION','drop_2','CONTESTS','APPS','PPG',
              'G','OG','SUBON','SUBOFF','YELLOW','SOFTRED','RED','GOALSAGAINST',
              'CLEANSHEETS','MINUTES','drop_3']
    regCols = ['SEASON','drop_1','COMPETITION','drop_2','CONTESTS','APPS','PPG',
               'G','A','OG','SUBON','SUBOFF','YELLOW','SOFTRED','RED','PENGOALS',
               'MINS_SEAS','MINS','drop_3']
    
    if position == 'Goalkeeper':
        cols = gkCols
    else:
        cols = regCols
        
    playerStats.columns = cols
    playerStats.drop([i for i in cols if 'drop_' in i],axis=1,inplace=True)
    playerStats=playerStats[~playerStats['SEASON'].isna()].copy()
    
    player = playerHref.split("/")[1]+playerHref.split("/")[-1]
    
    playerStats['ID'] = player
    playerStats['HREF'] = playerHref
    playerStats['POS']=position
    playerStats['CURR_VALUE'] = v
    playerStats['CLUB'] = clubs
    return playerStats,position


def getLeagueLinks(filepath='league'):
    with open('leagues') as f:
        leagueLines = list(f)
    
    leagueUrls = [i.replace("\n","") for i in leagueLines]
    return leagueUrls

allGKStats = pd.DataFrame()
allRegStats = pd.DataFrame()

leagueUrls=getLeagueLinks(filepath='league')

#leagueUrl = "https://www.transfermarkt.us/major-league-soccer/startseite/wettbewerb/MLS1"

for leagueUrl in leagueUrls:
    print(leagueUrl)
    leagueHrefs = getLeagueHrefs(leagueUrl) 
    
    ### ONE CLUB URL, LOOK UP PLAYERS
    
    #leagueHref = '/deportivo-la-guaira/kader/verein/26468/saison_id/2021'
    for leagueHref in leagueHrefs:
        
        try:
            print(leagueHref)
            roster,values,positions = getTeamUrlValues(leagueHref)
            
            ###ALL PLAYER STATS BY LOOPING ROSTER
            
            for r,v,p in zip(roster,values,positions):
                ### ONE PLAYER'S STATS
                try:
                    playerStats,position = getPlayerStats(r,v,p)
                
                    if position == 'Goalkeeper':
                        allGKStats=allGKStats.append(playerStats)
                    else:
                        allRegStats=allRegStats.append(playerStats)
                except:
                    pass
        except:
            pass
            


conn = sqlite3.connect("soccer.sqlite")

allRegStats.reset_index(
    drop=True).to_sql("regulars", conn, if_exists="replace",
                      index=False)
allGKStats.reset_index(
    drop=True).to_sql("goalkeepers", conn, if_exists="replace",
                      index=False)

conn.close()

allRegStats.to_csv('reg.csv',index=False)
allGKStats.to_csv('gk.csv',index=False)






