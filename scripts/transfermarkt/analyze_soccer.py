
import pandas as pd
import sqlite3
import numpy as np

pd.set_option('display.max_columns',500)


def makeSeasonYr(df):
    df['SEASON'] = df['SEASON'].str.replace(".0","",regex=False)
    df['SEASON_YR'] = df['SEASON'].str[-2:] 
    
    return df

def fill_minutes(s):
    s = s.str.replace("'","",regex=False).str.replace(".","",regex=False)
    s = pd.Series(np.where(s == "-","0",s),index=s.index)
    s = s.astype(float)
    
    return s

def fill_value(s):
    s = s.str.replace("$","",regex=False)
    mult = pd.Series(np.where(s.str.contains("m"),1000000,1000),index=s.index)
    s = pd.Series(np.where(s == "-",None,s),index=s.index)
    s = s.str.replace("m","",regex=False).str.replace("Th.","",regex=False)
    s = s.astype(float)
    s=s*mult
    return s
    
def fill_0(s):
    
    s = pd.Series(np.where(s == "-","0",s),index=s.index)
    s = s.astype(float)
    return s

def fill_nan(s):
    
    s = pd.Series(np.where(s == "-",None,s),index=s.index)
    s = s.astype(float)
    return s

conn = sqlite3.connect("soccer.sqlite")

allRegStats = pd.read_sql_query("SELECT * from regulars", conn)
allGKStats = pd.read_sql_query("SELECT * from goalkeepers", conn)

conn.close()

allRegStats = makeSeasonYr(allRegStats)
allGKStats = makeSeasonYr(allGKStats)

metrics = ['APPS','G','OG','SUBON','SUBOFF','YELLOW','SOFTRED','RED']
for metric in metrics:
    allRegStats[metric] = fill_0(allRegStats[metric])
    allGKStats[metric] = fill_0(allGKStats[metric])

allRegStats['A'] = fill_0(allRegStats['A'])
allRegStats['PENGOALS'] = fill_0(allRegStats['PENGOALS'])
allGKStats['GOALSAGAINST'] = fill_0(allGKStats['GOALSAGAINST'])
allGKStats['CLEANSHEETS'] = fill_0(allGKStats['CLEANSHEETS'])

metricsB = ['PPG']
for metric in metricsB:
    allRegStats[metric] = fill_nan(allRegStats[metric])
    allGKStats[metric] = fill_nan(allGKStats[metric])
    
metricsC = ['CURR_VALUE']
for metric in metricsC:
    allRegStats[metric] = fill_value(allRegStats[metric])
    allGKStats[metric] = fill_value(allGKStats[metric])


allGKStats['MINUTES']=fill_minutes(allGKStats['MINUTES'])
allRegStats['MINS_SEAS']=fill_minutes(allRegStats['MINS_SEAS'])
allRegStats['MINS']=fill_minutes(allRegStats['MINS'])

allGKStats.rename(mapper={'MINUTES':'MINS'},axis=1,inplace=True)


values = allGKStats.groupby(['ID']).agg(
    {'CURR_VALUE':np.nanmax}).reset_index()


allGKStats['90MINS'] = allGKStats['MINS']/90
allGKStats['POINTS'] = np.round(allGKStats['PPG'].fillna(0)*allGKStats['APPS'],0)

GKYrSum = allGKStats.groupby(['ID','SEASON_YR']).agg({
    'APPS':'sum',
    'POINTS':'sum',
    '90MINS':'sum',
    'GOALSAGAINST':'sum',
    'CLEANSHEETS':'sum',
    'YELLOW':'sum',
    'RED':'sum',
    'CONTESTS':'sum',
    'SUBON':'sum',
    'SUBOFF':'sum',
    'CURR_VALUE':'max'
    }).reset_index()

GKYrSum['APPS_PERC'] = GKYrSum['APPS']/GKYrSum['CONTESTS']

GKSum = GKYrSum.groupby('ID').agg(
    {
    'APPS':'sum',
    'POINTS':'sum',
    '90MINS':'sum',
    'GOALSAGAINST':'sum',
    'CLEANSHEETS':'sum',
    'YELLOW':'sum',
    'RED':'sum',
    'CONTESTS':'sum',
    'SUBON':'sum',
    'SUBOFF':'sum',
    'CURR_VALUE':'max'
     }
    ).reset_index()


GKSum['APPS_PERC'] = GKSum['APPS']/GKSum['CONTESTS']
GKSum['GA/90'] = GKSum['GOALSAGAINST']/GKSum['90MINS']
GKSum['POINTS/APPS'] = GKSum['POINTS']/GKSum['APPS']
GKSum['CS/APPS'] = GKSum['CLEANSHEETS']/GKSum['APPS']

GKSum.corr()

import matplotlib.pyplot as plt

plt.scatter(GKSum['GA/90'],GKSum['APPS_PERC'])


allGKStats[allGKStats['ID']=='jori429799']














