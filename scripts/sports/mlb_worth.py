
import pandas as pd
from requests_html import AsyncHTMLSession
pd.set_option('display.max_columns',500)
import matplotlib.pyplot as plt
import numpy as np

salaries = pd.read_csv("2022_salaries.csv")
salaries['Name'] = salaries['Name'].str.lower().str.replace(
    "#",'',regex=False).str.replace(
    "*","",regex=False).str.replace(
    ".","",regex=False).str.replace(
    "'","",regex=False).str.replace(
    "","",regex=False).str.replace(
    " jr","",regex=False)

#async def get_async_tables():
session = AsyncHTMLSession()

r = await session.get('https://www.spotrac.com/mlb/rankings/')

await r.html.arender()

all_salaries = pd.read_html(r.html.html)[0]

r.html.text

#return all_salaries


all_salaries = all_salaries[['Player','salary']].copy()
drop = all_salaries['Player'].str[-4:].unique()
for d in drop:
    all_salaries['Player'] = all_salaries['Player'].str.replace(d,"").str.strip()
    
    
all_salaries['Player'] = all_salaries['Player'].str.lower().str.replace(
    ".","",regex=False).str.replace(
    "'","",regex=False).str.replace(
    " jr","",regex=False)


import unidecode

salaries['Name']=salaries['Name'].apply(lambda x:unidecode.unidecode(x) )

salaries = salaries.merge(all_salaries,
               left_on=['Name'],
               right_on = ['Player'],
               how='left')

salaries = salaries[~salaries['salary'].isna()].copy()

salaries['salary']=salaries['salary'].str.replace(
    ",","",regex=False).str.replace('$','',regex=False).astype(float)


salaries[['Name','Age','Tm','G','PA','WAR▼','oWAR','dWAR','salary','Pos Summary']].fillna(0)

productive = salaries[salaries['WAR▼']>0].copy()


productive = productive[
    ['Name','Age','Tm','G','PA','WAR▼','oWAR','dWAR','salary','Pos Summary']].fillna(0)

mean_sal=np.mean(all_salaries['salary'].str.replace(
    ",","",regex=False).str.replace('$','',regex=False).astype(float))

median_war = np.nanmedian(salaries['WAR▼'])
median_sal = np.nanmedian(salaries['salary'])



plt.scatter(productive['salary'],productive['WAR▼'])
plt.show()

productive['cost_per_war']=productive['salary']/productive['WAR▼']

median_cost_per = np.median(productive['cost_per_war'])

mask1 = productive['WAR▼']>2
mask2 = productive['cost_per_war']<=median_cost_per

plt.scatter(
    np.log(productive[(mask1)&(mask2)]['cost_per_war']),
    productive[(mask1)&(mask2)]['WAR▼'])
plt.show()

productive[(mask1)&(mask2)]