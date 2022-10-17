

import pandas as pd
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 150)
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36',
}
response = requests.get("https://www.fastfoodprice.com/menu/wendys-prices/",
                        headers = headers)

soup = BeautifulSoup(response.text, 'html.parser')

lis = soup.find_all('li',{"class": "menu-item-type-post_type"})

links = []
for li in lis:
    links.append(li.a['href'])

links = list(set(links))

all_docs= {}
for link in links:
    print(link)
    response = requests.get(link,
                            headers = headers)
    
    all_docs[link]=response.text
    
    
all_cals = pd.DataFrame()
for link in all_docs.keys():
    pr = pd.read_html(all_docs[link])
    
    relevant_tbls = pr[1:-1]
    
    tbl = pd.concat(relevant_tbls)
    
    tbl = tbl[~tbl[2].isna()].copy()
    
    cals = tbl[(tbl[2].str.contains('Cal'))&
               (~tbl[2].str.contains('-'))&
               (~tbl[2].str.contains('Catering'))].copy()
    
    cals[2] = cals[2].str.replace('Cal','')
    cals=cals[cals[2].str.len()<=5].copy()
    cals[2]=cals[2].astype(float)
    
    cals[3] = cals[3].str.replace('$','',
                              regex=False).astype(float)
    cals = cals[~cals[3].isna()].copy()
    
    cals['source'] = link.replace(
        'https://www.fastfoodprice.com/menu/',''
        ).replace('/','').replace('-',' ').replace(
            ' prices',''
            )
    
    all_cals=all_cals.append(cals)
    
    
all_cals['int']=1

all_cals.to_csv("c:/users/jliv/calories_and_prices.csv",index=False)

import numpy as np

def regression(x):
    cols = [2,'int']
    return(np.linalg.pinv(x[cols].T@x[cols])@(x[cols].T@x[3]))

import time

st = time.time()
all_cals.groupby(['source']).apply(regression)
print(time.time()-st)

coefs = all_cals.groupby(['source']).apply(regression).reset_index()

coefs[['c0','c1']] = pd.DataFrame(coefs[0].to_list())


