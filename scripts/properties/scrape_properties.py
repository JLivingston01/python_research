
import pandas as pd
import requests
from bs4 import BeautifulSoup

url = "https://www.trulia.com/for_sale/New_Rochelle,NY/1p_baths/0-20000000_price/1p_sqft/sqft;d_sort/10_zm/"


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36',
}
response = requests.get(url,
                        headers = headers)

soup = BeautifulSoup(response.text)

pagination = soup.find_all('li',{'data-testid':'pagination-page-link'})
pgs = []
for pg in pagination:
    pgs.append(pg.text)

last_pg = pgs[-1]
pgs_formatted = [f'{str(i)}_p/' for i in range(1,int(last_pg)+1)]

urls = []
all_properties = []
for p in pgs_formatted:
    urlf = url+p
    response = requests.get(urlf,
                            headers = headers)
        
    soup = BeautifulSoup(response.text)

    cards = soup.find_all('div',{'data-testid':'property-card-details'})
    
    properties = []
    
    for card in cards:
            
        pr = card.find_all('div',{'data-testid':'property-price'})[0].text
        sq = card.find_all('div',{'data-testid':'property-floorSpace'})[0].text
        addr = card.find_all('div',{'data-testid':'property-address'})[0].text
        bds = card.find_all('div',{'data-testid':'property-beds'})[0].text
        bths = card.find_all('div',{'data-testid':'property-baths'})[0].text
        
        properties.append(
            [pr,sq,addr,bds,bths]
            )
    
    all_properties=all_properties+properties

all_properties_df = pd.DataFrame(all_properties)

all_properties_df[0] = all_properties_df[0].str.replace(
    '$','',regex=False).str.replace(',','').astype(float)

all_properties_df[1] = all_properties_df[1].str.replace(
    'sqft','',regex=False
    ).str[:6].str.strip().str.replace(
        ',',""
        ).str.replace('(','',regex=False).astype(float)

all_properties_df['ppsf']=all_properties_df[0]/all_properties_df[1]

all_properties_df.sort_values(by='ppsf',ascending=True).head(50)



