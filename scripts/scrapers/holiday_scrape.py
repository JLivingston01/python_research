# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:35:10 2020

@author: jliv
"""


import requests     
                      
from lxml import html

import pandas as pd


country = 'us'
countries = [
        'us',
        'uk',
        'germany',
        'france',
        'canada',
        'japan',
        'spain',
        'brazil',
        'denmark',
        'belgium',
        'mexico',
        'poland',
        'south-korea',
        'italy',
        'turkey',
        'netherlands',
        'sweden',
        'finland',
        'singapore',
        'austria',
        'greece',
        'new-zealand'
        ]
year = '2020'

fulldat = pd.DataFrame()

for country in countries:
    url = "https://www.timeanddate.com/holidays/"+country+"/"+year
    
    
    res = requests.get(url,headers = {'user-agent':'jays MAC'})
    
    tree = html.fromstring(res.content)
    
    # =============================================================================
    # 
    # 
    # =============================================================================
    ##
    tr_elements = tree.xpath('//table[contains(@id,"holidays-table")]/tbody/tr[contains(@id,"tr")]')
    
    #Create empty list
    col=[]
    i=0
    #For each row, store each first element (header) and an empty list
    for a in tr_elements:
        temp = []
        for t in a:
            i+=1
            name=t.text_content()
            temp.append(name)
        col.append(temp)
    
    if len(col[0]) == 5:
        dat = pd.DataFrame(col,columns = ['date','day_of_week','holiday','type','details'])
    elif len(col[0])==4:
        dat = pd.DataFrame(col,columns = ['date','day_of_week','holiday','type'])
        dat['details']=""
    dat['country']=country
    dat['year']=year
    
    fulldat = fulldat.append(dat)
    
    
    
    