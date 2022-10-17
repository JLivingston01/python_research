



import yfinance as yf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

import datetime as dt


indeces = ['^VIX','^DJI','^GSPC','^TNX']
    

positions=['AAPL',
    'AMD',
    'AMP',
    'BAGAX',
    'BFOCX',
    'CDW',
    'CLDR',
    'CRM',
    'DIA',
    'ENPH',
    'FIVN',
    'ICLR',
    'INTC',
    'JKE',
    'MA',
    'MET',
    'MSFT',
    'MU',
    'NOW',
    'OGIG',
    'OMF',
    'PRCOX',
    'SPXL',
    'SPY',
    'SPYG',
    'TRBCX',
    'TRNO',
    'USRT',
    'V',
    'WAMCX',
    'XLF',
    'XSD']

watching = ['NVDA',
            'SQ','PYPL','FIS','FISV',
            'AVID','HPE','WDC','DDD',
            'ORCL',
            'TROW','STT','APO','ARES',
            'PLUS','NSIT','SNX','ARW',
            'EGP','STAG','MNR',
            'GL','PRI','PRU','BHF',
            'AVGO','QCOM','ADI',
            'DFS','SC','CACC','GDOT',
            'TXN','ADI','MCHP',
            'AVTR','CRL','PRAH','SYNH',
            'ADBE','ADSK','WDAY','CDNS',
            'ZM','RNG','APPF','LPSN',
            'VMW','NLOK',
            'TSLA', 'GM','F',
            'GOOGL','FB','PINS','TWTR','MTCH',
            'AMZN','CHWY','EBAY','W','ETSY'
            ]

symbols = list(set(indeces+positions+watching))


tickers = yf.Tickers(' '.join(symbols))




#Recomendations
import json

info = {}
for i in tickers.tickers:
    info_dct = i.info
    info[info_dct['symbol']]=info_dct
    

with open('financial_data/info.json', 'w') as fp:
    json.dump(info, fp)


recos_reports=pd.DataFrame()
for i in tickers.tickers:
    recs = i.recommendations
    if recs is not None:
        recos = i.recommendations.reset_index()
        recos_g = recos.groupby(['Firm']).agg({'Date':'max'}).reset_index()
        recos_report=recos.merge(recos_g,on=['Firm','Date'],how='inner')
        recos_report['symbol'] = i.info['symbol']
        recos_reports=recos_reports.append(recos_report)
    

#!mkdir -p financial_data
recos_reports.to_csv("financial_data/recos_reports.csv",index=False)

mo_6 = dt.datetime.strftime(dt.datetime.today()-dt.timedelta(180),"%Y-%m-%d")

recent_recos = recos_reports[recos_reports['Date']>=mo_6].copy()

grade_summary=recent_recos.groupby(['symbol','To Grade']).agg({'Date':'count'}).reset_index()

grade_summary=grade_summary.merge(grade_summary.groupby(['symbol']).agg(
        {'Date':'sum'}).reset_index().rename({'Date':'total_ratings'},axis='columns'),on=['symbol'],how='left')

grade_summary['pct_ratings']=grade_summary['Date']/grade_summary['total_ratings']

grade_summary.columns=['symbol','To Grade','ratings','total_ratings','pct_ratings']

plt.pie(grade_summary[grade_summary['symbol']=='AAPL']['ratings'],
        labels = grade_summary[grade_summary['symbol']=='AAPL']['To Grade'])
plt.show()




#Value by assets - liabilities

value = pd.DataFrame()
for i in tickers.tickers:
    balance_sheet = i.quarterly_balance_sheet.T
    
    if len(balance_sheet.columns)>0:
        balance_sheet=balance_sheet[['Total Liab','Total Assets']]
        
        balance_sheet['value'] = balance_sheet['Total Assets']-balance_sheet['Total Liab']
        
        balance_sheet['symbol'] = i.info['symbol']
        
        value=value.append(balance_sheet)

value = value.reset_index()
value.columns = ['Date','Liab','Assets','Value','symbol']


value.to_csv("financial_data/value_reports.csv",index=False)

Latest_value = value.merge(value.groupby(['symbol']).agg({'Date':'max'}).reset_index(),on=['Date','symbol'],how='inner')


#Market Cap

market_cap = pd.DataFrame()
for i in tickers.tickers:
    cap = pd.DataFrame({'cap':[i.info['marketCap']]})
    cap['symbol'] = i.info['symbol']
    market_cap=market_cap.append(cap)
    
market_cap = market_cap[~market_cap['cap'].isna()].reset_index(drop=True)



valuation=Latest_value.merge(market_cap)
valuation['cap']=valuation['cap'].astype(float)

valuation['over_under_valuation'] =valuation['cap']/valuation['Value']-1 

valuation.to_csv("financial_data/valuation_and_market_cap.csv",index=False)


#History


history = pd.DataFrame()
for i in tickers.tickers:
    hist = i.history(period="max")
    hist['symbol'] = i.info['symbol']
    history=history.append(hist)

history.reset_index(inplace=True)
history.sort_values(by=['symbol','Date'],ascending=[True,True],inplace=True)

history.to_csv("financial_data/history.csv",index=False)



X=history[['symbol','Date','Close']].copy()
X['int']=1
X['daynum'] = X.groupby(['symbol'])['Date'].cumcount()+1
X['rolling'] = X.groupby(['symbol'])['Close'].rolling(window=30,min_periods=1).mean().reset_index().set_index('level_1')['Close']
X['momentum'] = X['Close']/X['rolling']-1

X[].groupby(['symbol'])['momentum'].rolling(window=30,min_periods=1)

X[['int','daynum']]


