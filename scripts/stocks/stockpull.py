
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
symbols = ['^VIX','DIA','SPY','SPYG','SPXL','BFOCX','XLF','XSD','JKE','SCHG']
    
"""   
key = ""

    

def pull_prices(symbols,key,delay=True):
    dat = pd.DataFrame()
    for symbol in symbols:
            
        URL = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol="+symbol+"&outputsize=full&apikey="+key+"&datatype=csv"
        csv=pd.read_csv(URL)
        csv['ticker'] = symbol
        csv = csv.reindex(sorted(csv.columns), axis=1)
        print(symbol)
        
        
        csv=csv.rename(mapper={'close':'close_',
                               'open':'open_',
                               'timestamp':'timestamp_'},axis='columns')
        
        csv.drop_duplicates(inplace=True)
        
        if delay==True:
            time.sleep(15)
        
        dat=dat.append(csv)
    
    return dat.reset_index(drop=True)
        

assets = pull_prices(symbols,key)


pd.set_option("display.max_columns",500)

assets

assets[assets['ticker']=='VIX'].values
"""

import yfinance as yf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

symbols = ['^VIX','^DJI','^GSPC','DIA','SPY','SPYG','SPXL','BFOCX','XLF','XSD','JKE','SCHG']
    

dat = yf.download(tickers = " ".join(symbols),
                  period = "max",
                  interval = "1d",
                  group_by = 'ticker')

dat.columns

assets = pd.DataFrame()
for s in symbols:
    
    dat = yf.Ticker(s)
    
    hist = dat.history(period="max")
    hist['symbol']=s
    assets = assets.append(hist)


assets.reset_index(inplace=True,drop=False)

plt.plot(np.log(assets[assets['symbol']=='^VIX'].set_index('Date')['Close']))
plt.plot(np.log(assets[assets['symbol']=='^GSPC'].set_index('Date')['Close']))
plt.plot(np.log(assets[assets['symbol']=='^DJI'].set_index('Date')['Close']))
plt.show()


cross= pd.crosstab(assets['Date'],assets['symbol'],values = assets['Close'],aggfunc='sum')

log_cross = np.log(cross)

log_cross.corr()

log_interday_change = log_cross-log_cross.shift(1)


log_interday_change.corr()

plt.plot(log_interday_change[log_interday_change.index>='2020-01-01']['^VIX'])
plt.plot(log_interday_change[log_interday_change.index>='2020-01-01']['^GSPC'])
plt.xticks(rotation=90)
plt.show()





vix_and_log_interday = log_cross-log_cross.shift(1)
vix_and_log_interday['^VIX'] = cross['^VIX']


vix_and_log_interday.corr()


Xs=cross[['^VIX']].copy()

for  i in range(1,5):
    Xs['VIX_-'+str(i)] = Xs['^VIX'].shift(i)
    
    
Xs['logVIX']=np.log(Xs['^VIX'])

Xs['changeVIX1'] = Xs['^VIX']-Xs['^VIX'].shift(1)
Xs['changeVIX4'] = Xs['^VIX']-Xs['^VIX'].shift(4)
Xs['changeVIX8'] = Xs['^VIX']-Xs['^VIX'].shift(8)
Xs['log_changeVIX1'] = np.log(Xs['^VIX'])-np.log(Xs['^VIX'].shift(1))
Xs['log_changeVIX4'] = np.log(Xs['^VIX'])-np.log(Xs['^VIX'].shift(4))
Xs['log_changeVIX8'] = np.log(Xs['^VIX'])-np.log(Xs['^VIX'].shift(8))

Xs[[i+'shift1' for i in Xs.columns if 'changeVIX' in i]]=Xs[[i for i in Xs.columns if 'changeVIX' in i]].shift(1)


Xs[[i+'shift4' for i in Xs.columns if 'changeVIX' in i]]=Xs[[i for i in Xs.columns if 'changeVIX' in i]].shift(4)


Xs['market'] = cross['^GSPC']
Xs['log_market'] = np.log(cross['^GSPC'])
Xs['market_change'] = cross['^GSPC']-cross['^GSPC'].shift(1)
Xs['log_market_change'] = np.log(cross['^GSPC'])-np.log(cross['^GSPC'].shift(1))


Xs.corr()

M = Xs[[i for i in Xs.columns if 'changeVIX' in i]+[i for  i in Xs.columns.values if 'market' in i]+
       ['^VIX','logVIX','VIX_-1', 'VIX_-2', 'VIX_-3', 'VIX_-4']].copy()
M['int']=1

M.dropna(inplace=True)

val = M.tail(300).copy()
train = M[~M.index.isin(val.index)].copy()

features = [i for i in Xs.columns if ('changeVIX' in i)&('shift' in i)] + \
    ['int']+['VIX_-1', 'VIX_-2', 'VIX_-3', 'VIX_-4']
kpi = 'log_market_change'

xt = train[features]
yt = train[kpi]

xv = val[features]
yv = val[kpi]

xv[features]

from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators =100,
                   booster ='gbtree',
                   n_jobs =4,
                   max_depth =10,
                   colsample_bytree=1,colsample_bylevel=1,colsample_bynode=1)

xgb.fit(xt,yt)

ypred = xgb.predict(xv)
ypred=pd.Series(ypred,index = yv.index)


yfit = xgb.predict(xt)
yfit=pd.Series(yfit,index = yt.index)


plt.plot(yv)
plt.plot(ypred)
plt.xticks(rotation=90)


plt.plot(yt)
plt.plot(yfit)
plt.xticks(rotation=90)

plt.scatter(yv,ypred)

1-sum(((yv)-(ypred))**2)/sum(((yv)-np.mean((yv)))**2)
1-sum((np.exp(yv)-np.exp(ypred))**2)/sum((np.exp(yv)-np.mean(np.exp(yv)))**2)



ax1.plot(yv)
plt.xticks(rotation=90)
ax2 = ax1.twinx()
ax2.plot(val['market'],color='red')
plt.xticks(rotation=90)

pd.set_option("display.max_columns",100)

fig,ax1 = plt.subplots()
ax1.plot(val['^VIX'])
plt.xticks(rotation=90)
ax2 = ax1.twinx()
ax2.plot(val['market'],color='red')
plt.xticks(rotation=90)

#Regress sp500

plt.plot(M['market'])
plt.show()

M['daynum'] = range(1,len(M)+1)
M['log_daynum'] = np.log(M['daynum'])

val = M.tail(300).copy()
train = M[(~M.index.isin(val.index))].copy()

features = ['int','daynum']
kpi = 'log_market'

xt = train[features]
yt = train[kpi]


weight = xt['daynum']
w = np.diag((weight/max(weight))**3.8)

xv = val[features]
yv = val[kpi]

coefs = np.linalg.pinv(np.array(xt.T@w)@xt)@(np.array(xt.T@w)@yt)

yfit=xt@coefs
ypred=xv@coefs

plt.plot(yv)
plt.plot(ypred)
plt.xticks(rotation=90)


plt.plot(yt)
plt.plot(yfit)
plt.xticks(rotation=90)
plt.show()

def model(x):
    coefs = np.linalg.pinv(x[features].T@x[features])@(x[features].T@x[kpi])
    return (x[features]@coefs)


def model_coefs(x):
    coefs = np.linalg.pinv(x[features].T@x[features])@(x[features].T@x[kpi])
    return coefs[0],coefs[1]
    
rolling_period = 1200
coef_list = []
for i in range(0,len(train)-rolling_period):
    coef_list.append(model_coefs(train[i:i+rolling_period]))
    if i%100 ==0:
        print(i)


train[[i+"_coef" for  i in features]] = pd.DataFrame(coef_list ,index= train.index[range(rolling_period,len(train))])


comp =  pd.DataFrame([train[features[i]]*train[[n+"_coef" for  n in features][i]] for  i in range(len(features))]).T

train['fit'] = np.sum(comp,axis=1)

    
plt.plot(train[train['fit']>0][kpi])
plt.plot(train[train['fit']>0]['fit'])
plt.show()





coefs = np.array(train[[i+"_coef" for  i in features]].tail(1))

comp =  pd.DataFrame([val[features[i]]*coefs[0][i] for  i in range(len(features))]).T

val['fit']=np.sum(comp,axis=1)


    
plt.plot(train[train['fit']>0][kpi])
plt.plot(train[train['fit']>0]['fit'])
plt.plot(val[val['fit']>0][kpi])
plt.plot(val[val['fit']>0]['fit'])
plt.show()


plt.plot(val[val['fit']>0][kpi])
plt.plot(val[val['fit']>0]['fit'])
plt.show()




train['rolling_']=train[kpi].rolling(window = rolling_period).mean()



