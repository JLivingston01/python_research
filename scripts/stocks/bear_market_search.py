



import yfinance as yf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

import datetime as dt


indeces = ['MSFT','^CPI','^DJI','^GSPC','^TNX','^VIX']
positions=[]
watching=[]

symbols = list(set(indeces+positions+watching))


tickers = yf.Tickers(' '.join(symbols))

sp500=tickers.tickers['^GSPC'].history(period="max")
sp500['date'] = sp500.index
sp500['year'] = sp500['date'].dt.year


sp500['cume_high']=sp500['High'].cummax()

sp500['Record']=np.where(sp500['cume_high']==sp500['High'],1,0)



cpi=pd.read_csv("cpi.csv")

cpi=cpi[(~cpi['period'].isna())].copy()

cpi=cpi[~(cpi['period'].str.contains("HALF"))].copy()

cpi['date']=pd.to_datetime(cpi['period'].str.replace("\n","-"))
cpi['month']=cpi['date'].dt.strftime("%Y-%m")

cpi['period']=cpi['date']
cpi.set_index('period',inplace=True)



sp500['month']=sp500['date'].dt.strftime("%Y-%m")

sp500=sp500.merge(cpi[['month','Original Data Value']],on=['month'],how='left')

sp500['year']=pd.to_datetime(sp500['date']).dt.year


hhic = pd.read_csv("median_household_income.csv")

hhic['year']=pd.to_datetime(hhic['DATE']).dt.year
hhic.columns = ['date','Median Income','year']

eom=sp500.groupby(['month']).agg({'date':'max'}).reset_index().merge(
        sp500[['date','Close']],on='date',how='left')

eom['year']=pd.to_datetime(eom['date']).dt.year

eom=eom.merge(hhic[['year','Median Income']],how='left')



eom=eom.merge(cpi[['month','Original Data Value']],on=['month'],how='left')
eom.columns = ['month','date','Close','year','Median Income','CPI']


eom['CPI 1MO %Change']=eom['CPI']/eom['CPI'].shift(1)-1
eom['CPI 1Y %Change']=eom['CPI']/eom['CPI'].shift(12)-1
eom['Close 1MO %Change']=eom['Close']/eom['Close'].shift(1)-1
eom['Close 1Y %Change']=eom['Close']/eom['Close'].shift(12)-1

eom.index=eom['date']

eom['Log Close']=np.log(eom['Close'])
eom['Log CPI']=np.log(eom['CPI'])


eom['Log Close 1MO %Change']=eom['Log Close']/eom['Log Close'].shift(1)-1
eom['Log CPI 1MO %Change']=eom['Log CPI']/eom['Log CPI'].shift(1)-1
eom['Month S&P500 Close - Log Scale']=eom['Log Close']



def make_dual_plot(x,y):
    fig,ax1=plt.subplots()
    ax1.plot(x,color='blue')
    plt.ylabel(x.name,color='blue')
    ax2=ax1.twinx()
    ax2.plot(y,color='red')
    plt.ylabel(y.name,color='red')
    plt.show()





eom.tail(300).corr()




make_dual_plot(eom['CPI 1Y %Change'].tail(358),
               eom['Month S&P500 Close - Log Scale'].tail(358))


make_dual_plot(eom['CPI 1Y %Change'].tail(600),
               eom['Month S&P500 Close - Log Scale'].tail(600))


gdp = pd.read_csv("GDP.csv")

gdp['month']=pd.to_datetime(gdp['DATE']).dt.strftime("%Y-%m")

gdp=eom.merge(gdp[['month','GDP']])
gdp['Log GDP']=np.log(gdp['GDP'])
gdp.index = gdp['date']

gdp['Log Median Income']=np.log(gdp['Median Income'])


make_dual_plot(gdp['Log GDP'].tail(240),
               gdp['Log CPI'].tail(240))



make_dual_plot(gdp['Median Income'].tail(150),
               gdp['CPI 1Y %Change'].tail(150))



make_dual_plot(gdp['Median Income'].tail(150),
               gdp['Month S&P500 Close - Log Scale'].tail(150))

make_dual_plot(gdp['Log Median Income'].tail(150),
               gdp['Log CPI'].tail(150))

def standard_score(col):
    m=np.nanmean(col)
    s=np.nanstd(col)
    std_score=(col-m)/s
    std_score=std_score+np.nanmin(std_score)
    return(std_score)
    
gdp['LMIS']=standard_score(gdp['Log Median Income'])
gdp['LCPIS']=standard_score(gdp['Log CPI'])

gdp['LMIS/LCPIS']=gdp['LMIS']/gdp['LCPIS']

make_dual_plot(gdp['LMIS/LCPIS'].tail(116),
               gdp['Month S&P500 Close - Log Scale'].tail(116))


make_dual_plot(gdp['Log CPI'].tail(116),
               gdp['Log Median Income'].tail(116))

MATRIX = gdp[['Log Median Income','Log CPI']].tail(116).dropna()
X = pd.DataFrame()
X['Log Median Income']=MATRIX['Log Median Income']
X['int']=1
X['monthnum']=range(1,len(X)+1)

Y = MATRIX['Log CPI']

coefs = np.linalg.pinv(X.T@X)@(X.T@Y)

Yfit = X@coefs

plt.plot(Y)
plt.plot(Yfit)
plt.show()

year_end_cpi=gdp.groupby(['year']).agg({'date':'max',
           'Median Income':'min'}).merge(
    gdp[['date','CPI']].reset_index(drop=True),on=['date'],how='left'
    ).dropna()
year_end_cpi.set_index('date',inplace=True)


plt.plot(year_end_cpi['CPI'])
plt.plot(year_end_cpi['Median Income'])
plt.show()


make_dual_plot(year_end_cpi['CPI'],
               year_end_cpi['Median Income'])


X=pd.DataFrame()
X['Median Income']=year_end_cpi['Median Income']
X['int']=1

Y = year_end_cpi['CPI']

coefs = np.linalg.pinv(X.T@X)@(X.T@Y)

Yfit = X@coefs

plt.plot(Y)
plt.plot(Yfit)
plt.show()

resid = Yfit-Y

plt.plot(resid)
plt.plot(gdp['Log Close'])


make_dual_plot(resid,
               gdp['Log Close'])





#Model
CPI_MATRIX=eom[['Log CPI']].tail(358).copy()
CPI_MATRIX['monthnum']=range(1,len(CPI_MATRIX)+1)
CPI_MATRIX['int']=1


CPI_MATRIX=CPI_MATRIX.merge(gdp[['Log GDP','Log Median Income']],left_index=True,right_index=True,how='left')


        

def fill_n(col,n):
    
    index= col.index
    
    for i in range(n):
        col=pd.Series(
                np.where(col.isna(),
              col.shift(1),
              col
              ),index=index)
        
    return pd.Series(col,index=index)

CPI_MATRIX['Log GDP']=fill_n(CPI_MATRIX['Log GDP'],2)
        
CPI_MATRIX['Log Median Income']=fill_n(CPI_MATRIX['Log Median Income'],2)

months = pd.Series(CPI_MATRIX.index,index=CPI_MATRIX.index).dt.month
for i in range(1,13):
    CPI_MATRIX['month_'+str(i)]=np.where(months==i,1,0)
    
CPI_MATRIX.dropna(inplace=True)

train='2016-01-01'
kpi='Log CPI'
feats = ['monthnum','Log GDP','Log Median Income','int']+[i for i in CPI_MATRIX.columns if ('month_' in i)&
        ('7' not in i)]


xt=CPI_MATRIX[CPI_MATRIX.index<train][feats].copy()
yt=CPI_MATRIX[CPI_MATRIX.index<train][kpi].copy()
xv=CPI_MATRIX[CPI_MATRIX.index>=train][feats].copy()
yv=CPI_MATRIX[CPI_MATRIX.index>=train][kpi].copy()

coefs = np.linalg.pinv(xt.T@xt+1e-2)@(xt.T@yt)

yfit=xt@coefs
ypred=xv@coefs

plt.plot(yt,color='Green')
plt.plot(yv,color='Green')
plt.plot(yfit)
plt.plot(ypred)
plt.show()


plt.plot(np.exp(yt),color='Green')
plt.plot(np.exp(yv),color='Green',label='CPI')
plt.plot(np.exp(yfit),label='CPI Model Fit')
plt.plot(np.exp(ypred),label='CPI Predicted')
plt.title("Model: GDP+Seasonality+Trend ~ CPI")
plt.ylabel("CPI")
plt.legend()
plt.show()


train='2016-01-01'
kpi='Log CPI'
feats = ['monthnum','Log Median Income','int']+[i for i in CPI_MATRIX.columns if ('month_' in i)&
        ('7' not in i)]


xt=CPI_MATRIX[CPI_MATRIX.index<train][feats].copy()
yt=CPI_MATRIX[CPI_MATRIX.index<train][kpi].copy()
xv=CPI_MATRIX[CPI_MATRIX.index>=train][feats].copy()
yv=CPI_MATRIX[CPI_MATRIX.index>=train][kpi].copy()

coefs = np.linalg.pinv(xt.T@xt+1e-2)@(xt.T@yt)

yfit=xt@coefs
ypred=xv@coefs

plt.plot(yt,color='Green')
plt.plot(yv,color='Green')
plt.plot(yfit)
plt.plot(ypred)
plt.show()


plt.plot(np.exp(yt),color='Green')
plt.plot(np.exp(yv),color='Green',label='CPI')
plt.plot(np.exp(yfit),label='CPI Model Fit')
plt.plot(np.exp(ypred),label='CPI Predicted')
plt.title("Model: GDP+Seasonality+Trend ~ CPI")
plt.ylabel("CPI")
plt.legend()
plt.show()

















records_set = sp500.groupby(['year']).agg({'Record':'sum'})

plt.plot(records_set['Record'])
plt.show()

for i in range(1,11):
    records_set['lag_'+str(i)]=records_set['Record'].shift(i)

records_set.dropna(inplace=True)


from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


mod=RandomForestRegressor(random_state=42,
                          n_estimators=50)



mod=XGBRegressor(random_state=42,
                          n_estimators=10)

mod=LinearRegression()

mod=SVR(degree=2,kernel='sigmoid')

mod=DecisionTreeRegressor(random_state=42,
                          min_samples_leaf=7
                          
                          )

from sklearn.naive_bayes import CategoricalNB

mod = CategoricalNB()

train = 2010

feats = [i for i in records_set.columns if 'lag_' in i]
kpi='Record'

records_seen = pd.Series(np.where(records_set['Record']>0,1,0),index=records_set.index)


xt=records_set[records_set.index<train][feats]
yt=records_set[records_set.index<train][kpi]
yt=records_seen[records_set.index<train]


xv=records_set[records_set.index>=train][feats]
yv=records_set[records_set.index>=train][kpi]
yv=records_seen[records_set.index>=train]

mod.fit(xt,yt)

yfit=pd.Series(mod.predict(xt),index=xt.index)
ypred=pd.Series(mod.predict(xv),index=xv.index)


plt.plot(records_set['Record'])

plt.plot(records_seen)
plt.plot(yfit)
plt.plot(ypred)
plt.show()


r2 = records_set.copy()

for counter in range(15):
    new_year = pd.DataFrame(r2.T[max(r2.index)].shift(1)).T
    new_year.index=[max(r2.index)+1]
    
    new_records = mod.predict(new_year[feats])[0]
    new_year['Record'].fillna(new_records,inplace=True)
    
    
    r2=r2.append(new_year)

plt.plot(r2['Record'])






cpi=tickers.tickers['^CPI'].history(period="max")

cpi_yearly = pd.read_csv("united-states-inflation-rate-cpi.csv")

cpi_yearly['date']=pd.to_datetime(cpi_yearly['date'])

cpi_yearly.set_index('date',inplace=True)

plt.plot(cpi_yearly[' Inflation Rate (%)'])


eoy=sp500.groupby(['year']).agg({'date':'max'})

eoy=eoy.merge(sp500[['date','Close']],on=['date'],how='left')
eoy.set_index('date',inplace=True)

plt.plot(eoy['Close']/eoy['Close'].shift(1)-1)
plt.plot(cpi_yearly[' Inflation Rate (%)'])


