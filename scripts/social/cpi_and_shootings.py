
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.set_option("display.max_columns",500)

dat = pd.read_csv("c:/users/jliv/downloads/cpi.csv")


dat = pd.melt(dat,value_vars=[i for i in dat.columns if i!='Year'],
        id_vars=['Year'])

dat.columns = ['Year','Month','CPI']


mapper = {'Jan':1,
 'Feb':2,
 'Mar':3,
 'Apr':4,
 'May':5,
 'Jun':6,
 'Jul':7,
 'Aug':8,
 'Sep':9,
 'Oct':10,
 'Nov':11,
 'Dec':12,
 }

dat['Month']=dat['Month'].map(mapper)


dat['YearMonth'] = dat['Year'].astype(str)+"-"+dat['Month'].astype(str)

dat['YearMonth']=pd.to_datetime(dat['YearMonth'])
dat.index = dat['YearMonth']

dat = dat.sort_values(by=['Year','Month'],ascending=[True,True]).reset_index(drop=True)


dat['YoY']=dat['CPI']/dat['CPI'].shift(12)-1

current = dat[dat['Year']>=2006].copy()

current.set_index('YearMonth',inplace=True)

plt.plot(current['YoY'])
plt.show()



shootings = pd.read_csv("c://users/jliv/downloads/shootings.csv")


shootings['OCCUR_DATE'] = pd.to_datetime(shootings['OCCUR_DATE'])

shootings = shootings[['OCCUR_DATE']].copy()

shootings['Month'] = shootings['OCCUR_DATE'].dt.month
shootings['Year'] = shootings['OCCUR_DATE'].dt.year

shootings['YearMonth'] = shootings['Year'].astype(str)+"-"+shootings['Month'].astype(str)
shootings['YearMonth']=pd.to_datetime(shootings['YearMonth'])

shootings = shootings.groupby(['YearMonth']).agg({'OCCUR_DATE':'count'}).reset_index()
shootings.index=shootings['YearMonth']

plt.plot(shootings['OCCUR_DATE'])
plt.show()

shootings['MONTH'] = shootings['YearMonth'].dt.month
for i in range(1,13):
    shootings['MONTH_'+str(i)]=np.where(shootings['MONTH']==i,1,0)
    

shootings['CPI'] = current['YoY']

shootings.corr()

shootings['MONTH_SIN']=np.sin(2*np.pi*shootings['MONTH']/12)
shootings['MONTH_COS']=np.cos(2*np.pi*shootings['MONTH']/12)

X = shootings[['MONTH_1',
 'MONTH_2',
 'MONTH_3',
 'MONTH_4',
 'MONTH_5',
 'MONTH_6',
 'MONTH_7',
 'MONTH_8',
 'MONTH_9',
 'MONTH_10',
 'MONTH_11',
 'MONTH_12',]+['CPI']]
Y = shootings['OCCUR_DATE']

trainind = X.index[:160]
valind = X.index[160:]

xt=X[X.index.isin(trainind)].copy()
xv=X[X.index.isin(valind)].copy()

yt = Y[Y.index.isin(trainind)].copy()
yv  = Y[Y.index.isin(valind)].copy()

coefs = np.linalg.pinv(xt.T@xt)@(xt.T@yt)


yfit = xt@coefs

ypred = xv@coefs


plt.plot(yt,color='blue')
plt.plot(yv,color='blue')
plt.plot(yfit,color='red')
plt.plot(ypred,color='orange')
plt.show()




from sklearn.neighbors import KNeighborsRegressor

X = shootings[['MONTH_SIN','MONTH_COS']+['CPI']]
Y = shootings['OCCUR_DATE']

trainind = X.index[:160]
valind = X.index[160:]

xt=X[X.index.isin(trainind)].copy()
xv=X[X.index.isin(valind)].copy()

yt = Y[Y.index.isin(trainind)].copy()
yv  = Y[Y.index.isin(valind)].copy()

means = np.mean(xt)
stds = np.std(xt)

xt=(xt-means)/stds
xv=(xv-means)/stds

mod = KNeighborsRegressor(n_neighbors=3)

mod.fit(xt,yt)


yfit = pd.Series(mod.predict(xt),index=xt.index)

ypred = pd.Series(mod.predict(xv),index=xv.index)


plt.plot(yt,color='blue')
plt.plot(yv,color='blue')
plt.plot(yfit,color='red')
plt.plot(ypred,color='orange')
plt.show()

