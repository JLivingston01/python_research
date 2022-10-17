



import yfinance as yf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

import datetime as dt


import tensorflow
from tensorflow import keras
from tensorflow import losses

symbols = ['^VIX','^DJI','^GSPC','^TNX']
    



dat = yf.download(tickers = " ".join(symbols),
                  period = "max",
                  interval = "1d",
                  group_by = 'ticker')

dat.columns.values



M = dat[[(i,'Close') for i in symbols]].copy()
N = dat[[(i,'Volume') for i in symbols]].copy()

M=M.merge(N,left_index=True,right_index=True,how='left')

M.columns = [i[0].replace("^","")+"_"+i[1] for i in M.columns]

for  i in M.columns:
    rm = M[i].rolling(4,min_periods=1).mean()
    M[i]=np.where(M[i].isna(),rm,M[i])


#KPI is SP500 % change over 15 days
    
M['GSPC_Close_change15'] = M['GSPC_Close'].shift(-15)/M['GSPC_Close']
    


#SP500 rolling momentum, shifted 2 days
M['GSPC_Close_rolling125'] = M['GSPC_Close'].rolling(window=125).mean()
M['GSPC_Close_momentum125'] = M['GSPC_Close']/M['GSPC_Close_rolling125']-1
M['GSPC_Close_momentum125_shift2']=M['GSPC_Close_momentum125'].shift(2)


M['GSPC_Close_rolling30'] = M['GSPC_Close'].rolling(window=30).mean()
M['GSPC_Close_momentum30'] = M['GSPC_Close']/M['GSPC_Close_rolling30']-1
M['GSPC_Close_momentum30_shift2']=M['GSPC_Close_momentum30'].shift(2)

M['GSPC_Close_rolling15'] = M['GSPC_Close'].rolling(window=15).mean()
M['GSPC_Close_momentum15'] = M['GSPC_Close']/M['GSPC_Close_rolling15']-1
M['GSPC_Close_momentum15_shift2']=M['GSPC_Close_momentum15'].shift(2)

#DJI rolling momentum shifted 2 days
M['DJI_Close_rolling125'] = M['DJI_Close'].rolling(window=125).mean()
M['DJI_Close_momentum125'] = M['DJI_Close']/M['DJI_Close_rolling125']-1
M['DJI_Close_momentum125_shift2']=M['DJI_Close_momentum125'].shift(2)


M['DJI_Close_rolling30'] = M['DJI_Close'].rolling(window=30).mean()
M['DJI_Close_momentum30'] = M['DJI_Close']/M['DJI_Close_rolling30']-1
M['DJI_Close_momentum30_shift2']=M['DJI_Close_momentum30'].shift(2)


M['DJI_Close_rolling15'] = M['DJI_Close'].rolling(window=15).mean()
M['DJI_Close_momentum15'] = M['DJI_Close']/M['DJI_Close_rolling15']-1
M['DJI_Close_momentum15_shift2']=M['DJI_Close_momentum15'].shift(2)

# VIX change, shifted 2 days

M["VIX_Close_change1_shift2"]=(M["VIX_Close"]/M["VIX_Close"].shift(1)-1).shift(2)
M["VIX_Close_change1_shift2_rolling10dev"]=M["VIX_Close_change1_shift2"].rolling(window=10).std()
M["VIX_Close_change1_shift2_rolling10mean"]=M["VIX_Close_change1_shift2"].rolling(window=10).mean()

# SP500 Change shifted 2 days

M["GSPC_Close_change1_shift2"]=(M['GSPC_Close']/M['GSPC_Close'].shift(1)-1).shift(2)
M["GSPC_Close_change8_shift2"]=(M['GSPC_Close']/M['GSPC_Close'].shift(8)-1).shift(2)
M["GSPC_Close_change20_shift2"]=(M['GSPC_Close']/M['GSPC_Close'].shift(20)-1).shift(2)


M["DJI_Close_change1_shift2"]=(M['DJI_Close']/M['DJI_Close'].shift(1)-1).shift(2)
M["DJI_Close_change8_shift2"]=(M['DJI_Close']/M['DJI_Close'].shift(8)-1).shift(2)
M["DJI_Close_change20_shift2"]=(M['DJI_Close']/M['DJI_Close'].shift(20)-1).shift(2)

#SP500 volume change, shifted 2 days

M['GSPC_Volume_change1_shift2']=(M["GSPC_Volume"]/M["GSPC_Volume"].shift(1)-1).shift(2)
M['GSPC_Volume_change10_shift2']=(M["GSPC_Volume"]/M["GSPC_Volume"].shift(10)-1).shift(2)

M['GSPC_Volume_change1_shift2_rolling10']=M['GSPC_Volume_change1_shift2'].rolling(window=10).mean()
M['GSPC_Volume_change10_shift2_rolling10']=M['GSPC_Volume_change10_shift2'].rolling(window=10).mean()


for i in ['GSPC_Close_change1_shift2',
       'GSPC_Close_change8_shift2',
       'GSPC_Close_change20_shift2',
       'DJI_Close_change1_shift2',
       'DJI_Close_change8_shift2',
       'DJI_Close_change20_shift2']:
    M[i+"_rolling10"] = M[i].rolling(window=10).mean()
    
    
M.columns
#Intercept
M['int'] = 1

M.dropna(inplace=True)

features = [
       'GSPC_Close_momentum125_shift2', 
       'GSPC_Close_momentum30_shift2',
       'DJI_Close_momentum125_shift2', 
       'DJI_Close_momentum30_shift2',
       'GSPC_Volume_change1_shift2',
       'GSPC_Volume_change10_shift2',
       'GSPC_Volume_change1_shift2_rolling10',
       'GSPC_Volume_change10_shift2_rolling10',
       'GSPC_Close_change1_shift2',
       'GSPC_Close_change8_shift2',
       'GSPC_Close_change20_shift2',
       'DJI_Close_change1_shift2',
       'DJI_Close_change8_shift2',
       'DJI_Close_change20_shift2',
       'VIX_Close_change1_shift2',
       'GSPC_Close_change1_shift2_rolling10',
       'GSPC_Close_change8_shift2_rolling10',
       'GSPC_Close_change20_shift2_rolling10',
       'DJI_Close_change1_shift2_rolling10',
       'DJI_Close_change8_shift2_rolling10',
       'DJI_Close_change20_shift2_rolling10',
       'VIX_Close_change1_shift2_rolling10dev',
       'VIX_Close_change1_shift2_rolling10mean',
       
       'int']

kpi = 'GSPC_Close_change15'


M.corr()[kpi]

split_train = dt.datetime.strftime(dt.datetime.today()-dt.timedelta(60),'%Y-%m-%d')
split_test = dt.datetime.strftime(dt.datetime.today()-dt.timedelta(25),'%Y-%m-%d')

xt = M[M.index<=split_train][features].copy()
yt = M[M.index<=split_train][kpi]


xv = M[(M.index>split_train)&(M.index<=split_test)][features].copy()
yv = M[(M.index>split_train)&(M.index<=split_test)][kpi]


xtest = M[M.index>split_test][features].copy()
ytest = M[M.index>split_test][kpi]
    
#Linear Model
expon =False
if expon ==True:
    coefs= np.linalg.pinv(xt.T@xt)@(xt.T@np.log(yt))
    
    yfit=np.exp(xt@coefs)
    ypred=np.exp(xv@coefs)
else:
    coefs= np.linalg.pinv(xt.T@xt)@(xt.T@(yt))
    
    yfit=xt@coefs
    ypred=xv@coefs

plt.plot(yt)
plt.plot(yfit)
plt.title('regression fit')
plt.xticks(rotation=90)
plt.show()
plt.plot(yv)
plt.plot(ypred)
plt.title('regression pred')
plt.xticks(rotation=90)
plt.show()

pd.DataFrame({'y':yv,'pred':ypred}).corr()


fig,ax1=plt.subplots()
ax1.plot(yv)
plt.xticks(rotation=90)
ax2=ax1.twinx()
ax2.plot(ypred,color='orange')
plt.title('regression pred')
plt.xticks(rotation=90)
plt.show()



#xgboost

import xgboost

xgb = xgboost.XGBRegressor(n_estimators=100,
                           max_depth =9,
                           colsample_bytree=1 ,
                           colsample_bylevel=.9,
                           colsample_bynode =.9,
                           n_jobs=4
                           ).fit(xt,yt)


yfit=xgb.predict(xt)
ypred=xgb.predict(xv)

plt.plot(yt)
plt.plot(pd.Series(yfit,index=yt.index))
plt.title('regression fit')
plt.xticks(rotation=90)
plt.show()
plt.plot(yv)
plt.plot(pd.Series(ypred,index=yv.index))
plt.title('regression pred')
plt.xticks(rotation=90)
plt.show()

print(pd.DataFrame({'y':yv,'pred':ypred}).corr())

print(round(np.mean((yv-ypred)**2),5))



#Keras nN

features = [
        'GSPC_Close_momentum125_shift2', 
       'GSPC_Close_momentum30_shift2',
       'DJI_Close_momentum125_shift2', 
       'DJI_Close_momentum30_shift2',
       'GSPC_Volume_change1_shift2',
       'GSPC_Volume_change10_shift2',
       'GSPC_Volume_change1_shift2_rolling10',
       'GSPC_Volume_change10_shift2_rolling10',
       'GSPC_Close_change1_shift2',
       'GSPC_Close_change8_shift2',
       'GSPC_Close_change20_shift2',
       'DJI_Close_change1_shift2',
       'DJI_Close_change8_shift2',
       'DJI_Close_change20_shift2',
       'VIX_Close_change1_shift2',
       'GSPC_Close_change1_shift2_rolling10',
       'GSPC_Close_change8_shift2_rolling10',
       'GSPC_Close_change20_shift2_rolling10',
       'DJI_Close_change1_shift2_rolling10',
       'DJI_Close_change8_shift2_rolling10',
       'DJI_Close_change20_shift2_rolling10',
       'VIX_Close_change1_shift2_rolling10dev',
       'VIX_Close_change1_shift2_rolling10mean',
       ]

split_train = dt.datetime.strftime(dt.datetime.today()-dt.timedelta(60),'%Y-%m-%d')
split_test = dt.datetime.strftime(dt.datetime.today()-dt.timedelta(25),'%Y-%m-%d')

xt = M[M.index<=split_train][features].copy()
yt = M[M.index<=split_train][kpi]


xv = M[(M.index>split_train)&(M.index<=split_test)][features].copy()
yv = M[(M.index>split_train)&(M.index<=split_test)][kpi]


xtest = M[M.index>split_test][features].copy()
ytest = M[M.index>split_test][kpi]

model = keras.Sequential([
    #keras.layers.LSTM(units=10),
    #keras.layers.Dense(len(xt.columns.values), activation=keras.activations.relu),
    keras.layers.Dense(15, activation=keras.activations.linear),
    keras.layers.Dropout(.15),
    keras.layers.Dense(15, activation=keras.activations.relu),
    keras.layers.Dropout(.15),
    keras.layers.Dense(15, activation=keras.activations.sigmoid),
    #keras.layers.Dense(len(xt.columns.values), activation=keras.activations.linear),
    #keras.layers.Dense(15, activation=keras.activations.linear),
    #keras.layers.Dense(1,activation = keras.activations.linear),
    keras.layers.Dense(1,activation = keras.activations.linear)
    #keras.layers.Dense(1,activation = keras.activations.exponential)
    ])

model.compile(optimizer=tensorflow.optimizers.Adam(), 
              loss=losses.mean_squared_error,
              #batch_size=32,
              #loss=losses.categorical_crossentropy,
              metrics=[keras.metrics.MeanSquaredError()])


model.fit(
        #x=np.array(xt).reshape((xt.shape[0],xt.shape[1],1)),
        x=np.array(xt).reshape(xt.shape[0],xt.shape[1],),
          y=np.array(yt), 
          batch_size = 2**10,
          validation_data=(np.array(xv).reshape(xv.shape[0],xv.shape[1],),np.array(yv)),
          epochs=24000)

model.fit(
        #x=np.array(xt).reshape((xt.shape[0],xt.shape[1],1)),
        x=np.array(xt).reshape(xt.shape[0],xt.shape[1],),
          y=np.array(yt), 
          #batch_size = 2**9,
          validation_data=(np.array(xv).reshape(xv.shape[0],xv.shape[1],),np.array(yv)),
          epochs=1800)



model.fit(
        #x=np.array(xt).reshape((xt.shape[0],xt.shape[1],1)),
        x=np.array(xt).reshape(xt.shape[0],xt.shape[1],),
          y=np.array(yt), 
          batch_size = 2**6,
          validation_data=(np.array(xv).reshape(xv.shape[0],xv.shape[1],),np.array(yv)),
          epochs=10000)

model.fit(
        #x=np.array(xt).reshape((xt.shape[0],xt.shape[1],1)),
        x=np.array(xt).reshape(xt.shape[0],xt.shape[1],),
          y=np.array(yt), 
          #batch_size = 2**9,
          validation_data=(np.array(xv).reshape(xv.shape[0],xv.shape[1],),np.array(yv)),
          epochs=1800)


ypred = model.predict(np.array(xv))
ypred = ypred.reshape(ypred.shape[0])

yfit = model.predict(np.array(xt))
yfit = yfit.reshape(yfit.shape[0])
    


plt.plot(yt)
plt.plot(pd.Series(yfit,index=yt.index))
plt.title('nn fit')
plt.xticks(rotation=90)
plt.show()
plt.plot(yv)
plt.plot(pd.Series(ypred,index=yv.index))
plt.title('nn pred')
plt.xticks(rotation=90)
plt.show()
    
fig = plt.figure(figsize=(8,8))
plt.scatter(yfit,yt,s=.5)
plt.scatter(ypred,yv,s=3)
plt.xlim(.8,1.2)
plt.ylim(.8,1.2)    
plt.show()


print(pd.DataFrame({'y':yv,'pred':ypred}).corr())

print(round(np.mean((yv-ypred)**2),5))
