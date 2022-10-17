


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



M = dat[[(i,'Close') for i in symbols]].copy()



symbol_cols = [i.replace("^","") for i in symbols]

M.columns = symbol_cols

for  i in symbol_cols:
    rm = M[i].rolling(4,min_periods=1).mean()
    M[i]=np.where(M[i].isna(),rm,M[i])

for i in symbol_cols:
    for  j in [2,5,8]:
        M[i+"_shift"+str(j)] = M[i].shift(j)
        

for i in symbol_cols:
    for  j in [2,5,8]:
        M[i+"_change"+str(j)] = M[i]/M[i].shift(j)-1

for i in [col for col in M.columns.values if '_change' in col]:
    for j in [2,5,8]:
        M[i+"_shift"+str(j)] = M[i].shift(j)
    
for i in [col for col in M.columns.values if ('_shift' in col)&('_change' in col)]:
    for j in [3,6,10,16]:
        M[i+"_rolling"+str(j)] = M[i].rolling(window=j,min_periods=1).mean()


pd.set_option("display.max_columns",32)

market = 'GSPC'

M.dropna(inplace=True)


#M[M['TNX_change8_shift8_rolling10'].isna()]

#M['TNX_change8_shift8_rolling10'].tail(50)
#M['TNX_change8_shift8'].tail(50)
#M['TNX_change8'].tail(50)
#M['TNX'].tail(50)


for j in [2,5,8]:
    M[market+"_pct_change_day"+str(j)] = M[market].shift(-j)/M[market]


M['daynum'] = range(1,len(M)+1) 
M['int'] = 1


M[[i for  i in M.columns.values if market+"_pct" in i] +[market]].tail(30)

M.columns.values

kpi = market+"_pct_change_day8"

#kpi = market

features = [i for i in M.columns.values if ('_shift' in i)&('_change' in i)]+['int']#+['daynum']
#features = [i for i in M.columns.values if ('_shift' in i)&('_change' in i)&('_rolling' in i)]+['int']#+['daynum']


split_train = dt.datetime.strftime(dt.datetime.today()-dt.timedelta(60),'%Y-%m-%d')
split_test = dt.datetime.strftime(dt.datetime.today()-dt.timedelta(25),'%Y-%m-%d')

xt = M[M.index<=split_train][features].copy()
yt = M[M.index<=split_train][kpi]


xv = M[(M.index>split_train)&(M.index<=split_test)][features].copy()
yv = M[(M.index>split_train)&(M.index<=split_test)][kpi]


xtest = M[M.index>split_test][features].copy()
ytest = M[M.index>split_test][kpi]
    
#Linear Model
expon =True
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

#KERAS MODEL


features = [i for i in M.columns.values if ('_shift' in i)&('_change' in i)]#+['int']#+['daynum']


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
    keras.layers.Dense(20, activation=keras.activations.linear),
    keras.layers.Dropout(.15),
    keras.layers.Dense(10, activation=keras.activations.sigmoid),
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
              metrics=[keras.metrics.MeanSquaredError(),
                       keras.metrics.MeanAbsolutePercentageError()])


model.fit(
        #x=np.array(xt).reshape((xt.shape[0],xt.shape[1],1)),
        x=np.array(xt).reshape(xt.shape[0],xt.shape[1],),
          y=np.array(yt), 
          batch_size = 2**10,
          validation_data=(np.array(xv).reshape(xv.shape[0],xv.shape[1],),np.array(yv)),
          epochs=6000)

model.summary()


ypred = model.predict(np.array(xv))
ypred = ypred.reshape(ypred.shape[0])

yfit = model.predict(np.array(xt))
yfit = yfit.reshape(yfit.shape[0])
    

plt.plot(yt)
plt.plot(pd.Series(yfit,index=yt.index))
plt.title('nn fit')
plt.show()
plt.plot(yv)
plt.plot(pd.Series(ypred,index=yv.index))
plt.title('nn pred')
plt.show()
    
    
    
model.fit(
        #x=np.array(xt).reshape((xt.shape[0],xt.shape[1],1)),
        x=np.array(xt).reshape(xt.shape[0],xt.shape[1],),
          y=np.array(yt), 
          batch_size = 2**7,
          validation_data=(np.array(xv).reshape(xv.shape[0],xv.shape[1],),np.array(yv)),
          epochs=1200)
    
    
    
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
plt.scatter(yfit,yt)
plt.scatter(ypred,yv)
plt.xlim(.8,1.2)
plt.ylim(.8,1.2)    
plt.show()

pd.DataFrame({'y':yv,'pred':ypred}).corr()


#!mkdir -p saved_models

model.save('saved_models/signal_model.h5')


#KERAS MODEL


features = [i for i in M.columns.values if ('_shift' in i)&('_change' in i)]#+['int']#+['daynum']


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
          #batch_size = 2**9,
          validation_data=(np.array(xv).reshape(xv.shape[0],xv.shape[1],),np.array(yv)),
          epochs=600)

model.fit(
        #x=np.array(xt).reshape((xt.shape[0],xt.shape[1],1)),
        x=np.array(xt).reshape(xt.shape[0],xt.shape[1],),
          y=np.array(yt), 
          batch_size = 2**9,
          validation_data=(np.array(xv).reshape(xv.shape[0],xv.shape[1],),np.array(yv)),
          epochs=3000)


model.fit(
        #x=np.array(xt).reshape((xt.shape[0],xt.shape[1],1)),
        x=np.array(xt).reshape(xt.shape[0],xt.shape[1],),
          y=np.array(yt), 
          #batch_size = 2**9,
          validation_data=(np.array(xv).reshape(xv.shape[0],xv.shape[1],),np.array(yv)),
          epochs=600)

model.fit(
        #x=np.array(xt).reshape((xt.shape[0],xt.shape[1],1)),
        x=np.array(xt).reshape(xt.shape[0],xt.shape[1],),
          y=np.array(yt), 
          batch_size = 2**9,
          validation_data=(np.array(xv).reshape(xv.shape[0],xv.shape[1],),np.array(yv)),
          epochs=3000)
model.summary()


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

pd.DataFrame({'y':yv,'pred':ypred}).corr()

yv_flag = np.where(yv>1,1,0)
ypred_flag = np.where(ypred>1,1,0)

np.mean(np.where(ypred_flag==yv_flag,1,0))


model.save('saved_models/model_15L15R15S1L.h5')



1-sum((yv-ypred)**2)/sum((yv-np.mean(yv))**2)
pd.merge(left=xt,right=yt,left_index=True,right_index=True,how='left').corr()['GSPC_pct_change_day8']


pd.merge(left=xv,right=yv,left_index=True,right_index=True,how='left').corr()['GSPC_pct_change_day8']
