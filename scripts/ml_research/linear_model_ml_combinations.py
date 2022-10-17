


import pandas as pd
import numpy as np
from random import sample
import matplotlib.pyplot as plt
from random import choices

dat=pd.read_csv("c:/users/jliv/downloads/auto-mpg.data",header=None)
"""
    1. mpg:           continuous
    2. cylinders:     multi-valued discrete
    3. displacement:  continuous
    4. horsepower:    continuous
    5. weight:        continuous
    6. acceleration:  continuous
    7. model year:    multi-valued discrete
    8. origin:        multi-valued discrete
    9. car name:      string (unique for each instance)
"""



pd.set_option("display.max_columns",19)
df=pd.DataFrame(dat[0].str.split(expand=True))

df=df[[0,1,2,3,4,5,6,7]].copy()


columns = ['mpg','cyl','disp','hp','weight','acc','yr','origin']

df.columns = columns

df.replace("?",np.nan,inplace=True)

for i in df.columns:
    df[i]=df[i].astype(float)

for i in columns:
    print(i,len(df[df[i].isna()]))

df['int']=1

df.dropna(inplace=True)


train=sample(list(df.index),int(len(df.index)*.8))
train.sort()
test=[i for i in df.index if i not in train]



kpi='mpg'


feats=['cyl', 'disp', 'hp', 'weight', 'acc', 'yr', 'origin','int']

xt=df[df.index.isin(train)][feats].copy()
yt=df[df.index.isin(train)][kpi]

means=np.mean(xt)
stds=np.std(xt)

xt=((xt-means)/stds).fillna(1)



xtest=df[df.index.isin(test)][feats].copy()
xtest=((xtest-means)/stds).fillna(1)

ytest=df[df.index.isin(test)][kpi]

coefs = np.linalg.pinv(xt.T@xt)@(xt.T@yt)

yfit=xt@coefs
ypred=xtest@coefs


plt.scatter(ytest,ypred)
plt.scatter(yt,yfit)
plt.title(kpi)
plt.show()

r2=1-sum((ytest-ypred)**2)/sum((ytest-np.mean(ytest))**2)
print("Linear Model")
print(r2)


featsdf=pd.DataFrame()
featsdf['feats']=columns[1:]+['int']
featsdf['coef']=coefs

print(featsdf)




#G Desc
lr=.001
epoch=5000
coefs_=np.random.normal(0,.01,size=len(xt.columns))
for e in range(0,epoch):

    grad=(yt-xt@coefs_)@xt

    coefs_=coefs_+lr*grad



yfit=xt@coefs_
ypred=xtest@coefs_


plt.scatter(ytest,ypred)
plt.scatter(yt,yfit)
plt.title(kpi)
plt.show()

r2=1-sum((ytest-ypred)**2)/sum((ytest-np.mean(ytest))**2)
print("Linear Model")
print(r2)


#Linear model on CYL and XGBOOST else

import xgboost

linear_feats=['cyl','acc']
other_feats=[i for i in xt.columns if i not in linear_feats]

lr=.001
epoch=10

mod=xgboost.XGBRegressor(n_estimators=50,
                         max_depth=3,
                         verbosity=0
                         )


coefs_ = np.random.normal(0,.01,size=len(linear_feats))

for e in range(0,epoch):
    mod.fit(xt[other_feats],yt-xt[linear_feats]@coefs_)
    grad=(yt-xt[linear_feats]@coefs_-mod.predict(xt[other_feats]))@xt[linear_feats]

    coefs_=coefs_+lr*grad



yfit=xt[linear_feats]@coefs_+mod.predict(xt[other_feats])
ypred=xtest[linear_feats]@coefs_+mod.predict(xtest[other_feats])


plt.scatter(ytest,ypred)
plt.scatter(yt,yfit)
plt.title(kpi)
plt.show()

r2=1-sum((ytest-ypred)**2)/sum((ytest-np.mean(ytest))**2)
print("Linear Model")
print(r2)


class mixed_model:
    
    def __init__(self,mod,lr,epoch):

        self.lr=lr
        self.epoch=epoch
        self.mod=mod

        
    def fit(self,x,y,linear_feats):
        
        self.x=x
        self.y=y
        self.linear_feats=linear_feats        
        self.other_feats=[i for i in self.x.columns if i not in self.linear_feats]
        self.coefs_=np.random.normal(0,.5,len(self.linear_feats))
        
        for e in range(0,epoch):
            self.mod.fit(self.x[self.other_feats],self.y-self.x[self.linear_feats]@self.coefs_)
            grad=(self.y-self.x[self.linear_feats]@self.coefs_-self.mod.predict(self.x[self.other_feats]))@self.x[self.linear_feats]
        
            self.coefs_=self.coefs_+self.lr*grad
        
    def predict(self,x):
        
        pred=x[self.linear_feats]@self.coefs_+self.mod.predict(x[self.other_feats])
        
        return pred
    
    
    
#MIXED MODEL
    
mixed_mod=mixed_model(mod=xgboost.XGBRegressor(n_estimators=25,
                         max_depth=3,
                         verbosity=0
                         ),
    lr=.01,
    epoch=1000
    )
    
mixed_mod.fit(xt,yt,linear_feats=['cyl','acc'])

ypred=mixed_mod.predict(xtest)
yfit=mixed_mod.predict(xt)

print("Mixed Model")
plt.scatter(yt,yfit)
plt.scatter(ytest,ypred)
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title(kpi)
plt.show()

r2=1-sum((ytest-ypred)**2)/sum((ytest-np.mean(ytest))**2)

print("R2",r2)

print('RMSE',np.mean((ytest-ypred)**2)**.5)
print(mixed_mod.coefs_)

#LINEAR MODEL
coefs = np.linalg.pinv(xt.T@xt)@(xt.T@yt)

yfit=xt@coefs
ypred=xtest@coefs

print("Linear Model")
plt.scatter(yt,yfit)
plt.scatter(ytest,ypred)
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title(kpi)
plt.show()

r2=1-sum((ytest-ypred)**2)/sum((ytest-np.mean(ytest))**2)

print("R2",r2)

print('RMSE',np.mean((ytest-ypred)**2)**.5)

featsdf=pd.DataFrame()
featsdf['feats']=columns[1:]+['int']
featsdf['coef']=coefs

print(featsdf)