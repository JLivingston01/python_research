

import pandas as pd
import numpy as np
from random import sample
from xgboost import XGBRegressor
from random import choices,seed
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec

from scipy.stats import t


#import os

#os.chdir("c://users/jliv/downloads/")

dat=pd.read_csv("auto-mpg.data",header=None)
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
    
df.dropna(inplace=True)



seed(42)
train=sample(list(df.index),int(len(df.index)*.8))
train.sort()
test=[i for i in df.index if i not in train]



kpi='mpg'


feats=['cyl', 'disp', 'hp', 'weight', 'acc', 'yr', 'origin']

X=df[df.index.isin(train)][feats].copy()
Y=df[df.index.isin(train)][kpi]


xtest=df[df.index.isin(test)][feats].copy()
ytest=df[df.index.isin(test)][kpi]

means=np.mean(X)
stds=np.std(X)

X=(X-means)/stds
xtest=(xtest-means)/stds

corrdf=X.copy()
corrdf[kpi]=Y
corrdf.corr()[kpi]
corrdf.corr()


seed(42)

fold = pd.Series(choices(range(1,9),k=len(X)),index=X.index)



kpi='mpg'


feats=['cyl', 'disp', 'hp', 'weight', 'acc', 'yr', 'origin']

xt=X.copy()
yt=Y.copy()

xv=xtest.copy()
yv=ytest.copy()

xt['int']=1
xv['int']=1


def r2_fn(true,pred):
    r2=1-sum((true-pred)**2)/sum((true-np.mean(true))**2)
    return r2



class partial_regression:
    
    def __init__(self,mod,max_iter,min_iter,burn_in=.7):
        
        self.epochs=max_iter
        self.min_iter=min_iter
        self.mod=mod
        self.burn_in=burn_in
    
    def check_converge(self,chain):
        X1= chain[int(len(chain)*self.burn_in):int(len(chain)*(1-(1-self.burn_in)/2))]
        X2= chain[int(len(chain)*(1-(1-self.burn_in)/2)):]
        
        v=len(X1)+len(X2)-2
        
        absT=abs((np.mean(X1)-np.mean(X2))/((np.var(X1)/len(X1)+np.var(X2)/len(X2))**.5))
        
        if absT<=t.ppf(1-.05/2, v):
            return 1
        else:
            return 0

    def fit(self,x,y,linear_feats):
        
        self.linear_feats = linear_feats
        self.other_feats = [i for i in x.columns.values if i not in self.linear_feats]
        
        self.coefs_=np.linalg.pinv(x[self.linear_feats].T@x[self.linear_feats])@(x[self.linear_feats].T@y)
        
        self.coefs_arr = []
        self.RMSE_chain=[]
        for e in range(1,self.epochs+1):
        
            self.mod.fit(xt[self.other_feats], yt-xt[self.linear_feats]@self.coefs_)
            
            self.coefs_ = np.linalg.pinv(xt[self.linear_feats].T@xt[self.linear_feats])@(xt[self.linear_feats].T@(yt-self.mod.predict(xt[self.other_feats])))
        
            resid = yt - (xt[self.linear_feats]@self.coefs_ + self.mod.predict(xt[self.other_feats]))
            
            RMSE = np.mean(resid**2)**.5
            
            self.coefs_arr.append(list(self.coefs_))
            self.RMSE_chain.append(RMSE)
            
            if e>self.min_iter:        
                self.converged = [self.check_converge(self.RMSE_chain)]+[self.check_converge(c) for c in np.array(self.coefs_arr).T]
                
                if self.converged[0]==1:
                    if min(self.converged[1:])<1:
                        print("Warning: Cost converged but some parameters may not have converged. Consider increasing min_iter.")
                    else:
                        print("Success: Cost and parameters converged.")
                    break
                elif (e == self.epochs):
                        print("Warning: All epochs completed and cost hasn't converged. Consider increasing max_iter.")
        
         
        self.coef_means=pd.Series(np.mean(np.array(self.coefs_arr)[int(e*self.burn_in):],axis=0),index=self.linear_feats)
        
        
        self.mod.fit(x[self.other_feats],y-x[self.linear_feats]@self.coef_means)
        
    
    def predict(self,x):
        
        #self.mod.fit(self.x[self.other_feats],self.y-self.x[self.linear_feats]@self.coef_means)
        pred=x[self.linear_feats]@self.coef_means+self.mod.predict(x[self.other_feats])
        
        return pred

linear_feats=['weight',
                      'disp',]


mod = partial_regression(
        
        mod = XGBRegressor(n_estimators=25,
                         max_depth=6,
                         random_state=42),
        max_iter=1000,
        min_iter=750,
        burn_in=.7
        )
        
mod.fit(xt,yt,linear_feats)

plt.plot(np.array(mod.RMSE_chain))
plt.show()

for i in range(len(mod.linear_feats)):
    plt.plot(np.array(mod.coefs_arr)[:,i])
    plt.title(mod.linear_feats[i])
    plt.show()


ypred=mod.predict(xv)


print('no intercept partial ',r2_fn(yv,ypred))



#LR
coefs = np.linalg.pinv(xt.T@xt)@(xt.T@yt)

ypred_2=xv@coefs


print('LR',r2_fn(yv,ypred_2))
#XGB

xgb = XGBRegressor(n_estimators=25,
                 max_depth=6,
                 random_state=42)
xgb.fit(xt,yt)


ypred_3=xgb.predict(xv)


print('xgb ',r2_fn(yv,ypred_3))



#Intercept Partial
linear_feats=['weight',
                      'disp',
                      'int']


mod = partial_regression(
        
        mod = XGBRegressor(n_estimators=25,
                         max_depth=6,
                         random_state=42),
        max_iter=1000,
        min_iter=750,
        burn_in=.7
        )
        
mod.fit(xt,yt,linear_feats)

plt.plot(np.array(mod.RMSE_chain))
plt.show()

for i in range(len(mod.linear_feats)):
    plt.plot(np.array(mod.coefs_arr)[:,i])
    plt.title(mod.linear_feats[i])
    plt.show()


ypred=mod.predict(xv)

def r2_fn(true,pred):
    r2=1-sum((true-pred)**2)/sum((true-np.mean(true))**2)
    return r2


print('intercept partial ',r2_fn(yv,ypred))
















#init coefs

linear_feats=['weight',
                      'disp',
                      'int']

other_feats = [i for i in xt.columns.values if i not in linear_feats]

epochs = 1500


mod = XGBRegressor(n_estimators=25,
                         max_depth=6,
                         random_state=42)

coefs = np.linalg.pinv(xt[linear_feats].T@xt[linear_feats])@(xt[linear_feats].T@yt)
#Ways to speed this up?? Gradient descent is actually fastest,intercept takes longest time

coefs_arr = []
RMSE_chain=[]
for e in range(epochs):

    mod.fit(xt[other_feats], yt-xt[linear_feats]@coefs)
    
    coefs = np.linalg.pinv(xt[linear_feats].T@xt[linear_feats])@(xt[linear_feats].T@(yt-mod.predict(xt[other_feats])))

    resid = yt - (xt[linear_feats]@coefs + mod.predict(xt[other_feats]))
    
    RMSE = np.mean(resid**2)**.5
    
    coefs_arr.append(list(coefs))
    RMSE_chain.append(RMSE)
    
    if e>epochs*.1:        
        converged = [check_converge(RMSE_chain)]#+[check_converge(c) for c in np.array(coefs_arr).T]
        
        if np.min(converged)==1:
            break
    
plt.plot(np.array(RMSE_chain))
plt.title('RMSE')
plt.show()

for i in range(len(linear_feats)):
    plt.plot(np.array(coefs_arr)[:,i])
    plt.title(linear_feats[i])
    plt.show()    
    print(check_converge(np.array(coefs_arr)[:,i]))





