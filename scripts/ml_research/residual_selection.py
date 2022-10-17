

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from random import sample,seed,choices




pd.set_option("display.max_columns",50)
dat=pd.read_csv("C:/Users/jliv/Downloads/AirQualityUCI/AirQualityUCI.csv",
                sep=";")

dat=dat[~dat['Date'].isna()].copy()

dat.drop([i for i in dat.columns if ('Unnamed' in i)|(i=='NMHC(GT)')],axis=1,inplace=True)


for i in dat.columns:
    if i not in ['Date','Time']:
        dat[i]=dat[i].astype(str).str.replace(",",".").astype(float)
        dat[i]=np.where(dat[i]<0,np.nan,dat[i])
        
        
"""
0 Date (DD/MM/YYYY)
1 Time (HH.MM.SS)
2 True hourly averaged concentration CO in mg/m^3 (reference analyzer)
3 PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted)
4 True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer)
5 True hourly averaged Benzene concentration in microg/m^3 (reference analyzer)
6 PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)
7 True hourly averaged NOx concentration in ppb (reference analyzer)
8 PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted)
9 True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)
10 PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)
11 PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted)
12 Temperature in Â°C
13 Relative Humidity (%)
14 AH Absolute Humidity
"""

for i in dat.columns:
    if i not in ['Date','Time']:
        plt.hist(dat[i],bins=30)
        plt.title(i)
        plt.show()

dat['Hour']=dat['Time'].str[:2]

dat['Date']=pd.to_datetime(dat['Date'], format='%d/%m/%Y')


dat['Month']=dat['Date'].dt.month
dat['DOW']=dat['Date'].dt.dayofweek

temporal_feats = ['Date','Time','Hour','Month','DOW']+\
    [i for i in dat.columns if ('Month' in i)|('Hour' in i)|('DOW' in i)]

for i in dat.columns:
    if i not in temporal_feats:
        plt.plot(dat.groupby(['Hour']).agg({i:'mean'}))
        plt.title(i)
        plt.show()
        

for i in dat.columns:
    if i not in temporal_feats:
        plt.plot(dat.groupby(['Month']).agg({i:'mean'}))
        plt.title(i)
        plt.show()
        

for i in dat.columns:
    if i not in temporal_feats:
        plt.plot(dat.groupby(['DOW']).agg({i:'mean'}))
        plt.title(i)
        plt.show()


for i in dat['Month'].unique():
    dat['Month_'+str(i)]=np.where(dat['Month']==i,1,0)
    
    
for i in dat['Hour'].unique():
    dat['Hour_'+str(i)]=np.where(dat['Hour']==i,1,0)
    
    
for i in dat['DOW'].unique():
    dat['DOW_'+str(i)]=np.where(dat['DOW']==i,1,0)



not_feats=['Date','Time','Hour','Month']
candidate_feats = [i for i in dat.columns.values if i not in not_feats]

kpis=['CO(GT)',
 'PT08.S1(CO)',
 'C6H6(GT)',
 'PT08.S2(NMHC)',
 'NOx(GT)',
 'PT08.S3(NOx)',
 'NO2(GT)',
 'PT08.S4(NO2)',
 'PT08.S5(O3)']

for i in kpis:
    print(i,len(dat[dat[i]<=0]),len(dat[dat[i].isna()]))
    dat['log_'+i]=np.log(dat[i])



seed(42)
train_ind=sample(list(dat.index),int(len(dat)*.8))
train_ind.sort()
test_ind=[i for i in list(dat.index) if i not in train_ind]

fold=choices(list(range(1,11)),k=len(train_ind))

dat['INT']=1

train=dat[dat.index.isin(train_ind)].copy()
test=dat[dat.index.isin(test_ind)].copy()

train['fold']=fold
train.columns

pd.set_option("display.max_rows",70)
train.dropna(axis = 0, how = 'any',inplace=True)

test.dropna(axis = 0, how = 'any',inplace=True)

#,'log_PT08.S2(NMHC)'
feats = ['INT']+\
    [i for i in train.columns if ('Month_' in i)&(i!='Month_6')]+\
    [i for i in train.columns if ('Hour_' in i)&(i!='Hour_3')]+\
    [i for i in train.columns if ('DOW_' in i)&(i!='DOW_3')]

kpi = 'log_CO(GT)'

train2=train[~train[kpi].isna()].copy()
test2=test[~test[kpi].isna()].copy()

xt_init=train2[feats].copy()
yt_init=train2[kpi]



xv_init=test2[feats].copy()
yv_init=test2[kpi]




coefs = np.linalg.pinv(xt_init.T@xt_init)@(xt_init.T@yt_init)

yfit=xt_init@coefs
ypred=xv_init@coefs


print("R-squared: ",1-sum((ypred-yv_init)**2)/sum((yv_init-np.mean(yv_init))**2))


abs(train2.corr()[kpi]).sort_values(ascending=False)

#Forward algorithm for cross val resids
def linear_model(x,y,xtest,ytest):
    
    coefs = np.linalg.pinv(x.T@x)@(x.T@y)
        
    ypred=pd.Series(xtest@coefs,index=ytest.index)
    
    return ypred

    
def forward_select(learner,data,must_include,candidates,kpi,folds,max_features=100,set_seed=None):
    
    selected=", ".join(must_include)
    new=[]
    
    r2_init = -1
    improvements = [1e-3,1e-3,1e-3,1e-3,1e-3]
    
    if set_seed is not None:
        seed(set_seed)
    fold=choices(list(range(1,folds+1)),k=len(data))
    data['fold']=fold
    
    
    r=0
    while r < max_features:
        val_r2=[]
        
        resids=pd.Series()
        
        must_include=must_include+new
        
        new=[]
        
        for i in range(1,folds+1):
            
            xv=data[data['fold']==i][must_include].copy()
            xt=data[data['fold']!=i][must_include].copy()
            
            
            yv=data[data['fold']==i][kpi].copy()
            yt=data[data['fold']!=i][kpi].copy()
            
            
            #coefs = np.linalg.pinv(xt.T@xt)@(xt.T@yt)
        
            ypred=learner(x=xt,y=yt,xtest=xv,ytest=yv)
        
            r2=1-sum((ypred-yv)**2)/sum((yv-np.mean(yv))**2)
            
            resid=pd.Series(ypred-yv,index=yv.index)
            resids=resids.append(resid)
            
            val_r2.append(r2)
            
        print(r,selected,np.mean(val_r2))
        
        corrdf=train2[must_include+candidates].copy()
        corrdf['resid']=resids
        
        correlations = abs(corrdf.corr()['resid']).sort_values(ascending=False)
        
        selected=correlations[correlations.index!='resid'].head(1).index[0]
        
        new=new+[selected]
        
        candidates=[keep for keep in candidates if keep not in must_include+new]
    
        r+=1
        improvement=r2-r2_init
        r2_init=r2
        
        improvements.append(improvement)
        
        if np.mean(improvements[len(improvements)-5:])<0:
            break
        
    
        if len(candidates)==0:
            break
        if selected in must_include:
            break
    
    return(must_include)

m1=['INT']

c1 = [i for i in xt_init.columns if i not in m1]

must_include=forward_select(learner=linear_model,
                            data=train2,
                            must_include=m1,
                            candidates=c1,
                            kpi=kpi,
                            folds=10,
                            max_features=100,
                            set_seed=42)


xt=train2[must_include].copy()
yt=train2[kpi]

xv=test2[must_include].copy()
yv=test2[kpi]

coefs = np.linalg.pinv(xt.T@xt)@(xt.T@yt)

yfit=xt@coefs
ypred=xv@coefs


print("R-squared: ",1-sum((ypred-yv)**2)/sum((yv-np.mean(yv))**2))


#Use with other algorithms, a similar test Rsquared is achieved with few features vs a model of many features

from sklearn.ensemble import RandomForestRegressor


def tree_model(x,y,xtest,ytest):
    
    mod = RandomForestRegressor(n_estimators=50,
                                max_depth=6).fit(x,y)
    
    ypred=pd.Series(mod.predict(xtest),index=ytest.index)
    
    return ypred
    
    
m1=['INT']

c1 = [i for i in xt_init.columns if i not in m1]

must_include=forward_select(learner=tree_model,
                            data=train2,
                            must_include=m1,
                            candidates=c1,
                            kpi=kpi,
                            folds=10,
                            max_features=100,
                            set_seed=42)


xt=train2[must_include].copy()
yt=train2[kpi]

xv=test2[must_include].copy()
yv=test2[kpi]

ypred=tree_model(xt,yt,xv,yv)


print("R-squared: ",1-sum((ypred-yv)**2)/sum((yv-np.mean(yv))**2))

    
    
xt=train2[feats].copy()
yt=train2[kpi]

xv=test2[feats].copy()
yv=test2[kpi]

ypred=tree_model(xt,yt,xv,yv)


print("R-squared: ",1-sum((ypred-yv)**2)/sum((yv-np.mean(yv))**2))

    

    
    
    
    
    
#Use with other algorithms, a similar test Rsquared is achieved with few features vs a model of many features

import xgboost
xgboost.XGBRegressor

def tree_model(x,y,xtest,ytest):
    
    mod = xgboost.XGBRegressor(n_estimators=50,
                                max_depth=None).fit(x,y)
    
    ypred=pd.Series(mod.predict(xtest),index=ytest.index)
    
    return ypred
    
    
m1=['INT']

c1 = [i for i in xt_init.columns if i not in m1]

must_include=forward_select(learner=tree_model,
                            data=train2,
                            must_include=m1,
                            candidates=c1,
                            kpi=kpi,
                            folds=10,
                            max_features=100,
                            set_seed=42)


xt=train2[must_include].copy()
yt=train2[kpi]

xv=test2[must_include].copy()
yv=test2[kpi]

ypred=tree_model(xt,yt,xv,yv)


print("R-squared: ",1-sum((ypred-yv)**2)/sum((yv-np.mean(yv))**2))

    
    
xt=train2[feats].copy()
yt=train2[kpi]

xv=test2[feats].copy()
yv=test2[kpi]

ypred=tree_model(xt,yt,xv,yv)


print("R-squared: ",1-sum((ypred-yv)**2)/sum((yv-np.mean(yv))**2))

    
    
    
    
    
    
    
    