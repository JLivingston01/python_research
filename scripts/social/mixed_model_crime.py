

import pandas as pd
import numpy as np
from random import sample
from xgboost import XGBRegressor
from random import choices
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.stats import t


dat = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00211/CommViolPredUnnormalizedData.txt",
        header=None)


f=open("c://users/jliv/downloads/violence_names.txt",)
all_lines = f.readlines()

f.close()

cols  = [i.split(":")[0].replace("-- ","") for i in all_lines if (len(i.split(":"))>1)]


kpis=[i for i in all_lines if (len(i.split(":"))>1)]
kpis=[i.split(":")[0].replace("-- ","") for i in kpis if "GOAL" in i.split(":")[1]]

informative = [i for i in all_lines if (len(i.split(":"))>1)]
informative=[i.split(":")[0].replace("-- ","") for i in informative if "not predictive" in i.split(":")[1]]

predictors=[i for i in cols if (i not in kpis+informative)]

dat.columns = cols

dat.replace("?",np.nan,inplace=True)

missing_list=[]
for i in cols:
    missing=len(dat[dat[i].isna()])
    if missing>0:
        missing_list.append([i,missing])


missing=pd.DataFrame(missing_list,columns = ['col','missing'])
droplist=list(missing[missing['missing']>=500]['col'])

dat.drop(droplist,axis=1,inplace=True)

for state in dat.state.unique():
    dat['STATE_'+state]=np.where(dat['state']==i,1,0)

dat.drop(['state'],axis=1,inplace=True)

predictors=[i for i in predictors if i in dat.columns]
kpis=[i for i in kpis if i in dat.columns]


dat=dat[predictors+kpis].dropna()

dat[predictors]=dat[predictors].astype(float)


train=sample(list(dat.index),int(len(dat.index)*.8))
train.sort()
test=[i for i in dat.index if i not in train]

kpi='murdPerPop'

X=dat[predictors].copy()
Y=dat[kpi]

xt=X[X.index.isin(train)].copy()
xv=X[X.index.isin(test)].copy()

yt=Y[X.index.isin(train)].copy()
yv=Y[X.index.isin(test)].copy()

m=np.mean(xt,axis=0)
s=np.std(xt,axis=0)


xt=(xt-m)/s
xv=(xv-m)/s

xt['int']=1
xv['int']=1

dat.corr()[kpi].sort_values()

fold = pd.Series(choices(range(1,11),k=len(X)),index=X.index)



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
        #self.coefs_=np.random.normal(0,.5,len(self.linear_feats))
        self.coefs_=np.zeros(len(self.linear_feats))
        
        for e in range(0,self.epoch):
            self.mod.fit(self.x[self.other_feats],self.y-self.x[self.linear_feats]@self.coefs_)
            grad=(self.y-self.x[self.linear_feats]@self.coefs_-self.mod.predict(self.x[self.other_feats]))@self.x[self.linear_feats]
        
            self.coefs_=self.coefs_+self.lr*grad
        
    def predict(self,x):
        
        pred=x[self.linear_feats]@self.coefs_+self.mod.predict(x[self.other_feats])
        
        return pred

mixed_r2s=[]
mixed_rmses=[]

xgb_r2s=[]
xgb_rmses=[]


linear_r2s=[]
linear_rmses=[]

for fo in fold.unique():
    
    xv=X[fold==fo][predictors].copy()
    yv=Y[fold==fo].copy()
    
    xt=X[fold!=fo][predictors].copy()
    yt=Y[fold!=fo].copy()
    
    m=np.mean(xt,axis=0)
    s=np.std(xt,axis=0)
    xt=(xt-m)/s
    xv=(xv-m)/s
    
    xt['int']=1
    xv['int']=1
    
        
    mod=mixed_model(
            mod=XGBRegressor(n_estimators=25,
                             random_state=42),
            lr=.01,
            epoch=50
            )
            
    
    mod.fit(xt,yt,
            linear_feats=['racepctblack',
                          'int']
            )
    
        
    
    #yfit=mod.predict(xt)
    ypred=mod.predict(xv)
    
    
    r2=1-sum((yv-ypred)**2)/sum((yv-np.mean(yv))**2)
      
    rmse=sum((yv-ypred)**2)**.5
    
    mixed_r2s.append(r2)
    mixed_rmses.append(rmse)
    
    
        
    xgb_mod=XGBRegressor(n_estimators=25,
                             random_state=42).fit(xt,yt)
    
    xgb_ypred=xgb_mod.predict(xv)
    
    xgb_r2=1-sum((yv-xgb_ypred)**2)/sum((yv-np.mean(yv))**2)
    
    xgb_rmse=sum((yv-xgb_ypred)**2)**.5

    xgb_r2s.append(xgb_r2)
    xgb_rmses.append(xgb_rmse)
    
    
    
    linear_coefs = np.linalg.pinv(xt.T@xt)@(xt.T@yt)
    linear_ypred=xv@linear_coefs
    
    linear_r2=1-sum((yv-linear_ypred)**2)/sum((yv-np.mean(yv))**2)
    
    linear_rmse=sum((yv-linear_ypred)**2)**.5


    linear_r2s.append(linear_r2)
    linear_rmses.append(linear_rmse)
    
    
    
method=np.median
print(method(mixed_r2s),
    method(xgb_r2s),
    method(linear_r2s)
    )

print(
    method(mixed_rmses),
    method(xgb_rmses),
    method(linear_rmses))



####


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
    
df.dropna(inplace=True)



train=sample(list(df.index),int(len(df.index)*.8))
train.sort()
test=[i for i in df.index if i not in train]



kpi='mpg'


feats=['cyl', 'disp', 'hp', 'weight', 'acc', 'yr', 'origin']

xt=df[df.index.isin(train)][feats].copy()
yt=df[df.index.isin(train)][kpi]


xv=df[df.index.isin(test)][feats].copy()
yv=df[df.index.isin(test)][kpi]

means=np.mean(xt)
stds=np.std(xt)

xt=(xt-means)/stds
xv=(xv-means)/stds

corrdf=xt.copy()
corrdf[kpi]=yt
corrdf.corr()[kpi]
corrdf.corr()

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
        #self.coefs_=np.random.normal(0,.5,len(self.linear_feats))
        self.coefs_=np.zeros(len(self.linear_feats))
        self.rmse_ = []
        self.coefs_per_epoch=[]
        
        for e in range(0,self.epoch):
            self.mod.fit(self.x[self.other_feats],self.y-self.x[self.linear_feats]@self.coefs_)
            resid = (self.y-self.x[self.linear_feats]@self.coefs_-self.mod.predict(self.x[self.other_feats]))
            grad=resid@self.x[self.linear_feats]
            self.rmse_.append(np.mean(resid**2)**.5)
            
        
            self.coefs_=self.coefs_+self.lr*grad
            self.coefs_per_epoch.append(list(self.coefs_))
            
            self.epochs_completed_=e
            
            self.converged_ = []
            
            if e>=self.epoch*.25:
                
                """
                Must run 1/4 of epochs.
                Stopping Criteria:
                    
                T-test of sample means for parameter estimates and model loss with:
                    X1: Third quarter of parameter chain
                    X2: Fourth quarter of parameter chain
                
                If all parameters and loss achieve convergence with 95% confidence:
                    Break
                If the final Epoch is reached without some parameters or loss converging:
                    Deliver warning to increase epoch parameter
                    
                """
                
                for i in range(len(self.coefs_.index)):
                    parameter_chain=np.array(self.coefs_per_epoch)[:,i]
                    
                    X1= parameter_chain[int(e*.5):int(e*.75)]
                    X2= parameter_chain[int(e*.75):]
                    
                    v=len(X1)+len(X2)-2
                    
                    T=(np.mean(X1)-np.mean(X2))/((np.var(X1)/len(X1)+np.var(X2)/len(X2))**.5)
                    
                    absT=abs(T)
                        
                    if absT<=t.ppf(1-.05/2, v):
                        self.converged_.append(1)
                    else:
                        self.converged_.append(0)
                        
                parameter_chain=self.rmse_
                
                X1= parameter_chain[int(e*.5):int(e*.75)]
                X2= parameter_chain[int(e*.75):]
                
                v=len(X1)+len(X2)-2
                
                T=(np.mean(X1)-np.mean(X2))/((np.var(X1)/len(X1)+np.var(X2)/len(X2))**.5)
                
                absT=abs(T)
            
                if absT<=t.ppf(1-.05/2, v):
                    self.converged_.append(1)
                else:
                    self.converged_.append(0)
            
                """
                if absT<=t.ppf(1-.05/2, v):
                    if np.mean(self.converged_)!=1:
                        print("Warning: Some parameters may not have converged, perhaps increase epochs.")
                    break"""
                
                #If all parameters converged, break; if last epoch is reached without convergence, produce warning
                if np.mean(self.converged_)==1:

                    break
                elif (np.mean(self.converged_)!=1)&(e==self.epoch):

                    print("Warning: Some parameters or Loss may not have converged, perhaps increase epochs.")
                    
        self.coef_means=pd.Series(np.mean(np.array(self.coefs_per_epoch)[int(self.epochs_completed_*.5):,],axis=0),index=self.coefs_.index)
        self.mod.fit(self.x[self.other_feats],self.y-self.x[self.linear_feats]@self.coef_means)
        
        
    def predict(self,x):
        
        #self.mod.fit(self.x[self.other_feats],self.y-self.x[self.linear_feats]@self.coef_means)
        pred=x[self.linear_feats]@self.coef_means+self.mod.predict(x[self.other_feats])
        
        return pred
    
    
    def predict_last_coefs(self,x):
        
        #self.mod.fit(self.x[self.other_feats],self.y-self.x[self.linear_feats]@self.coefs_)
        pred=x[self.linear_feats]@self.coefs_+self.mod.predict(x[self.other_feats])
        
        return pred

xt['int']=1
xv['int']=1

mod=mixed_model(
        mod=XGBRegressor(n_estimators=25,
                         max_depth=6,
                         random_state=42),
        lr=.1,
        epoch=500
        )
        

mod.fit(xt,yt,
        linear_feats=['weight',
                      'disp',
                      'int']
        )

#ypred=mod.predict_last_coefs(xv)
mod.coefs_
mod.coef_means


ypred=mod.predict(xv)

r2=1-sum((yv-ypred)**2)/sum((yv-np.mean(yv))**2)
  
rmse=np.mean((yv-ypred)**2)**.5


for i in range(len(mod.coefs_.index)):
    plt.plot(np.array(mod.coefs_per_epoch)[:,i])
    plt.title("Coefficient of "+mod.coefs_.index[i])
    plt.xlabel("Epoch")
    plt.show()
    
plt.plot(mod.rmse_)
plt.title("Training RMSE")
plt.xlabel("Epoch")
plt.show()
print("mixed model R2,RSME: ",round(r2,3),round(rmse,2))

gs = gridspec.GridSpec(2, 2)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])
ax4 = plt.subplot(gs[3])

ax1.plot(np.array(mod.coefs_per_epoch)[:,0])
ax1.set_title(mod.coefs_.index[0])
ax1.set_xticklabels([])
ax2.plot(np.array(mod.coefs_per_epoch)[:,1])
ax2.set_title(mod.coefs_.index[1])
ax2.set_xticklabels([])
ax3.plot(np.array(mod.coefs_per_epoch)[:,2])
ax3.set_title(mod.coefs_.index[2])
ax3.set_xlabel("Epoch")
ax4.plot(mod.rmse_)
ax4.set_title("RMSE")
ax4.set_xlabel("Epoch")
plt.show()



gs = gridspec.GridSpec(2, 2)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])
ax4 = plt.subplot(gs[3])

converged = int(mod.epochs_completed_*.5)

ax1.hist(np.array(mod.coefs_per_epoch)[converged:,0])
ax1.set_title(mod.coefs_.index[0])
ax2.hist(np.array(mod.coefs_per_epoch)[converged:,1])
ax2.set_title(mod.coefs_.index[1])
ax3.hist(np.array(mod.coefs_per_epoch)[converged:,2])
ax3.set_xlabel(mod.coefs_.index[2])
ax4.hist(mod.rmse_[converged:])
ax4.set_xlabel("RMSE")
plt.show()



#Testing Parameter Convergence

for i in range(len(mod.coefs_.index)):
    parameter_chain=np.array(mod.coefs_per_epoch)[:,i]
    column=mod.coefs_.index[i]
    X1= parameter_chain[int(mod.epochs_completed_*.5):int(mod.epochs_completed_*.75)]
    X2= parameter_chain[int(mod.epochs_completed_*.75):]
    
    v=len(X1)+len(X2)-2
    
    T=(np.mean(X1)-np.mean(X2))/((np.var(X1)/len(X1)+np.var(X2)/len(X2))**.5)
    
    absT=abs(T)
    
    print(column+" Converged: ",~(absT>t.ppf(1-.05/2, v)))


#Testing Parameter Convergence

parameter_chain=mod.rmse_

X1= parameter_chain[int(mod.epochs_completed_*.5):int(mod.epochs_completed_*.75)]
X2= parameter_chain[int(mod.epochs_completed_*.75):]

v=len(X1)+len(X2)-2

T=(np.mean(X1)-np.mean(X2))/((np.var(X1)/len(X1)+np.var(X2)/len(X2))**.5)

absT=abs(T)
print("RMSE Converged: ",~(absT>t.ppf(1-.05/2, v)))


##Regression

coef=np.linalg.pinv(xt.T@xt)@(xt.T@yt)

yfit_regression=xt@coef
ypred_regression=xv@coef

coef_df=pd.DataFrame(
        {'feat':xt.columns,
         'coef':coef}
        )

r2_regression=1-sum((yv-ypred_regression)**2)/sum((yv-np.mean(yv))**2)
  
rmse_regression=np.mean((yv-ypred_regression)**2)**.5
print("Regression R2,RSME: ",round(r2_regression,3),round(rmse_regression,2))

##XGB

xgb = mod.mod.fit(xt,yt)

ypred_xgb=pd.Series(xgb.predict(xv),index=yv.index)

r2_xgb=1-sum((yv-ypred_xgb)**2)/sum((yv-np.mean(yv))**2)
  
rmse_xgb=np.mean((yv-ypred_xgb)**2)**.5

print("XGB R2,RSME: ",round(r2_xgb,3),round(rmse_xgb,2))



