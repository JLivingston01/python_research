

import pandas as pd
import numpy as np
from random import sample
from xgboost import XGBRegressor
from random import choices,seed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.stats import t


import os

os.chdir("c://users/jliv/downloads/")

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


class mixed_model:
    
    def __init__(self,mod,lr,epoch,optimization):

        self.lr=lr
        self.epoch=epoch
        self.mod=mod
        self.optimization=optimization

        
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
            
            if self.optimization =='Newtonian':
                H = np.linalg.pinv(self.x[self.linear_feats].T@self.x[self.linear_feats])
                term = grad@H
            else:
                term = grad
                self.coefs_=self.coefs_+self.lr*grad
            
            self.coefs_=self.coefs_+self.lr*term
            
            self.coefs_per_epoch.append(list(self.coefs_))
            
            self.epochs_completed_=e
            
            self.converged_ = []
            
            #if e>=80:
            if e >= self.epoch*.1:  
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
                
                for i in range(len(self.linear_feats)):
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
                elif (np.mean(self.converged_)!=1)&(e==self.epoch-1):

                    print("Warning: Some parameters or Loss may not have converged, perhaps increase epochs.")
                    
        self.coef_means=pd.Series(np.mean(np.array(self.coefs_per_epoch)[int(self.epochs_completed_*.5):,],axis=0),index=self.linear_feats)
        self.mod.fit(self.x[self.other_feats],self.y-self.x[self.linear_feats]@self.coef_means)
        
        
    def predict(self,x):
        
        #self.mod.fit(self.x[self.other_feats],self.y-self.x[self.linear_feats]@self.coef_means)
        pred=x[self.linear_feats]@self.coef_means+self.mod.predict(x[self.other_feats])
        
        return pred
    
    
    def predict_last_coefs(self,x):
        
        #self.mod.fit(self.x[self.other_feats],self.y-self.x[self.linear_feats]@self.coefs_)
        pred=x[self.linear_feats]@self.coefs_+self.mod.predict(x[self.other_feats])
        
        return pred

X['int']=1
xtest['int']=1

rmse_PR=[]
rmse_REG=[]
rmse_XGB=[]
r2_PR=[]
r2_REG=[]
r2_XGB=[]

for f in range(1,9):
        
    xt=X[fold!=f].copy()
    yt=Y[fold!=f]
    
    
    xv=X[fold==f].copy()
    yv=Y[fold==f]
            
    mod=mixed_model(
            mod=XGBRegressor(n_estimators=25,
                             max_depth=6,
                             random_state=42),
            lr=.1,
            epoch=500,
            optimization='Gradient'
            )
            
    
    mod.fit(xt,yt,
            linear_feats=['weight',
                          'disp',
                          'int']
            )
    
    
    ypred=mod.predict(xv)
    
    r2=1-sum((yv-ypred)**2)/sum((yv-np.mean(yv))**2)
      
    rmse=np.mean((yv-ypred)**2)**.5
    
    rmse_PR.append(rmse)
    r2_PR.append(r2)
    print("mixed model R2,RSME: ",round(r2,3),round(rmse,2))
    
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
    
    rmse_REG.append(rmse_regression)
    r2_REG.append(r2_regression)
    ##XGB
    
    xgb = mod.mod.fit(xt,yt)
    
    ypred_xgb=pd.Series(xgb.predict(xv),index=yv.index)
    
    r2_xgb=1-sum((yv-ypred_xgb)**2)/sum((yv-np.mean(yv))**2)
      
    rmse_xgb=np.mean((yv-ypred_xgb)**2)**.5
    
    
    rmse_XGB.append(rmse_xgb)
    r2_XGB.append(r2_xgb)
    

cv_out=pd.DataFrame({'fold':range(1,9)})
cv_out['rmse_PR']=rmse_PR
cv_out['rmse_REG']=rmse_REG
cv_out['rmse_XGB']=rmse_XGB
cv_out['r2_PR']=r2_PR
cv_out['r2_REG']=r2_REG
cv_out['r2_XGB']=r2_XGB

print(np.round(np.mean(cv_out,axis=0),3))


#TEST



kpi='mpg'


feats=['cyl', 'disp', 'hp', 'weight', 'acc', 'yr', 'origin']

xt=X.copy()
yt=Y.copy()

xv=xtest.copy()
yv=ytest.copy()

"""
mod=mixed_model(
        mod=XGBRegressor(n_estimators=25,
                         max_depth=6,
                         random_state=42),
        lr=10,
        epoch=2800,
        optimization='Newtonian'
        )
   """     

mod=mixed_model(
        mod=XGBRegressor(n_estimators=25,
                         max_depth=6,
                         random_state=42),
        lr=.1,
        epoch=1500,
        optimization='Gradient'
        )

mod.fit(xt,yt,
        linear_feats=['weight',
                      'disp',
                      'int']
        )


#ypred=mod.predict_last_coefs(xv)
mod.coefs_
mod.converged_
mod.coefs_per_epoch
mod.epochs_completed_

#pd.DataFrame(round(mod.coef_means,2),columns=['Coefficient']).to_csv("downloads/pr_coef.csv")

ypred=mod.predict(xv)

r2=1-sum((yv-ypred)**2)/sum((yv-np.mean(yv))**2)
  
rmse=np.mean((yv-ypred)**2)**.5


for i in range(len(mod.linear_feats)):
    plt.plot(np.array(mod.coefs_per_epoch)[:,i])
    plt.title("Coefficient of "+mod.linear_feats[i])
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
ax1.set_title(mod.linear_feats[0])
ax1.set_xticklabels([])
ax2.plot(np.array(mod.coefs_per_epoch)[:,1])
ax2.set_title(mod.linear_feats[1])
ax2.set_xticklabels([])
ax3.plot(np.array(mod.coefs_per_epoch)[:,2])
ax3.set_title(mod.linear_feats[2])
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
ax1.set_title(mod.linear_feats[0])
ax2.hist(np.array(mod.coefs_per_epoch)[converged:,1])
ax2.set_title(mod.linear_feats[1])
ax3.hist(np.array(mod.coefs_per_epoch)[converged:,2])
ax3.set_xlabel(mod.linear_feats[2])
ax4.hist(mod.rmse_[converged:])
ax4.set_xlabel("RMSE")
plt.show()



#Testing Parameter Convergence

for i in range(len(mod.linear_feats)):
    parameter_chain=np.array(mod.coefs_per_epoch)[:,i]
    column=mod.linear_feats[i]
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

round(coef_df,2).to_csv("coef_df.csv")
r2_regression=1-sum((yv-ypred_regression)**2)/sum((yv-np.mean(yv))**2)
  
rmse_regression=np.mean((yv-ypred_regression)**2)**.5
print("Regression R2,RSME: ",round(r2_regression,3),round(rmse_regression,2))



##XGB

xgb = mod.mod.fit(xt,yt)

ypred_xgb=pd.Series(xgb.predict(xv),index=yv.index)

r2_xgb=1-sum((yv-ypred_xgb)**2)/sum((yv-np.mean(yv))**2)
  
rmse_xgb=np.mean((yv-ypred_xgb)**2)**.5

print("XGB R2,RSME: ",round(r2_xgb,3),round(rmse_xgb,2))


"""
Testing high multicollinearity behavior. XGB is robust to this, regression is not.

Two features highly correlated are CYL and DISP. 

"""
print('Looking at multicollinearity:')
#Multicollinearity

mod=mixed_model(
        mod=XGBRegressor(n_estimators=25,
                         max_depth=6,
                         random_state=42),
        lr=3,
        epoch=3000,
        optimization='Newtonian'
        )

mod.fit(xt,yt,
        linear_feats=['weight',
                      'disp',
                      'cyl','yr',
                      'int']
        )


ypred=mod.predict(xv)

r2=1-sum((yv-ypred)**2)/sum((yv-np.mean(yv))**2)
  
rmse=np.mean((yv-ypred)**2)**.5


for i in range(len(mod.linear_feats)):
    plt.plot(np.array(mod.coefs_per_epoch)[:,i])
    plt.title("Coefficient of "+mod.linear_feats[i])
    plt.xlabel("Epoch")
    plt.show()
    
plt.plot(mod.rmse_)
plt.title("Training RMSE")
plt.xlabel("Epoch")
plt.show()
print("mixed model R2,RSME: ",round(r2,3),round(rmse,2))


mm_pr_coefs = pd.DataFrame(mod.coef_means,columns=['Coef'])
round(mm_pr_coefs.merge(round(corrdf.corr()[[kpi]],2),how='left',left_index=True,right_index=True),2).to_csv('multicollinear_pr_coefs.csv')

round(pd.DataFrame(mod.coef_means,columns=['Coef']),2).to_csv('multicollinear_pr_coefs.csv')


round(corrdf.corr()[kpi],2)
corrdf.corr()

##Regression


xt1=xt[['weight','disp','cyl','yr','int']].copy()
xv1=xv[['weight','disp','cyl','yr','int']].copy()
coef=np.linalg.pinv(xt1.T@xt1)@(xt1.T@yt)

yfit_regression=xt1@coef
ypred_regression=xv1@coef

coef_df=pd.DataFrame(
        {'feat':xt1.columns,
         'coef':coef}
        )
coef_df.set_index('feat',inplace=True)

round(coef_df.merge(round(corrdf.corr()[[kpi]],2),how='left',left_index=True,right_index=True),2).to_csv('collinear_coefs_reg.csv')



r2_regression=1-sum((yv-ypred_regression)**2)/sum((yv-np.mean(yv))**2)
  
rmse_regression=np.mean((yv-ypred_regression)**2)**.5
print("Regression R2,RSME: ",round(r2_regression,3),round(rmse_regression,2))


#Regression Embedded
from sklearn.linear_model import LinearRegression
mod=mixed_model(
        mod=LinearRegression(fit_intercept=False),
        lr=1,
        epoch=500,
        optimization='Newtonian'
        )

mod.fit(xt,yt,
        linear_feats=['weight',
                      'disp',
                      'cyl','yr',
                      'int']
        )


ypred=mod.predict(xv)

r2=1-sum((yv-ypred)**2)/sum((yv-np.mean(yv))**2)
  
rmse=np.mean((yv-ypred)**2)**.5


for i in range(len(mod.linear_feats)):
    plt.plot(np.array(mod.coefs_per_epoch)[:,i])
    plt.title("Coefficient of "+mod.linear_feats[i])
    plt.xlabel("Epoch")
    plt.show()
    
plt.plot(mod.rmse_)
plt.title("Training RMSE")
plt.xlabel("Epoch")
plt.show()
print("mixed model R2,RSME: ",round(r2,3),round(rmse,2))


mm_pr_with_regression_coefs = pd.DataFrame(mod.coef_means,columns=['Coef'])
round(mm_pr_with_regression_coefs.merge(round(corrdf.corr()[[kpi]],2),how='left',left_index=True,right_index=True),2).to_csv("multicollinear_pr_with_regression.csv")




#Learner too-strong
"""
If learner is too strong, linear model may learn nothing: coefficients nearly to zero.
"""

print("studying strength of learner:")
mod=mixed_model(
        mod=XGBRegressor(n_estimators=100,
                         max_depth=6,
                         random_state=42),
        lr=1,
        epoch=2500,
        optimization='Newtonian'
        )

mod.fit(xt,yt,
        linear_feats=['weight',
                      'disp',
                      'int']
        )



ypred=mod.predict(xv)

r2=1-sum((yv-ypred)**2)/sum((yv-np.mean(yv))**2)
  
rmse=np.mean((yv-ypred)**2)**.5


for i in range(len(mod.linear_feats)):
    plt.plot(np.array(mod.coefs_per_epoch)[:,i])
    plt.title("Coefficient of "+mod.linear_feats[i])
    plt.xlabel("Epoch")
    plt.show()
    
plt.plot(mod.rmse_)
plt.title("Training RMSE")
plt.xlabel("Epoch")
plt.show()
print("mixed model R2,RSME: ",round(r2,3),round(rmse,2))
