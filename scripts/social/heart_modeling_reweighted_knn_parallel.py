
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load Data, Fill NA with column means and Standardize
dat = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",header=None,sep="\t")[0]

X=pd.DataFrame(np.array(list(dat.str.split())))

for i in X.columns.values:
    X[i]=X[i].replace("?",np.nan).astype(float)
    X[i].fillna(np.nanmean(X[i]),inplace=True)
    
Y=np.array(X[0])
X=X[[i for i in X.columns.values if i>0]]


X=(X-np.mean(X))/np.std(X)
X['int']=1
X=np.array(X)

#Initialize 600 chains of 30,000 iterations to evaluate 8 model coefficients, with regressions scored by the R-squared.
draws = 600
w0=np.random.normal(0,.01,(X.shape[1],draws))

score0=1-np.sum((Y.reshape(len(Y),1)-X@w0)**2,axis=0)/sum((Y-np.mean(Y))**2)

delta=np.zeros((X.shape[1],draws))
stepsize=.0001

updates = 0
while updates < 30000:
    w1=w0+np.random.normal(delta,stepsize)
    score1=1-np.sum((Y.reshape(len(Y),1)-X@w1)**2,axis=0)/sum((Y-np.mean(Y))**2)
    
    delta = np.where(score1>score0,w1-w0,delta)
    w0=np.where(score1>score0,w1,w0)
    print(sum(np.where(score1>score0,1,0)))
    score0=score1
    updates+=1


coef_est=np.round(np.mean(w0,axis=1),2)
print(coef_est)



coef_actual=np.round(np.linalg.inv(X.T@X)@(X.T@Y),2)
print(coef_actual)


def dist_plot(i, fontsize=12):
    plt.hist(w0[i],bins=30,label='est. distribution')
    plt.bar(coef_actual[i],100,color='.1',width=1,alpha=.5,label='true coef')
    plt.ylim((0,60))
    plt.legend()
    plt.title("feat_"+str(i)+"; Mean: "+str(coef_est[i])+", Exact: "+str(coef_actual[i]))
    
    
fig = plt.figure(figsize=(10,15))
ax1 = fig.add_subplot(4,2,1)
ax1 = dist_plot(0, fontsize=12)
ax2 = fig.add_subplot(4,2,2)
ax2 = dist_plot(1, fontsize=12)
ax3 = fig.add_subplot(4,2,3)
ax3 = dist_plot(2, fontsize=12)
ax4 = fig.add_subplot(4,2,4)
ax4 = dist_plot(3, fontsize=12)
ax5 = fig.add_subplot(4,2,5)
ax5 = dist_plot(4, fontsize=12)
ax6 = fig.add_subplot(4,2,6)
ax6 = dist_plot(5, fontsize=12)
ax7 = fig.add_subplot(4,2,7)
ax7 = dist_plot(6, fontsize=12)
ax8 = fig.add_subplot(4,2,8)
ax8 = dist_plot(7, fontsize=12)
plt.show()


#KNN
import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier

dat = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv")

pd.set_option("display.max_columns",500)
dat.tail()

covars = ['age','anaemia','creatinine_phosphokinase',
          'diabetes','ejection_fraction','high_blood_pressure',
          'platelets','serum_creatinine','serum_sodium',
          'sex','smoking','time']


X=dat[covars].copy()
Y=dat['DEATH_EVENT']


X=(X-np.mean(X,axis=0))/np.std(X,axis=0)


random.seed(42)
index = np.array(random.choices([1,2,3,4,5,6],k=len(X)))

draws = 100
delta=np.zeros((X.shape[1],draws))
stepsize=.1
w0=np.random.normal(1,.05,(X.shape[1],draws))

score0list=[]
for fold in [1,2,3,4,5]:
    xv = X[index==fold].copy()
    yv = Y[index==fold].copy()
    
    xt = X[~pd.Series(index).isin([fold,6])].copy()
    yt = Y[~pd.Series(index).isin([fold,6])].copy()
    
    
    xtl=np.array([np.array(xt)*w0[:,i] for i in range(draws)])
    xvl=np.array([np.array(xv)*w0[:,i] for i in range(draws)])
    
    
    knn=np.array([KNeighborsClassifier().fit(xtl[i],yt).predict(xvl[i]) for i in range(draws)])
    
    score0 = np.mean(np.where(knn[:,np.array(yv)==1]==np.array(yv)[np.array(yv)==1],1,0),axis=1)

    
    score0list.append(score0)

score0=np.mean(np.array(score0list),axis=0)

#np.mean(np.where(knn[:,np.array(yv)==1]==np.array(yv)[np.array(yv)==1],1,0),axis=1)


updates = 0
while updates<300:
    w1 = w0+np.random.normal(delta,stepsize)
    
    
    score1list=[]
    for fold in [1,2,3,4,5]:
        xv = X[index==fold].copy()
        yv = Y[index==fold].copy()
        
        xt = X[~pd.Series(index).isin([fold,6])].copy()
        yt = Y[~pd.Series(index).isin([fold,6])].copy()
        
        
        xtl=np.array([np.array(xt)*w1[:,i] for i in range(draws)])
        xvl=np.array([np.array(xv)*w1[:,i] for i in range(draws)])
        
        
        knn=np.array([KNeighborsClassifier().fit(xtl[i],yt).predict(xvl[i]) for i in range(draws)])
        
        score1 = np.mean(np.where(knn[:,np.array(yv)==1]==np.array(yv)[np.array(yv)==1],1,0),axis=1)
        
        score1list.append(score1)
    
    score1=np.mean(np.array(score1list),axis=0)
    
    delta = np.where(score1>score0,w1-w0,delta)
    w0=np.where(score1>score0,w1,w0)
    print(sum(np.where(score1>score0,1,0)))
    score0=score1
    updates+=1
   


np.mean(w0,axis=1)


xv = X[index==6].copy()
yv = Y[index==6].copy()

xt = X[pd.Series(index).isin([1,2,3,4,5])].copy()
yt = Y[pd.Series(index).isin([1,2,3,4,5])].copy()



xtf=np.array(xt)*np.mean(w0,axis=1)
xvf=np.array(xv)*np.mean(w0,axis=1)

knn=KNeighborsClassifier().fit(xtf,yt).predict(xvf) 

acc1 = np.mean(np.where(knn==np.array(yv),1,0))
rec1 = np.mean(np.where(knn[np.array(yv)==1]==np.array(yv)[np.array(yv)==1],1,0))
prec1 = np.mean(np.where(knn[knn==1]==np.array(yv)[knn==1],1,0))




knn=KNeighborsClassifier().fit(xt,yt).predict(xv) 

acc = np.mean(np.where(knn==np.array(yv),1,0))
rec = np.mean(np.where(knn[np.array(yv)==1]==np.array(yv)[np.array(yv)==1],1,0))
prec= np.mean(np.where(knn[knn==1]==np.array(yv)[knn==1],1,0))


#### Class Development

import numpy as np
import random
import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier


from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


dat = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv")

pd.set_option("display.max_columns",500)
dat.tail()

covars = ['age','anaemia','creatinine_phosphokinase',
          'diabetes','ejection_fraction','high_blood_pressure',
          'platelets','serum_creatinine','serum_sodium',
          'sex','smoking','time']


X=dat[covars].copy()
Y=dat['DEATH_EVENT']

X=(X-np.mean(X,axis=0))/np.std(X,axis=0)
"""
random.seed(42)
index = pd.Series(np.array(random.choices([1,2,3,4,5,6],k=len(X))),index=X.index)
"""

random.seed(42)
train_index = random.sample(list(X.index),int(len(X)*.8))

xtest = X[~X.index.isin(train_index)].copy()
ytest = Y[~Y.index.isin(train_index)].copy()

X = X[X.index.isin(train_index)].copy()
Y = Y[Y.index.isin(train_index)].copy()

"""
xtest=X[index==6].copy()
ytest=Y[index==6].copy()

X=X[index!=6].copy()
Y=Y[index!=6].copy()
"""

class model_cv:
    
    def __init__(self,mod,draws=100,stepsize=.1,folds=5,epochs=300):
        
        self.draws=draws
        self.stepsize=stepsize
        self.folds=folds
        self.epochs = epochs
        self.mod=mod
        
    def fit(self,X,Y):
        
        self.weights=np.random.normal(1,.05,(X.shape[1],self.draws))
        
        index = pd.Series(np.array(random.choices(list(range(1,self.folds+1)),k=len(X))),index=X.index)
        
        score0list=[]
        for fold in range(1,self.folds+1):
            xv = X[index==fold].copy()
            yv = Y[index==fold].copy()
            
            xt = X[index!=fold].copy()
            yt = Y[index!=fold].copy()
            
            
            xtl=np.array([np.array(xt)*self.weights[:,i] for i in range(self.draws)])
            xvl=np.array([np.array(xv)*self.weights[:,i] for i in range(self.draws)])
            
            
            knn=np.array([self.mod.fit(xtl[i],yt).predict(xvl[i]) for i in range(self.draws)])
            
            score0 = np.mean(np.where(knn[:,np.array(yv)==1]==np.array(yv)[np.array(yv)==1],1,0),axis=1)
        
            
            score0list.append(score0)
        
        score0=np.mean(np.array(score0list),axis=0)
                
        
        delta=np.zeros((X.shape[1],self.draws))
        updates = 0
        while updates<self.epochs:
            w1 = self.weights+np.random.normal(delta,self.stepsize)
            
            
            score1list=[]
            for fold in range(1,self.folds+1):
                xv = X[index==fold].copy()
                yv = Y[index==fold].copy()
                
                xt = X[index!=fold].copy()
                yt = Y[index!=fold].copy()
                
                
                xtl=np.array([np.array(xt)*w1[:,i] for i in range(self.draws)])
                xvl=np.array([np.array(xv)*w1[:,i] for i in range(self.draws)])
                
                
                knn=np.array([self.mod.fit(xtl[i],yt).predict(xvl[i]) for i in range(self.draws)])
                
                score1 = np.mean(np.where(knn[:,np.array(yv)==1]==np.array(yv)[np.array(yv)==1],1,0),axis=1)
                
                score1list.append(score1)
            
            score1=np.mean(np.array(score1list),axis=0)
            
            delta = np.where(score1>score0,w1-self.weights,delta)
            self.weights=np.where(score1>score0,w1,self.weights)
            print(sum(np.where(score1>score0,1,0)))
            score0=score1
            updates+=1
        
        self.mod.fit(X*np.mean(self.weights,axis=1),Y)
            
    def predict(self,X):

        pred = self.mod.predict(X*np.mean(self.weights,axis=1))
        
        return pred
            

mod = model_cv(mod = KNeighborsClassifier(n_neighbors=5)
               ,
               epochs=300)

mod.fit(X,Y)    
        
pred = mod.predict(xtest)


results = pd.DataFrame({'actual':ytest,'pred':pred},index=ytest.index)
results['cnt']=1
results = results.groupby(['actual','pred']).agg({'cnt':'sum'}).reset_index()
results['proportion']=results['cnt']/sum(results['cnt'])
print(type(mod).__name__,"\n", results)



def make_model(mod,xtrain,ytrain,xtest,ytest):
    mod.fit(xtrain,ytrain)
    pred = pd.Series(mod.predict(xtest),index=ytest.index)
    
    results = pd.DataFrame({'actual':ytest,'pred':pred},index=ytest.index)
    results['cnt']=1
    results = results.groupby(['actual','pred']).agg({'cnt':'sum'}).reset_index()
    results['proportion']=results['cnt']/sum(results['cnt'])
    print(type(mod).__name__,"\n", results)

for i in [
          KNeighborsClassifier(n_neighbors=k) for k in range(2,10)]+\
          [DecisionTreeClassifier(min_samples_leaf=2),
          LinearSVC(),
          LogisticRegression(),
          RandomForestClassifier(),
          XGBClassifier(),
          ]:
    
        
    make_model(i,X,Y,xtest,ytest)


