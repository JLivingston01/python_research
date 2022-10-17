
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import numpy as np
import pandas as pd

import random

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

dat = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv")

pd.set_option("display.max_columns",500)
dat.tail()

covars = ['age','anaemia','creatinine_phosphokinase',
          'diabetes','ejection_fraction','high_blood_pressure',
          'platelets','serum_creatinine','serum_sodium',
          'sex','smoking','time']

random.seed(42)
train_index = random.sample(list(dat.index),int(len(dat)*.8))

X=dat[covars].copy()
Y=dat['DEATH_EVENT']


xt=X[X.index.isin(train_index)].copy()
yt=Y[X.index.isin(train_index)].copy()

xv=X[~X.index.isin(train_index)].copy()
yv=Y[~X.index.isin(train_index)].copy()

m=np.mean(xt,axis=0)
s=np.std(xt,axis=0)

xt_s=(xt-m)/s

xv_s=(xv-m)/s

pca = PCA(n_components=2).fit(xt_s)

xt_pc=pd.DataFrame(pca.transform(xt_s),columns= ['pc1','pc2'],index=xt_s.index)
xv_pc=pd.DataFrame(pca.transform(xv_s),columns= ['pc1','pc2'],index=xv_s.index)



def make_model(mod,xt_s,yt,xv_s,yv,points):
    mod.fit(xt_s,yt)
    pred = pd.Series(mod.predict(xv_s),index=yv.index)
    
        
    points['pred']=mod.predict(points[['pc1','pc2']])
        
    plt.scatter(points[points['pred']==1]['pc1'],points[points['pred']==1]['pc2'])
    plt.scatter(points[points['pred']==0]['pc1'],points[points['pred']==0]['pc2'])
    plt.show()
    
    results = pd.DataFrame({'actual':yv,'pred':pred},index=yv.index)
    results['cnt']=1
    results = results.groupby(['actual','pred']).agg({'cnt':'sum'}).reset_index()
    results['proportion']=results['cnt']/sum(results['cnt'])
    print(type(mod).__name__,"\n", results)


plt.scatter(xt_pc['pc1'],xt_pc['pc2'])
plt.show()

mins = np.min(xt_pc,axis=0)
maxs = np.max(xt_pc,axis=0)

X1,Y1 = np.meshgrid(np.linspace(mins[0],maxs[0],num=50),
            np.linspace(mins[1],maxs[1],num=50))

index_set = []
for x in range(0,50):
    for y in range(0,50):
        
        index_set.append([np.linspace(mins[0],maxs[0],num=50)[x],
                          np.linspace(mins[1],maxs[1],num=50)[y]])

points = pd.DataFrame(index_set,columns=['pc1','pc2'])



for i in [DecisionTreeClassifier(),
          LinearSVC(),
          LogisticRegression(),
          KNeighborsClassifier(n_neighbors=5),
          RandomForestClassifier(),
          XGBClassifier(objective='binary:logistic', use_label_encoder=False)]:
    
        
    make_model(i,xt_pc,yt,xv_pc,yv,points)

mod = XGBClassifier(objective='binary:logistic', use_label_encoder=False)

mod.fit(xt_pc,yt)
pred = pd.Series(mod.predict(xv_pc),index=yv.index)

    
points['pred']=mod.predict(np.array(points[[0,1]]))
    
plt.scatter(points[points['pred']==1][0],points[points['pred']==1][1])
plt.scatter(points[points['pred']==0][0],points[points['pred']==0][1])
plt.show()


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

mod.fit(xt_pc,yt)    
        
pred = mod.predict(xv_pc)


results = pd.DataFrame({'actual':yv,'pred':pred},index=yv.index)
results['cnt']=1
results = results.groupby(['actual','pred']).agg({'cnt':'sum'}).reset_index()
results['proportion']=results['cnt']/sum(results['cnt'])
print(type(mod).__name__,"\n", results)







