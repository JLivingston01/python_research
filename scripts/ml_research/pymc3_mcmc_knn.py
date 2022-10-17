


import numpy as np
import pymc3 as pm 
import pandas as pd
"""
This monte carlo algorithm aproximates the "true" value of the interesting 
parameter/s using a random walk of normally distributed steps with mean 0 or
a mean of the last accepted step in the walk for the parameter. 
"""
truth=5


with pm.Model() as model:
    #hyper priors
    #boundednormal = pm.Bound(pm.Normal, lower = 0.5, upper = .95)
    boundednormal2 = pm.Bound(pm.Normal, lower = .0000001)
    
    
    #offline_media_mu = pm.Normal('w_mean',mu = 2, sd = 1)
    #offline_media_sigma = boundednormal('w_sig',mu = 5, sd = 3)
    
    #Priors
    w = pm.Normal('w_', mu = 1, sd = 10)
    
    #Error
    eps = boundednormal2('eps',mu = .0001, sd = .5)
    
    
    #Expected
    
    Est = w
    
    #Likelihood
    like = pm.Normal('like',mu = Est, sd = eps, observed = truth)

     
   
with model:
    trace = pm.sample(draws = 2000, tune = 2000, init = None,step=pm.Metropolis(), progressbar = True)
  
pm.traceplot(trace)


summary = pm.summary(trace)





import random
import matplotlib.pyplot as plt
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


acc = []

for i in list(range(2,30)):
    avgscore=[]
    for t in [1,2,3,4,5]:
                
        xv = X[index==t].copy()
        yv = Y[index==t].copy()
        
        xt = X[~pd.Series(index).isin([t,6])].copy()
        yt = Y[~pd.Series(index).isin([t,6])].copy()


        knn = KNeighborsClassifier(n_neighbors=i, 
                                       weights='distance', 
                                       algorithm='auto', leaf_size=30, p=2, 
                                       metric='euclidean', metric_params=None, 
                                       n_jobs=None)
        
        knn.fit(xt,yt)
        
        tp=sum(np.where((knn.predict(xv)==1)&(yv==1),1,0))
        fp=sum(np.where((knn.predict(xv)==1)&(yv==0),1,0))
        tn=sum(np.where((knn.predict(xv)==0)&(yv==0),1,0))
        fn=sum(np.where((knn.predict(xv)==0)&(yv==1),1,0))
        
        precision=tp/(tp+fp)
        recall=tp/(tp+fn)
        
        #score = (sum(np.where(knn.predict(xv*w0)==yv,1,0)))/(len(yv))
        score = recall

        avgscore.append(score)
    acc.append(np.mean(avgscore))

plt.plot(acc)
plt.xticks(list(range(28)),list(range(2,30)))
plt.show()


k=5


X


with pm.Model() as model:
    #hyper priors
    #boundednormal = pm.Bound(pm.Normal, lower = 0.5, upper = .95)
    boundednormal2 = pm.Bound(pm.Normal, lower = .0000001)
    
    
    #offline_media_mu = pm.Normal('w_mean',mu = 2, sd = 1)
    #offline_media_sigma = boundednormal('w_sig',mu = 5, sd = 3)
    
    #Priors
    
    
    w_age=pm.Normal('w_age', mu = 1, sd = 10)
    w_anaemia=pm.Normal('w_anaemia', mu = 1, sd = 10)
    w_creatinine_phosphokinase=pm.Normal('w_creatinine_phosphokinase', mu = 1, sd = 10)
    w_diabetes=pm.Normal('w_diabetes', mu = 1, sd = 10)
    w_ejection_fraction=pm.Normal('w_ejection_fraction', mu = 1, sd = 10)
    w_high_blood_pressure=pm.Normal('w_high_blood_pressure', mu = 1, sd = 10)
    w_platelets=pm.Normal('w_platelets', mu = 1, sd = 10)
    w_serum_creatinine=pm.Normal('w_serum_creatinine', mu = 1, sd = 10)
    w_serum_sodium=pm.Normal('w_serum_sodium', mu = 1, sd = 10)
    w_sex=pm.Normal('w_sex', mu = 1, sd = 10)
    w_smoking=pm.Normal('w_smoking', mu = 1, sd = 10)
    w_time=pm.Normal('w_time', mu = 1, sd = 10)
    
    
    #Error
    eps = boundednormal2('eps',mu = .0001, sd = .5)
    
    
    #Expected

    score2list=[]
    for val in [1,2,3,4,5]:
        
        xv = X[pd.Series(index).isin([val])].copy()
        yv = Y[pd.Series(index).isin([val])].copy()
        
        xt = X[~pd.Series(index).isin([val,6])].copy()
        yt = Y[~pd.Series(index).isin([val,6])].copy()

        
        knn = KNeighborsClassifier(n_neighbors=k, 
                                           weights='distance', 
                                           algorithm='auto', leaf_size=30, p=2, 
                                           metric='euclidean', metric_params=None, 
                                           n_jobs=None)
        
        
        xtt=xt['age'].values*w_age
        xvv=xv['age'].values*w_age
        
        knn.fit(xtt,yt)
        
    
        tp=sum(np.where((knn.predict(xvv)==1)&(yv==1),1,0))
        fp=sum(np.where((knn.predict(xvv)==1)&(yv==0),1,0))
        tn=sum(np.where((knn.predict(xvv)==0)&(yv==0),1,0))
        fn=sum(np.where((knn.predict(xvv)==0)&(yv==1),1,0))
                    
        
        precision=tp/(tp+fp)
        recall=tp/(tp+fn)
        
        score2 = recall
        score2list.append(score2)
        
    score2=np.mean(score2list)    

        
    Est = score2
    
    #Likelihood
    like = pm.Normal('like',mu = Est, sd = eps, observed = 1)

     
   
with model:
    trace = pm.sample(draws = 2000, tune = 2000, init = None,step=pm.Metropolis(), progressbar = True)
  
pm.traceplot(trace)


summary = pm.summary(trace)


