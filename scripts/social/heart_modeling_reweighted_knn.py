
import numpy as np
"""
This monte carlo algorithm aproximates the "true" value of the interesting 
parameter/s using a random walk of normally distributed steps with mean 0 or
a mean of the last accepted step in the walk for the parameter. 
"""
truth=5

tss = []
for j in range(50):
    ts = []
    stepsizes = [.01,.05,.1,.5,1,5,10]
    index=0
    while len(ts) < len(stepsizes):
        w0 = 0
        
        score1 = abs(truth-w0)
        score=score1
        delta = 0
        
        t = 0
        u = 0
        
        
        stepsize=stepsizes[index]
        while (score1 > .5)&(t<1000):
            w1 = w0+np.random.normal(delta,stepsize)
            
            score2 = abs(truth-w1)
            
            if -score2>-score1:
                delta = w1-w0
                w0 = w1
                score1=score2
                u+=1
            t+=1
            
            print(t,score1,u)
        
        if score1 <=.5:
            ts.append(t)
            index+=1
            
    tss.append(ts)


tss=np.array(tss)
stepsize = stepsizes[np.argmin(np.mean(tss,axis=0))]

truth = 5

w0 = 0
        
score1 = abs(truth-w0)
score=score1
delta = 0

t = 0
u = 0

while (score1 > .5)&(t<1000):
    w1 = w0+np.random.normal(delta,stepsize)
    
    score2 = abs(truth-w1)
    
    if -score2>-score1:
        delta = w1-w0
        w0 = w1
        score1=score2
        u+=1
    t+=1
    
    print(t,score1,u)





    
    


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
dat = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv")

pd.set_option("display.max_columns",500)
dat.tail()

covars = ['age','anaemia','creatinine_phosphokinase',
          'diabetes','ejection_fraction','high_blood_pressure',
          'platelets','serum_creatinine','serum_sodium',
          'sex','smoking','time']


X=dat[covars].copy()
Y=dat['DEATH_EVENT']

Yodds = Y/(1-Y)
Yodds = np.where(Yodds==np.inf,1e16,1e-16)
Ylogodds = np.log(Yodds)

X=(X-np.mean(X,axis=0))/np.std(X,axis=0)
X['int']=1

random.seed(42)
index = np.array(random.choices([1,2,3,4,5],k=len(X)))


xv = X[index==5].copy()
yv = Ylogodds[index==5].copy()

xt = X[index!=5].copy()
yt = Ylogodds[index!=5].copy()


coefs = np.linalg.pinv(xt.T@xt)@(xt.T@yt)


predtlogodds = xt@coefs
predvlogodds = xv@coefs

predt=np.exp(predtlogodds)/(1+np.exp(predtlogodds))
predt=np.where(predt>.5,1,0)
predv=np.exp(predvlogodds)/(1+np.exp(predvlogodds))
predv=np.where(predv>.5,1,0)

act_t = np.exp(yt)/(1+np.exp(yt))
act_t=np.where(act_t>.5,1,0)
act_v = np.exp(yv)/(1+np.exp(yv))
act_v=np.where(act_v>.5,1,0)


logregt_acc=sum(np.where(predt==act_t,1,0))/len(predt)
logregv_acc = sum(np.where(predv==act_v,1,0))/len(predv)
print("logreg training acc:",logregt_acc,"val acc:",logregv_acc)

from sklearn.linear_model import LogisticRegression

xv = X[index==5].copy()
yv = Y[index==5].copy()

xt = X[index!=5].copy()
yt = Y[index!=5].copy()

lr = LogisticRegression(fit_intercept=False,solver = 'newton-cg',penalty='l2')
lr.fit(xt,yt)

sum(np.where(lr.predict(xt)==yt,1,0))/len(yt)
sum(np.where(lr.predict(xv)==yv,1,0))/len(yv)


#BASE KNN Maximizing Recall
from sklearn.neighbors import KNeighborsClassifier


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
        score = precision

        avgscore.append(score)
    acc.append(np.mean(avgscore))

plt.plot(acc)
plt.xticks(list(range(28)),list(range(2,30)))
plt.show()


    #k=18
k=4
k=16
def model_precision(X,Y,w,k):
    
    random.seed(42)
    index = np.array(random.choices([1,2,3,4,5,6],k=len(X)))
    
    
    
    initscores=[]   
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
        
        knn.fit(xt*w,yt)
        
    
        tp=sum(np.where((knn.predict(xv*w)==1)&(yv==1),1,0))
        fp=sum(np.where((knn.predict(xv*w)==1)&(yv==0),1,0))
        tn=sum(np.where((knn.predict(xv*w)==0)&(yv==0),1,0))
        fn=sum(np.where((knn.predict(xv*w)==0)&(yv==1),1,0))
    
        precision=tp/(tp+fp)
        recall=tp/(tp+fn)
        #score = (sum(np.where(knn.predict(xv*w0)==yv,1,0)))/(len(yv))
        score = precision
        initscores.append(score)
    
    score=np.mean(initscores)
    
    return score

def model_recall(X,Y,w,k):
    
    random.seed(42)
    index = np.array(random.choices([1,2,3,4,5,6],k=len(X)))
    
    
    
    initscores=[]   
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
        
        knn.fit(xt*w,yt)
        
    
        tp=sum(np.where((knn.predict(xv*w)==1)&(yv==1),1,0))
        fp=sum(np.where((knn.predict(xv*w)==1)&(yv==0),1,0))
        tn=sum(np.where((knn.predict(xv*w)==0)&(yv==0),1,0))
        fn=sum(np.where((knn.predict(xv*w)==0)&(yv==1),1,0))
    
        precision=tp/(tp+fp)
        recall=tp/(tp+fn)
        #score = (sum(np.where(knn.predict(xv*w0)==yv,1,0)))/(len(yv))
        score = recall
        initscores.append(score)
    
    score=np.mean(initscores)
    
    return score
    

def sequential_MCMC(X,Y,model_fn,draws=30,no_update_limit=120,
                    stepsize=.1,step_shrinkage=.9,
                    delta_reset=20,):
    
    #INITIAL SCORE
    w0 = np.ones(len(X.columns.values))
    score = model_fn(X,Y,w0,k)
    scoreinit=score
    
    wfin = []
    scores = []
    while len(wfin)<draws:
        
        noupdate=0
        deltachosen=False
        stepsize=stepsize
        score=scoreinit
        delta=np.random.normal(0,stepsize/2,len(covars))
        w0=np.ones(len(X.columns.values))
        
        while noupdate<no_update_limit:
            w1 = w0+np.random.normal(delta,stepsize,len(X.columns.values))
            score2 = model_fn(X,Y,w1,k)
            
            if score2>score:
                print(score2,score,"accepted",noupdate)
                deltachosen==True
                score=score2
                delta = w1-w0
                w0=w1
                noupdate=0
            else:
                #print(score2,score)
                noupdate+=1
                if deltachosen==False:
                    delta=np.random.normal(0,stepsize/2,len(X.columns.values))
                if noupdate%delta_reset==delta_reset:
                    deltachosen=False
                    stepsize=stepsize*step_shrinkage
                    delta=np.random.normal(0,stepsize/2,len(X.columns.values))
                    
            
        if score>scoreinit:
            wfin.append(w0)
            scores.append(score)
    
    wfin_arr=np.vstack(wfin)
    return(wfin_arr,scores)
    
wfin_arr,scores=sequential_MCMC(X,Y,model_fn=model_precision,draws=30,no_update_limit=120,
                    stepsize=.1,step_shrinkage=.9,
                    delta_reset=20)
    
    

print(np.mean(wfin_arr,axis=0))
print(np.std(wfin_arr,axis=0))


for i in range(12):
    
    plt.hist(wfin_arr.T[i],bins=10)
    plt.title(covars[i])
    plt.show()
    
    

method=np.median

xv = X[pd.Series(index).isin([6])].copy()
yv = Y[pd.Series(index).isin([6])].copy()

xt = X[pd.Series(index).isin([1,2,3,4,5])].copy()
yt = Y[pd.Series(index).isin([1,2,3,4,5])].copy()

wf=method(wfin_arr,axis=0)


knn = KNeighborsClassifier(n_neighbors=k, 
                                   weights='distance', 
                                   algorithm='auto', leaf_size=30, p=2, 
                                   metric='euclidean', metric_params=None, 
                                   n_jobs=None)
knn.fit(xt*wf,yt)

tp=sum(np.where((knn.predict(xv*wf)==1)&(yv==1),1,0))
fp=sum(np.where((knn.predict(xv*wf)==1)&(yv==0),1,0))
tn=sum(np.where((knn.predict(xv*wf)==0)&(yv==0),1,0))
fn=sum(np.where((knn.predict(xv*wf)==0)&(yv==1),1,0))
print('precision: ',tp/(tp+fp))
print('recall: ',tp/(tp+fn))
print('accuracy: ',(tp+tn)/(tp+fn+fp+tn))




knn = KNeighborsClassifier(n_neighbors=k, 
                                   weights='distance', 
                                   algorithm='auto', leaf_size=30, p=2, 
                                   metric='euclidean', metric_params=None, 
                                   n_jobs=None)


knn.fit(xt,yt)

tp=sum(np.where((knn.predict(xv)==1)&(yv==1),1,0))
fp=sum(np.where((knn.predict(xv)==1)&(yv==0),1,0))
tn=sum(np.where((knn.predict(xv)==0)&(yv==0),1,0))
fn=sum(np.where((knn.predict(xv)==0)&(yv==1),1,0))
print('precision: ',tp/(tp+fp))
print('recall: ',tp/(tp+fn))
print('accuracy: ',(tp+tn)/(tp+fn+fp+tn))





wfin_arr,scores=sequential_MCMC(X,Y,model_fn=model_recall,draws=30,no_update_limit=120,
                    stepsize=.1,step_shrinkage=.9,
                    delta_reset=20)
    

knn = KNeighborsClassifier(n_neighbors=k, 
                                   weights='distance', 
                                   algorithm='auto', leaf_size=30, p=2, 
                                   metric='euclidean', metric_params=None, 
                                   n_jobs=None)
knn.fit(xt*wf,yt)

tp=sum(np.where((knn.predict(xv*wf)==1)&(yv==1),1,0))
fp=sum(np.where((knn.predict(xv*wf)==1)&(yv==0),1,0))
tn=sum(np.where((knn.predict(xv*wf)==0)&(yv==0),1,0))
fn=sum(np.where((knn.predict(xv*wf)==0)&(yv==1),1,0))
print('precision: ',tp/(tp+fp))
print('recall: ',tp/(tp+fn))
print('accuracy: ',(tp+tn)/(tp+fn+fp+tn))




knn = KNeighborsClassifier(n_neighbors=k, 
                                   weights='distance', 
                                   algorithm='auto', leaf_size=30, p=2, 
                                   metric='euclidean', metric_params=None, 
                                   n_jobs=None)


knn.fit(xt,yt)

tp=sum(np.where((knn.predict(xv)==1)&(yv==1),1,0))
fp=sum(np.where((knn.predict(xv)==1)&(yv==0),1,0))
tn=sum(np.where((knn.predict(xv)==0)&(yv==0),1,0))
fn=sum(np.where((knn.predict(xv)==0)&(yv==1),1,0))
print('precision: ',tp/(tp+fp))
print('recall: ',tp/(tp+fn))
print('accuracy: ',(tp+tn)/(tp+fn+fp+tn))


    
    
initscores=[]   
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
    
    stepsize=.1
    w0=np.ones(len(covars))
    delta=np.random.normal(0,stepsize/2,len(covars))
    knn.fit(xt*w0,yt)
    

    tp=sum(np.where((knn.predict(xv*w0)==1)&(yv==1),1,0))
    fp=sum(np.where((knn.predict(xv*w0)==1)&(yv==0),1,0))
    tn=sum(np.where((knn.predict(xv*w0)==0)&(yv==0),1,0))
    fn=sum(np.where((knn.predict(xv*w0)==0)&(yv==1),1,0))

    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    #score = (sum(np.where(knn.predict(xv*w0)==yv,1,0)))/(len(yv))
    score = recall
    initscores.append(score)

score=np.mean(initscores)
scoreinit=score
#sum(np.where(knn.predict(xv*w0)==yv,1,0))/len(yv)
#sum(np.where(knn.predict(xt*w0)==yt,1,0))/len(yt)


wfin=[]
scores = []
while len(wfin)<30:
    
    
    noupdate=0
    deltachosen=False
    score=scoreinit
    stepsize=.1
    delta=np.random.normal(0,stepsize/2,len(covars))
    w0=np.ones(len(covars))
    #iteration=0
    
    while noupdate<120:
        
        #iteration+=1
        #val = iteration%4+1
        
        score2list=[]
        for val in [1,2,3,4,5]:
            
            xv = X[pd.Series(index).isin([val])].copy()
            yv = Y[pd.Series(index).isin([val])].copy()
            
            xt = X[~pd.Series(index).isin([val,6])].copy()
            yt = Y[~pd.Series(index).isin([val,6])].copy()
    
            
            w1 = w0+np.random.normal(delta,stepsize,len(covars))
            
            knn = KNeighborsClassifier(n_neighbors=k, 
                                               weights='distance', 
                                               algorithm='auto', leaf_size=30, p=2, 
                                               metric='euclidean', metric_params=None, 
                                               n_jobs=None)
        
            
            knn.fit(xt*w1,yt)
            
            tp=sum(np.where((knn.predict(xv*w1)==1)&(yv==1),1,0))
            fp=sum(np.where((knn.predict(xv*w1)==1)&(yv==0),1,0))
            tn=sum(np.where((knn.predict(xv*w1)==0)&(yv==0),1,0))
            fn=sum(np.where((knn.predict(xv*w1)==0)&(yv==1),1,0))
                        
            
            precision=tp/(tp+fp)
            recall=tp/(tp+fn)
            #score2 = sum(np.where(knn.predict(xv*w1)==yv,1,0))/len(yv)
            score2 = recall
            score2list.append(score2)
            
        score2=np.mean(score2list)    
        if score2>score:
            print(score2,score,"accepted",noupdate)
            deltachosen==True
            score=score2
            delta = w1-w0
            w0=w1
            noupdate=0
        else:
            #print(score2,score)
            noupdate+=1
            if deltachosen==False:
                delta=np.random.normal(0,stepsize/2,len(covars))
            if noupdate%20==20:
                deltachosen=False
                stepsize=stepsize*.9
                delta=np.random.normal(0,stepsize/2,len(covars))
                
                
    if score>scoreinit:
        wfin.append(w0)
        scores.append(score)


wfin_arr=np.vstack(wfin)


print(np.mean(wfin_arr,axis=0))
print(np.std(wfin_arr,axis=0))

for i in range(12):
    
    plt.hist(wfin_arr.T[i],bins=10)
    plt.title(covars[i])
    plt.show()
    
    
method=np.mean

xv = X[pd.Series(index).isin([6])].copy()
yv = Y[pd.Series(index).isin([6])].copy()

xt = X[pd.Series(index).isin([1,2,3,4,5])].copy()
yt = Y[pd.Series(index).isin([1,2,3,4,5])].copy()

wf=method(wfin_arr,axis=0)


knn = KNeighborsClassifier(n_neighbors=k, 
                                   weights='distance', 
                                   algorithm='auto', leaf_size=30, p=2, 
                                   metric='euclidean', metric_params=None, 
                                   n_jobs=None)
knn.fit(xt*wf,yt)

tp=sum(np.where((knn.predict(xv*wf)==1)&(yv==1),1,0))
fp=sum(np.where((knn.predict(xv*wf)==1)&(yv==0),1,0))
tn=sum(np.where((knn.predict(xv*wf)==0)&(yv==0),1,0))
fn=sum(np.where((knn.predict(xv*wf)==0)&(yv==1),1,0))
print('precision: ',tp/(tp+fp))
print('recall: ',tp/(tp+fn))
print('accuracy: ',(tp+tn)/(tp+fn+fp+tn))




knn = KNeighborsClassifier(n_neighbors=k, 
                                   weights='distance', 
                                   algorithm='auto', leaf_size=30, p=2, 
                                   metric='euclidean', metric_params=None, 
                                   n_jobs=None)


knn.fit(xt,yt)

tp=sum(np.where((knn.predict(xv)==1)&(yv==1),1,0))
fp=sum(np.where((knn.predict(xv)==1)&(yv==0),1,0))
tn=sum(np.where((knn.predict(xv)==0)&(yv==0),1,0))
fn=sum(np.where((knn.predict(xv)==0)&(yv==1),1,0))
print('precision: ',tp/(tp+fp))
print('recall: ',tp/(tp+fn))
print('accuracy: ',(tp+tn)/(tp+fn+fp+tn))



scores_ordered = sorted(range(len(scores)), key=lambda k: scores[k])
wfin_sorted = wfin_arr[scores_ordered]
wfin_selected = wfin_sorted[15:]
wf_sort=method(wfin_selected,axis=0)
knn = KNeighborsClassifier(n_neighbors=k, 
                                   weights='distance', 
                                   algorithm='auto', leaf_size=30, p=2, 
                                   metric='euclidean', metric_params=None, 
                                   n_jobs=None)


knn.fit(xt*wf_sort,yt)

tp=sum(np.where((knn.predict(xv*wf_sort)==1)&(yv==1),1,0))
fp=sum(np.where((knn.predict(xv*wf_sort)==1)&(yv==0),1,0))
tn=sum(np.where((knn.predict(xv*wf_sort)==0)&(yv==0),1,0))
fn=sum(np.where((knn.predict(xv*wf_sort)==0)&(yv==1),1,0))
print('precision: ',tp/(tp+fp))
print('recall: ',tp/(tp+fn))
print('accuracy: ',(tp+tn)/(tp+fn+fp+tn))








#BASE KNN Maximizing Precision
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")


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
        score = precision

        avgscore.append(score)
    acc.append(np.mean(avgscore))

plt.plot(acc)
plt.xticks(list(range(28)),list(range(2,30)))
plt.show()


    #k=18
k=17
    
initscores=[]   
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
    
    stepsize=.1
    w0=np.ones(len(covars))
    delta=np.random.normal(0,stepsize/2,len(covars))
    knn.fit(xt*w0,yt)
    

    tp=sum(np.where((knn.predict(xv*w0)==1)&(yv==1),1,0))
    fp=sum(np.where((knn.predict(xv*w0)==1)&(yv==0),1,0))
    tn=sum(np.where((knn.predict(xv*w0)==0)&(yv==0),1,0))
    fn=sum(np.where((knn.predict(xv*w0)==0)&(yv==1),1,0))

    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    #score = (sum(np.where(knn.predict(xv*w0)==yv,1,0)))/(len(yv))
    score = round(precision,5)
    initscores.append(score)

score=np.mean(initscores)
scoreinit=round(score,5)
#sum(np.where(knn.predict(xv*w0)==yv,1,0))/len(yv)
#sum(np.where(knn.predict(xt*w0)==yt,1,0))/len(yt)



wfin=[]
scores = []
while len(wfin)<30:
    
    
    noupdate=0
    deltachosen=False
    score=scoreinit
    stepsize=.1
    delta=np.random.normal(0,stepsize/2,len(covars))
    w0=np.ones(len(covars))
    #iteration=0
    
    while noupdate<120:
        
        #iteration+=1
        #val = iteration%4+1
        
        score2list=[]
        for val in [1,2,3,4,5]:
            
            xv = X[pd.Series(index).isin([val])].copy()
            yv = Y[pd.Series(index).isin([val])].copy()
            
            xt = X[~pd.Series(index).isin([val,6])].copy()
            yt = Y[~pd.Series(index).isin([val,6])].copy()
    
            
            w1 = w0+np.random.normal(delta,stepsize,len(covars))
            
            knn = KNeighborsClassifier(n_neighbors=k, 
                                               weights='distance', 
                                               algorithm='auto', leaf_size=30, p=2, 
                                               metric='euclidean', metric_params=None, 
                                               n_jobs=None)
        
            
            knn.fit(xt*w1,yt)
            
            tp=sum(np.where((knn.predict(xv*w1)==1)&(yv==1),1,0))
            fp=sum(np.where((knn.predict(xv*w1)==1)&(yv==0),1,0))
            tn=sum(np.where((knn.predict(xv*w1)==0)&(yv==0),1,0))
            fn=sum(np.where((knn.predict(xv*w1)==0)&(yv==1),1,0))
                        
            
            precision=tp/(tp+fp)
            recall=tp/(tp+fn)
            #score2 = sum(np.where(knn.predict(xv*w1)==yv,1,0))/len(yv)
            score2 = round(precision,5)
            score2list.append(score2)
            
        score2=round(np.mean(score2list) ,5   )
        if score2>score:
            print(score2,score,"accepted",noupdate)
            deltachosen==True
            score=score2
            delta = w1-w0
            w0=w1
            noupdate=0
        else:
            #print(score2,score)
            noupdate+=1
            if deltachosen==False:
                delta=np.random.normal(0,stepsize/2,len(covars))
            if noupdate%20==20:
                deltachosen=False
                stepsize=stepsize*.9
                delta=np.random.normal(0,stepsize/2,len(covars))
                
                
    if score>scoreinit:
        wfin.append(w0)
        scores.append(score)


wfin_arr=np.vstack(wfin)


print(np.mean(wfin_arr,axis=0))
print(np.std(wfin_arr,axis=0))

for i in range(12):
    
    plt.hist(wfin_arr.T[i],bins=10)
    plt.title(covars[i])
    plt.show()
    
    
method=np.mean

xv = X[pd.Series(index).isin([6])].copy()
yv = Y[pd.Series(index).isin([6])].copy()

xt = X[pd.Series(index).isin([1,2,3,4,5])].copy()
yt = Y[pd.Series(index).isin([1,2,3,4,5])].copy()

wf=method(wfin_arr,axis=0)


knn = KNeighborsClassifier(n_neighbors=k, 
                                   weights='distance', 
                                   algorithm='auto', leaf_size=30, p=2, 
                                   metric='euclidean', metric_params=None, 
                                   n_jobs=None)
knn.fit(xt*wf,yt)

tp=sum(np.where((knn.predict(xv*wf)==1)&(yv==1),1,0))
fp=sum(np.where((knn.predict(xv*wf)==1)&(yv==0),1,0))
tn=sum(np.where((knn.predict(xv*wf)==0)&(yv==0),1,0))
fn=sum(np.where((knn.predict(xv*wf)==0)&(yv==1),1,0))
print('reweighted precision: ',tp/(tp+fp))
print('recall: ',tp/(tp+fn))
print('accuracy: ',(tp+tn)/(tp+fn+fp+tn))




knn = KNeighborsClassifier(n_neighbors=k, 
                                   weights='distance', 
                                   algorithm='auto', leaf_size=30, p=2, 
                                   metric='euclidean', metric_params=None, 
                                   n_jobs=None)


knn.fit(xt,yt)

tp=sum(np.where((knn.predict(xv)==1)&(yv==1),1,0))
fp=sum(np.where((knn.predict(xv)==1)&(yv==0),1,0))
tn=sum(np.where((knn.predict(xv)==0)&(yv==0),1,0))
fn=sum(np.where((knn.predict(xv)==0)&(yv==1),1,0))
print('unweighted precision: ',tp/(tp+fp))
print('recall: ',tp/(tp+fn))
print('accuracy: ',(tp+tn)/(tp+fn+fp+tn))



scores_ordered = sorted(range(len(scores)), key=lambda k: scores[k])
wfin_sorted = wfin_arr[scores_ordered]
wfin_selected = wfin_sorted[15:]
wf_sort=method(wfin_selected,axis=0)
knn = KNeighborsClassifier(n_neighbors=k, 
                                   weights='distance', 
                                   algorithm='auto', leaf_size=30, p=2, 
                                   metric='euclidean', metric_params=None, 
                                   n_jobs=None)


knn.fit(xt*wf_sort,yt)

tp=sum(np.where((knn.predict(xv*wf_sort)==1)&(yv==1),1,0))
fp=sum(np.where((knn.predict(xv*wf_sort)==1)&(yv==0),1,0))
tn=sum(np.where((knn.predict(xv*wf_sort)==0)&(yv==0),1,0))
fn=sum(np.where((knn.predict(xv*wf_sort)==0)&(yv==1),1,0))
print('selective precision: ',tp/(tp+fp))
print('recall: ',tp/(tp+fn))
print('accuracy: ',(tp+tn)/(tp+fn+fp+tn))


len(yv)










#BASE KNN Maximizing Precision & Recall
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")


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
        score = (precision+recall)/2

        avgscore.append(score)
    acc.append(np.mean(avgscore))

plt.plot(acc)
plt.xticks(list(range(28)),list(range(2,30)))
plt.show()


    #k=18
k=10
    
initscores=[]   
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
    
    stepsize=.1
    w0=np.ones(len(covars))
    delta=np.random.normal(0,stepsize/2,len(covars))
    knn.fit(xt*w0,yt)
    

    tp=sum(np.where((knn.predict(xv*w0)==1)&(yv==1),1,0))
    fp=sum(np.where((knn.predict(xv*w0)==1)&(yv==0),1,0))
    tn=sum(np.where((knn.predict(xv*w0)==0)&(yv==0),1,0))
    fn=sum(np.where((knn.predict(xv*w0)==0)&(yv==1),1,0))

    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    #score = (sum(np.where(knn.predict(xv*w0)==yv,1,0)))/(len(yv))
    score = round((precision+recall)/2,5)
    initscores.append(score)

score=np.mean(initscores)
scoreinit=round(score,5)
#sum(np.where(knn.predict(xv*w0)==yv,1,0))/len(yv)
#sum(np.where(knn.predict(xt*w0)==yt,1,0))/len(yt)



wfin=[]
scores = []
while len(wfin)<30:
    
    
    noupdate=0
    deltachosen=False
    score=scoreinit
    stepsize=.1
    delta=np.random.normal(0,stepsize/2,len(covars))
    w0=np.ones(len(covars))
    #iteration=0
    
    while noupdate<120:
        
        #iteration+=1
        #val = iteration%4+1
        
        score2list=[]
        for val in [1,2,3,4,5]:
            
            xv = X[pd.Series(index).isin([val])].copy()
            yv = Y[pd.Series(index).isin([val])].copy()
            
            xt = X[~pd.Series(index).isin([val,6])].copy()
            yt = Y[~pd.Series(index).isin([val,6])].copy()
    
            
            w1 = w0+np.random.normal(delta,stepsize,len(covars))
            
            knn = KNeighborsClassifier(n_neighbors=k, 
                                               weights='distance', 
                                               algorithm='auto', leaf_size=30, p=2, 
                                               metric='euclidean', metric_params=None, 
                                               n_jobs=None)
        
            
            knn.fit(xt*w1,yt)
            
            tp=sum(np.where((knn.predict(xv*w1)==1)&(yv==1),1,0))
            fp=sum(np.where((knn.predict(xv*w1)==1)&(yv==0),1,0))
            tn=sum(np.where((knn.predict(xv*w1)==0)&(yv==0),1,0))
            fn=sum(np.where((knn.predict(xv*w1)==0)&(yv==1),1,0))
                        
            
            precision=tp/(tp+fp)
            recall=tp/(tp+fn)
            #score2 = sum(np.where(knn.predict(xv*w1)==yv,1,0))/len(yv)
            score2 = round((precision+recall)/2,5)
            score2list.append(score2)
            
        score2=round(np.mean(score2list) ,5   )
        if score2>score:
            print(score2,score,"accepted",noupdate)
            deltachosen==True
            score=score2
            delta = w1-w0
            w0=w1
            noupdate=0
        else:
            #print(score2,score)
            noupdate+=1
            if deltachosen==False:
                delta=np.random.normal(0,stepsize/2,len(covars))
            if noupdate%20==20:
                deltachosen=False
                stepsize=stepsize*.9
                delta=np.random.normal(0,stepsize/2,len(covars))
                
                
    if score>scoreinit:
        wfin.append(w0)
        scores.append(score)


wfin_arr=np.vstack(wfin)


print(np.mean(wfin_arr,axis=0))
print(np.std(wfin_arr,axis=0))

for i in range(12):
    
    plt.hist(wfin_arr.T[i],bins=10)
    plt.title(covars[i])
    plt.show()
    
    
method=np.mean

xv = X[pd.Series(index).isin([6])].copy()
yv = Y[pd.Series(index).isin([6])].copy()

xt = X[pd.Series(index).isin([1,2,3,4,5])].copy()
yt = Y[pd.Series(index).isin([1,2,3,4,5])].copy()

wf=method(wfin_arr,axis=0)


knn = KNeighborsClassifier(n_neighbors=k, 
                                   weights='distance', 
                                   algorithm='auto', leaf_size=30, p=2, 
                                   metric='euclidean', metric_params=None, 
                                   n_jobs=None)
knn.fit(xt*wf,yt)

tp=sum(np.where((knn.predict(xv*wf)==1)&(yv==1),1,0))
fp=sum(np.where((knn.predict(xv*wf)==1)&(yv==0),1,0))
tn=sum(np.where((knn.predict(xv*wf)==0)&(yv==0),1,0))
fn=sum(np.where((knn.predict(xv*wf)==0)&(yv==1),1,0))
print('reweighted precision: ',tp/(tp+fp))
print('recall: ',tp/(tp+fn))
print('accuracy: ',(tp+tn)/(tp+fn+fp+tn))




knn = KNeighborsClassifier(n_neighbors=k, 
                                   weights='distance', 
                                   algorithm='auto', leaf_size=30, p=2, 
                                   metric='euclidean', metric_params=None, 
                                   n_jobs=None)


knn.fit(xt,yt)

tp=sum(np.where((knn.predict(xv)==1)&(yv==1),1,0))
fp=sum(np.where((knn.predict(xv)==1)&(yv==0),1,0))
tn=sum(np.where((knn.predict(xv)==0)&(yv==0),1,0))
fn=sum(np.where((knn.predict(xv)==0)&(yv==1),1,0))
print('unweighted precision: ',tp/(tp+fp))
print('recall: ',tp/(tp+fn))
print('accuracy: ',(tp+tn)/(tp+fn+fp+tn))



scores_ordered = sorted(range(len(scores)), key=lambda k: scores[k])
wfin_sorted = wfin_arr[scores_ordered]
wfin_selected = wfin_sorted[15:]
wf_sort=method(wfin_selected,axis=0)
knn = KNeighborsClassifier(n_neighbors=k, 
                                   weights='distance', 
                                   algorithm='auto', leaf_size=30, p=2, 
                                   metric='euclidean', metric_params=None, 
                                   n_jobs=None)


knn.fit(xt*wf_sort,yt)

tp=sum(np.where((knn.predict(xv*wf_sort)==1)&(yv==1),1,0))
fp=sum(np.where((knn.predict(xv*wf_sort)==1)&(yv==0),1,0))
tn=sum(np.where((knn.predict(xv*wf_sort)==0)&(yv==0),1,0))
fn=sum(np.where((knn.predict(xv*wf_sort)==0)&(yv==1),1,0))
print('selective precision: ',tp/(tp+fp))
print('recall: ',tp/(tp+fn))
print('accuracy: ',(tp+tn)/(tp+fn+fp+tn))


len(yv)





