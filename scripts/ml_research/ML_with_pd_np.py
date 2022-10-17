

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dat=pd.read_csv("c:/users/jliv/downloads/qsar_aquatic_toxicity.csv",header=None,sep=";")
dat = dat.rename(mapper = {0:'TSPA',1:'SAaac',2:'H-050',3:'MLOGP',4:'RDCHI',5:'GATS1p',6:'nN',7:'C-040',8:'LC50'},axis='columns')

#EUC
dat.corr()['LC50']

variables = ['MLOGP', 'RDCHI', 'GATS1p']


train = dat.sample(int(len(dat)*.8)).copy()
test = dat[~dat.index.isin(train.index)]

X = train[variables].copy()
X['int']=1

Y=train['LC50']

testx = test[variables].copy()
testx['int']=1

testy=test['LC50']

#LINEAR REGRESSION 
'''
Model: Y = X@B
Cost Function: C = .5*(X@B-Y)**2
Gradient: grad_C=X.T@(X@B-Y)
Hessian: hess_C=X.T@X

Explicit Solution: 
    X.T@(X@B-Y)=0
    X.T@(X@B)-X.T@Y=0
    X.T@(X@B)=X.T@Y
    (X.T@X)@B=X.T@Y
    B=(X.T@X)**-1@(X.T@Y)
'''

#Explicit Solution

explicit_coefs = np.linalg.inv(X.T@X)@(X.T@Y)

#Newtonian Descent, random initial coefs, update B_n+1 = B_n - hess**-1@grad

newt_coefs = np.random.normal(0,.5,4)
grad = X.T@(X@newt_coefs-Y)
hess = X.T@X
newt_coefs=newt_coefs-np.linalg.inv(hess)@grad

#Gradient Descent with constant learning rate

grad_coefs = np.random.normal(0,.5,4)
lr = .00001
for i in range(50000):
    grad_coefs=grad_coefs-lr*(X.T@(X@grad_coefs-Y))



# LOGISTIC REGRESSION Data Prep
    
dat = pd.read_csv("c:/users/jliv/downloads/adult.data",header=None)

dat=dat.rename(mapper = {0:'age',
                          1:'workclass',
                          2:'fnlwgt',
                          3:'education',
                          4:'education_num',
                          5:'marital_status',
                          6:'occupation',
                          7:'relationship',
                          8:'race',
                          9:'sex',
                          10:'capital_gain',
                          11:'capital_loss',
                          12:'hours_per_week',
                          13:'native_country',
                          14:'income'},axis='columns')
    
    

dat['income']=np.where(dat['income']==" >50K",1,0)

onehot_cols = ['workclass','education','marital_status','occupation','relationship','race','sex']
matchdf = []
for j in onehot_cols:
    for i in dat[j].unique():
        dat[j+"_"+i]=np.where(dat[j]==i,1,0)
        
        match=max([sum(np.where(dat[j+"_"+i]==dat['income'],1,0)),sum(np.where(dat[j+"_"+i]!=dat['income'],1,0))])
        matchdf.append([j+"_"+i,match])
        
matchdf=pd.DataFrame(matchdf,columns=['dim','match'])
matchdf.sort_values(by='match',ascending=False,inplace=True)
matchdf.reset_index(inplace=True,drop=True)
plt.plot(matchdf['match']) 

variables=  list(matchdf['dim'][:14])


X = dat[variables].copy()
X['int']=1
Y=dat['income']

#Explicit Solution  
#Calculate log-odds from label
#Linear model for log odds


LO = Y/(1-Y)
LO = np.log(np.where(LO == np.inf,1e14,1e-14))

logistic_explicit_coefs = np.linalg.inv(X.T@X)@(X.T@LO)


Opred = np.exp(X@logistic_explicit_coefs)

Ypred = Opred/(1+Opred)
Ypred = np.where(Ypred>.5,1,0)


err = np.where(Ypred==Y,1,0)

acc = np.mean(err)

#Descent solution

logistic_coefs = np.random.normal(0,.5,len(X.columns.values))


def sigmoid(x):
    return 1/(1+np.e**(-x))

def del_sig(x):
    return sigmoid(x)*(1-sigmoid(x))

def predict(X,B):
    return sigmoid(X@B)

def activation(x):
    return np.where(x>.5,1,0)

lr=.001
grad0 = np.array([1000 for i in logistic_coefs])
tol = 1e-8
for i in range(500):
    grad = (X.T@((predict(X,logistic_coefs)-Y)*del_sig(predict(X,logistic_coefs))))
    logistic_coefs=logistic_coefs-lr*grad
    
    print(i,round(np.mean(np.where(activation(predict(X,logistic_coefs))==Y,1,0)),5))
    if max(grad/grad0-1)<=tol:
        break
    else:
        grad0=grad
    
pred = np.where(predict(X,logistic_coefs)>.5,1,0)

err = np.where(pred==Y,1,0)

acc = np.mean(err)

randompred=np.where(np.random.normal(0,1,len(pred))>0,1,0)


randerr = np.where(randompred==Y,1,0)
randacc = np.mean(randerr)


tp = sum(np.where((pred==1)&(Y==1),1,0))
tn = sum(np.where((pred==0)&(Y==0),1,0))

fp = sum(np.where((pred==1)&(Y==0),1,0))
fn = sum(np.where((pred==0)&(Y==1),1,0))

precision=tp/(tp+fp)

recall=tp/(tp+fn)

accuracy=(tp+tn)/(tp+tn+fp+fn)

#Kmeans
d1 = np.random.normal(3,.5,(10,3))
d2=np.random.normal(5,.5,(8,3))

d3=np.random.normal(7,.5,(8,3))


d = np.vstack((d1,d2,d3))

centroids=3

c=np.random.normal(np.mean(d),np.std(d),(centroids,d.shape[1]))


def kmeans(dat,centroids,max_iter):
    
    d = dat    
    c=np.random.normal(np.mean(d),np.std(d),(centroids,d.shape[1]))
    
    def mydist(d,c):
        distarray=[]
        for i in range(c.shape[0]):
            distarray.append(np.sum((d-c[i])**2,axis=1)**.5)
        
        distarray=np.array(distarray)
        return distarray   
        
    for j in range(16):
        dists = mydist(d,c).T
        clusts=np.argmin(dists,axis=1)
        for i in range(centroids):
            c[i]=np.mean(d[clusts==i],axis=0)
        
    return clusts
        
kmeans(d,3,16)