import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dat=pd.read_csv("c:/users/jliv/downloads/qsar_aquatic_toxicity.csv",header=None,sep=";")
dat = dat.rename(mapper = {0:'TSPA',1:'SAaac',2:'H-050',3:'MLOGP',4:'RDCHI',5:'GATS1p',6:'nN',7:'C-040',8:'LC50'},axis='columns')

#EUC
dat.corr()
dat['GATS1p_inv']=dat['GATS1p']*-1
variables = ['MLOGP', 'RDCHI', 'GATS1p_inv']

np.round(dat[variables+['LC50']].corr(),2)
plt.matshow(dat[variables+['LC50']].corr())
plt.xticks(range(len(variables)+1),variables+['LC50'])
plt.yticks(range(len(variables)+1),variables+['LC50'])
cb = plt.colorbar()
plt.show()


euc_betas = []
eucmaes = []

linear_betas = []
linmaes = []

t=0
while t < 100:
    train = dat.sample(int(len(dat)*.8)).copy()
    test = dat[~dat.index.isin(train.index)]
               
    
    trainx = train[variables].copy()
    mins=np.min(trainx,axis=0)
    stds = np.std(trainx,axis=0)
    for i in trainx.columns.values:
        trainx[i]=(trainx[i])/stds[i]
        mins[i]=min(trainx[i])
        trainx[i]=trainx[i]-min(trainx[i])
    
    
    trainx['int']=1
    
    trainx2 = trainx**2
    
    trainy=train['LC50']
    
    testx = test[variables].copy()
    for i in testx.columns.values:
        testx[i]=(testx[i])/stds[i]
        testx[i]=testx[i]-mins[i]
    
    testx['int']=1
    
    testx2 = testx**2
    
    testy=test['LC50']
    #EUC
    eucbeta = np.random.normal(1,.005,len(variables)+1)
      
    
    e1 = 1000
    tol = 1e-6
    
    e=[]
    
    X=np.array(trainx2.reset_index(drop=True).copy())
    Y=np.array(trainy.reset_index(drop=True))
    eb2 = eucbeta**2
    
    for i in range(5000):
        pred = (X@eb2)**.5
        
        grad = .5*X.T@(1-Y*(X@eb2)**(-.5))
        
        hess = .25*np.outer((X.T@Y),(X.T@(X@eb2)**(-1.5)))
        
        direction=np.linalg.pinv(hess)@grad
        if max(direction)< tol:
            break
        else:
            err = np.mean(abs((np.array(testx2)@eb2)**.5-testy.values))
            e.append(err)
            eb2=eb2-direction
        
    
        
    print("EUC MAE:",round(np.mean(abs((np.array(testx2)@eb2)**.5-testy.values)),3))
    
    #plt.plot(e)
    #plt.show()
    linbeta = np.linalg.pinv(trainx.T@trainx)@(trainx.T@trainy)
    
    print("LinReg Prediction MAE:",round(np.mean(abs((testx@linbeta).values-testy.values)),3))
    
    euc_betas.append(list(eb2**.5))
    eucmaes.append(np.mean(abs((np.array(testx2)@eb2)**.5-testy.values)))
    
    linear_betas.append(list(linbeta))
    linmaes.append(np.mean(abs((testx@linbeta).values-testy.values)))
    
    t+=1
    


bins = np.linspace(-1, 4, 100)

euc_betas = np.array(euc_betas)

plt.hist(euc_betas.T[0],bins=bins,label=variables[0],alpha=.6)
plt.hist(euc_betas.T[1],bins=bins,label=variables[1],alpha=.6)
plt.hist(euc_betas.T[2],bins=bins,label=variables[2],alpha=.6)
plt.hist(euc_betas.T[3],bins=bins,label='int',alpha=.6)
plt.legend()
plt.title("Euc Regression")
plt.xlim(-1,4)
plt.show()


linear_betas = np.array(linear_betas)

plt.hist(linear_betas.T[0],bins=bins,label=variables[0],alpha=.6)
plt.hist(linear_betas.T[1],bins=bins,label=variables[1],alpha=.6)
plt.hist(linear_betas.T[2],bins=bins,label=variables[2],alpha=.6)
plt.hist(linear_betas.T[3],bins=bins,label='int',alpha=.6)
plt.legend()
plt.title("Linear Regression")
plt.xlim(-1,4)
plt.show()


euc_iqr=np.percentile(euc_betas.T,75,axis=1)-np.percentile(euc_betas.T,25,axis=1)
lin_iqr=np.percentile(linear_betas.T,75,axis=1)-np.percentile(linear_betas.T,25,axis=1)

print("Lin iqr",lin_iqr)
print("Euc iqr",euc_iqr)

plt.scatter(range(4),euc_iqr,label="iqr spread, euc")
plt.scatter(range(4),lin_iqr,label="iqr spread, linear")
plt.legend()
plt.xticks(range(4),variables+['int'])
plt.show()


lin_stab=np.std(linear_betas.T,axis=1)
euc_stab=np.std(euc_betas.T,axis=1)
print("Lin Stability",lin_stab)
print("Euc Stability",euc_stab)

plt.scatter(range(4),euc_stab,label="std spread, euc")
plt.scatter(range(4),lin_stab,label="std spread, linear")
plt.legend()
plt.xticks(range(4),variables+['int'])
plt.show()


print("Lin MAE",np.mean(linmaes))
print("Euc MAE",np.mean(eucmaes))

plt.hist(linmaes,alpha=.5)
plt.hist(eucmaes,alpha=.5)
plt.show()

 
   

    
    
'''

euc_betas = []
eucmaes = []

linear_betas = []
linmaes = []

t=0
while t < 1:
    train = dat.sample(int(len(dat)*.8)).copy()
    test = dat[~dat.index.isin(train.index)]
               
    
    trainx = train[variables].copy()
    mins=np.min(trainx,axis=0)
    stds = np.std(trainx,axis=0)
    for i in trainx.columns.values:
        trainx[i]=(trainx[i])/stds[i]
        mins[i]=min(trainx[i])
        trainx[i]=trainx[i]-min(trainx[i])
    
    
    trainx['int']=1
    
    trainx2 = trainx**2
    
    trainy=train['LC50']
    
    testx = test[variables].copy()
    for i in testx.columns.values:
        testx[i]=(testx[i])/stds[i]
        testx[i]=testx[i]-mins[i]
    
    testx['int']=1
    
    testx2 = testx**2
    
    
    #EUC
    eucbeta = np.random.normal(1,.005,len(variables)+1)
      
    
    e1 = 1000
    tol = 1e-6
    
    e=[]
    
    for count in range(6000):
        

        pred = np.sqrt((trainx2)@(eucbeta**2))
        err = pred-trainy.T
        
        e2 = np.mean(abs(err))
        e.append(e2)
       
        if (e1-e2< tol):
            break
        else:
            e1=e2
            grad = ((np.sqrt((trainx2)@(eucbeta**2))-trainy.T)*.5*((trainx2)@(eucbeta**2))**(-.5))@(trainx2)
            eucbeta = eucbeta-.0001*grad
            
    #plt.plot(e[500:])
    
    #plt.plot((pd.Series(e).shift(1)-pd.Series(e))[500:])
    
    pred = np.sqrt((trainx2)@(eucbeta**2))
   
    TRAINERROR =np.mean(abs(pred.values-trainy.values))
    
    testpred = np.sqrt((testx2)@(eucbeta**2))
    
    
    testy=test['LC50']
    TESTERROR =np.mean(abs(testpred.values-testy.values))
    
    print(t,'',TRAINERROR,'   test error:',TESTERROR)
    euc_betas.append(list(eucbeta))
    eucmaes.append(TESTERROR)
    
    #LR
    linearbeta=np.linalg.inv(trainx.T@trainx)@(trainx.T@trainy)
    trainpred = trainx@linearbeta
    TRAINERROR =np.mean(abs(trainpred.values-trainy.values))
    
    testpred = testx@linearbeta
    TESTERROR =np.mean(abs(testpred.values-testy.values))
    linear_betas.append(list(linearbeta))
    linmaes.append(TESTERROR)
    
    t+=1
    

 
bins = np.linspace(-1, 4, 100)

euc_betas = np.array(euc_betas)

plt.hist(euc_betas.T[0],bins=bins,label=variables[0],alpha=.6)
plt.hist(euc_betas.T[1],bins=bins,label=variables[1],alpha=.6)
plt.hist(euc_betas.T[2],bins=bins,label=variables[2],alpha=.6)
plt.hist(euc_betas.T[3],bins=bins,label='int',alpha=.6)
plt.legend()
plt.title("Euc Regression")
plt.xlim(-1,4)
plt.show()


linear_betas = np.array(linear_betas)

plt.hist(linear_betas.T[0],bins=bins,label=variables[0],alpha=.6)
plt.hist(linear_betas.T[1],bins=bins,label=variables[1],alpha=.6)
plt.hist(linear_betas.T[2],bins=bins,label=variables[2],alpha=.6)
plt.hist(linear_betas.T[3],bins=bins,label='int',alpha=.6)
plt.legend()
plt.title("Linear Regression")
plt.xlim(-1,4)
plt.show()


euc_iqr=np.percentile(euc_betas.T,75,axis=1)-np.percentile(euc_betas.T,25,axis=1)
lin_iqr=np.percentile(linear_betas.T,75,axis=1)-np.percentile(linear_betas.T,25,axis=1)

print("Lin iqr",lin_iqr)
print("Euc iqr",euc_iqr)

plt.scatter(range(4),euc_iqr,label="iqr spread, euc")
plt.scatter(range(4),lin_iqr,label="iqr spread, linear")
plt.legend()
plt.xticks(range(4),variables+['int'])
plt.show()


lin_stab=np.std(linear_betas.T,axis=1)
euc_stab=np.std(euc_betas.T,axis=1)
print("Lin Stability",lin_stab)
print("Euc Stability",euc_stab)

plt.scatter(range(4),euc_stab,label="std spread, euc")
plt.scatter(range(4),lin_stab,label="std spread, linear")
plt.legend()
plt.xticks(range(4),variables+['int'])
plt.show()


print("Lin MAE",np.mean(linmaes))
print("Euc MAE",np.mean(eucmaes))

plt.hist(linmaes,alpha=.5)
plt.hist(eucmaes,alpha=.5)
plt.show()

 
   

    '''
    
    
    
    
    
    