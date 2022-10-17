
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


    
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

len(dat)

variables=  list(matchdf['dim'][:])


X = dat[variables].copy()
X['int']=1
Y=dat['income']

def entropy(x):
    t = pd.Series(x)
    term = (t.groupby(by=t).apply(lambda x: len(x))/len(t))
    return sum(-term*np.log(term))

def improvement(d1,d2,d3):
    return (len(d1)*entropy(d1['income'])-(len(d2)*entropy(d2['income'])+len(d3)*entropy(d3['income'])))/(len(d1))

dat['criteria']='parent,'
dd = dat.sample(1000)


def split(subdat,grpidinit=0,maxnode=300,minnode = 50):
    #subdat = dat.copy()
    grpid = grpidinit+3
    if (entropy(subdat['income'])==0)|(len(subdat)<maxnode):
        subdat['node']=grpid
        return subdat
    else:
        imp = []
        for i in variables:
            splt0=subdat.copy()
            splt1 = subdat[subdat[i]==0].copy()
            splt2 = subdat[subdat[i]==1].copy()
            
            imp.append(improvement(splt0,splt1,splt2))
            
            
        choice = variables[np.argmax(imp)]
        
        
        splt1 = subdat[subdat[choice]==0].copy()
        splt2 = subdat[subdat[choice]==1].copy()
        
        splt1['criteria']=splt1['criteria']+choice+str(0)+","
        splt2['criteria']=splt2['criteria']+choice+str(1)+","
        
 #       if (len(splt1)<minnode)|(len(splt2)<minnode):
            
            
        return [split(splt1,grpidinit=grpid+1,maxnode=maxnode),split(splt2,grpidinit=grpid+2,maxnode=maxnode)]
        
groups = split(dd,maxnode=100)   


output=[]
def removeNestings(l): 
    for i in l: 
        if type(i) == list: 
            removeNestings(i) 
        else: 
            output.append(i) 
            
removeNestings(groups)

output = pd.concat(output)

predtab=pd.pivot_table(output,
                       index=['node'],
                       values=['income'],
                       aggfunc='mean').reset_index().rename(mapper={'income':'pred'},
                                                  axis='columns')

ddd = pd.merge(output,predtab,on=['node'],how='left')

ddd['pred']=np.where(ddd['pred']>.5,1,0)

np.mean(np.where(ddd['pred']==ddd['income'],1,0))

entropy(dat[dat['income']==1]['income'])
