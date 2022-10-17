
import pandas as pd
import numpy as np
from random import sample
import matplotlib.pyplot as plt
from random import choices

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

dat

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

df['int']=1

df.dropna(inplace=True)


train=sample(list(df.index),int(len(df.index)*.8))
train.sort()
test=[i for i in df.index if i not in train]

df['log_mpg']=np.log(df['mpg'])

kpi='mpg'


feats=['cyl', 'disp', 'hp', 'weight', 'acc', 'yr', 'origin','int']

xt=df[df.index.isin(train)][feats].copy()
yt=df[df.index.isin(train)][kpi]

xtest=df[df.index.isin(test)][feats].copy()
ytest=df[df.index.isin(test)][kpi]

coefs = np.linalg.pinv(xt.T@xt)@(xt.T@yt)

yfit=xt@coefs
ypred=xtest@coefs


plt.scatter(ytest,ypred)
plt.scatter(yt,yfit)
plt.title(kpi)
plt.show()

r2=1-sum((ytest-ypred)**2)/sum((ytest-np.mean(ytest))**2)
print("Linear Model")
print(r2)

if kpi == 'log_mpg':
    r2=1-sum((np.exp(ytest)-np.exp(ypred))**2)/sum((np.exp(ytest)-np.mean(np.exp(ytest)))**2)
    print(r2)
        
    plt.scatter(np.exp(ytest),np.exp(ypred))
    plt.scatter(np.exp(yt),np.exp(yfit))
    plt.title(kpi)
    plt.show()


featsdf=pd.DataFrame()
featsdf['feats']=columns[1:]+['int']
featsdf['coef']=coefs

print(featsdf)





def model_fixed_coefs(fixed_columns,
                      fixed_coefs_possible,
                      x,y):

    remaining_columns = [i for i in x.columns if i not in fixed_columns]
    
    firstpred=x[fixed_columns]@np.array(fixed_coefs_possible).T
    
    firstresid=y[:,None]-np.array(firstpred)
    
    xtr=x[remaining_columns].copy()
    remaining_coefs=np.linalg.pinv(xtr.T@xtr)@(xtr.T@firstresid)
    
    
    feat_df=pd.DataFrame()
    feat_df['feats']=fixed_columns
    feat_df[['coef'+str(i) for i in range(len(fixed_coefs_possible))]]=pd.DataFrame(fixed_coefs_possible).T
    
    feat_dfb=pd.DataFrame()
    feat_dfb['feats']=remaining_columns
    feat_dfb[['coef'+str(i) for i in range(len(fixed_coefs_possible))]]=pd.DataFrame(remaining_coefs)
    
    feat_df=feat_df.append(feat_dfb).reset_index(drop=True)

    return feat_df


def cross_validate_possible_coefs(fixed_columns,
                                  fixed_coefs_possible,
                                  xt,yt,nfolds):
    
    folds = pd.Series(choices(list(range(1,nfolds+1)),k=len(xt)),index=xt.index)

    r2s_all=[]
    for fold in range(1,nfolds+1):
        
        xtt=xt[folds!=fold].copy()
        xtv=xt[folds==fold].copy()
        
        ytt=yt[folds!=fold].copy()
        ytv=yt[folds==fold].copy()
    
        feat_df=model_fixed_coefs(fixed_columns=fixed_columns,
                              fixed_coefs_possible=fixed_coefs_possible,
                              x=xtt,y=ytt)
    
        preds=xtv[list(feat_df.feats)]@np.array(feat_df[[i for i in feat_df if 'coef' in i]])
    
        r2_all=1-np.sum((ytv[:,None]-preds)**2,axis=0)/np.sum((ytv-np.mean(ytv))**2,axis=0)
        
        r2s_all.append(list(r2_all))
        
    
    cv_r2s=np.mean(r2s_all,axis=0)
    
    whichmax=np.argmax(cv_r2s)
    
    fixed_coefs_selection=fixed_coefs_possible[whichmax]
    
    return fixed_coefs_selection,cv_r2s


fixed_columns=['int','acc']
fixed_coefs_possible=[[8,-.5],[5,-1.5],[2,1]]
#fixed_coefs_possible=[[i] for i in np.linspace(8,15,90)]

fixed_coefs_selection,cv_r2s= cross_validate_possible_coefs(fixed_columns=fixed_columns,
                                  fixed_coefs_possible=fixed_coefs_possible,
                                  xt=xt,
                                  yt=yt,
                                  nfolds=10)

feat_df=model_fixed_coefs(fixed_columns=fixed_columns,
                      fixed_coefs_possible=[fixed_coefs_selection],
                      x=xt,y=yt)


preds=xtest[list(feat_df.feats)]@np.array(feat_df[[i for i in feat_df if 'coef' in i]])

fits=xt[list(feat_df.feats)]@np.array(feat_df[[i for i in feat_df if 'coef' in i]])

r2=1-np.sum((ytest[:,None]-preds)**2,axis=0)/np.sum((ytest-np.mean(ytest))**2,axis=0)
print("Fixed Coef Model")
print(r2)
print(feat_df)
plt.scatter(ytest,preds[0])
plt.scatter(yt,fits[0])
plt.show()








feat_df=model_fixed_coefs(fixed_columns=['int','acc'],
                      fixed_coefs_possible=[[10,1.5]],
                      x=xt,y=yt)

preds=xtest[list(feat_df.feats)]@np.array(feat_df[[i for i in feat_df if 'coef' in i]])

fits=xt[list(feat_df.feats)]@np.array(feat_df[[i for i in feat_df if 'coef' in i]])

r2=1-np.sum((ytest[:,None]-preds)**2,axis=0)/np.sum((ytest-np.mean(ytest))**2,axis=0)
print("Fixed Coef Model")
print(r2)
print(feat_df)
plt.scatter(ytest,preds[0])
plt.scatter(yt,fits[0])
plt.show()



feat_df=model_fixed_coefs(fixed_columns=[],
                      fixed_coefs_possible=[[]],
                      x=xt,y=yt)

preds=xtest[list(feat_df.feats)]@np.array(feat_df[[i for i in feat_df if 'coef' in i]])

fits=xt[list(feat_df.feats)]@np.array(feat_df[[i for i in feat_df if 'coef' in i]])

r2=1-np.sum((ytest[:,None]-preds)**2,axis=0)/np.sum((ytest-np.mean(ytest))**2,axis=0)
print("Fixed Coef Model")
print(r2)
print(feat_df)
plt.scatter(ytest,preds[0])
plt.scatter(yt,fits[0])
plt.show()