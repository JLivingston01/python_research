
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 150)
import matplotlib.pyplot as plt
import datetime as dt

dat = pd.read_csv('horsing_dat.csv')

def featurize(dat):
        
    
    dat = dat[~(dat['ML'].isna())].copy()
    dat=dat[~dat['ML'].str.contains('AE')]
    dat=dat[~dat['ML'].str.contains('SCR')]
    dat=dat[~dat['ML'].str.contains('MTO')]
    dat = dat[~dat['Sire'].isna()].copy()
    dat = dat[~(dat['ML'].str.contains('Sire of Kentucky'))].copy()

    dat[['num','denom']]=dat['ML'].str.split("/",expand=True)
    
    
    dat['num'] = dat['num'].astype(float)
    dat['denom'] = dat['denom'].astype(float)
    dat['odds'] = dat['num']/dat['denom']
    
    
    dat['ml_prob']=1/(1+dat['odds'])
    
    dat['PLACE'].fillna(5,inplace=True)
    
    dat['WIN']=np.where(dat['PLACE']==1,1,0)
    
    def calc_win_perc(dat , by='Runner'):
        dat[f'{by}_RACES']=dat.groupby([by])['WIN'].cumcount()+1
        dat[f'{by}_WINS']=dat.groupby([by])['WIN'].cumsum()
        
        dat[f'{by}_RACES']=dat.groupby([by])[f'{by}_RACES'].shift(1)
        dat[f'{by}_WINS']=dat.groupby([by])[f'{by}_WINS'].shift(1)
        dat[f'{by}_WIN_PERC']=dat[f'{by}_WINS']/dat[f'{by}_RACES']
        
        return dat
    
    dat = calc_win_perc(dat , by='Runner')
    dat = calc_win_perc(dat , by='Sire')
    dat = calc_win_perc(dat , by='Trainer')
    dat = calc_win_perc(dat , by='Jockey')
    
    racegroup = ['TRACK','RACENUM','DATE']
    bestodds = dat.groupby(racegroup).agg(
        {'ml_prob':np.nanmin}).reset_index().fillna(0)
    bestodds.columns = ['TRACK','RACENUM','DATE','BESTPROB']
    dat = dat.merge(bestodds,
              on=racegroup,
              how='left')
    dat['ml_prob_delta']=dat['ml_prob'] - dat['BESTPROB']
    
    def best_in_group(dat,col):
        bestodds = dat.groupby(racegroup).agg(
            {f'{col}':np.nanmax}).reset_index().fillna(0)
        bestodds.columns = ['TRACK','RACENUM','DATE',f'BEST{col}']
        dat = dat.merge(bestodds,
                  on=racegroup,
                  how='left')
        dat[f'{col}_delta']=dat[col] - dat[f'BEST{col}']
        
        return dat
    
    dat = best_in_group(dat,'Runner_WIN_PERC')
    dat = best_in_group(dat,'Sire_WIN_PERC')
    dat = best_in_group(dat,'Trainer_WIN_PERC')
    dat = best_in_group(dat,'Jockey_WIN_PERC')
    
    
    return dat
dat = featurize(dat)
dat.corr()['WIN']

dat[dat['Runner']=="Dreamer's Moon"]


kpi = 'WIN'

feats = (
    ['ml_prob','ml_prob_delta']+
    [i for i in dat.columns if ('_WIN_PERC' in i)&('BEST' not in i)]
    )


corrs = dat.corr()['WIN'].sort_values()
corrs2 = corrs[abs(corrs)>=.1].copy()
selected = [i for i in corrs2.index if i in feats]

train_cut = '2022-06-01'
train_start = '2020-06-01'
#train_cut = dt.datetime.strftime(dt.date.today()-dt.timedelta(5),'%Y-%m-%d')
train_records=(dat['DATE']<train_cut)
pre_train_records=(dat['DATE']<train_start)


xt = dat[(~pre_train_records)&(train_records)][feats].fillna(0).copy()
xv = dat[(~pre_train_records)&(~train_records)][feats].fillna(0).copy()

yt = dat[(~pre_train_records)&(train_records)][kpi].copy()
yv = dat[(~pre_train_records)&(~train_records)][kpi].copy()



"""
from sklearn.ensemble import StackingRegressor

from sklearn.linear_model import TweedieRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

mod = StackingRegressor(estimators=[
    ('GLM',TweedieRegressor(power=1,link='log',max_iter=1000)),
    ('RF',RandomForestRegressor(random_state=42,
                                n_estimators=50,
                                max_depth=15,
                                n_jobs=-1,
                                bootstrap=True,
                                max_samples=None,
                                max_features=None,
                                verbose=2
                                )),
    ('LM',LinearRegression())
    ],
    final_estimator=RandomForestRegressor(n_estimators=50)
    )

mod.fit(xt,yt)
"""

#from sklearn.linear_model import LogisticRegression

#mod = LogisticRegression(max_iter=10000,tol=1e-12).fit(xt,yt)

#yfit = pd.Series(np.argmax(mod.predict_proba(xt),axis=1),index=xt.index)
#ypred = pd.Series(np.argmax(mod.predict_proba(xv),axis=1),index=xv.index)

#from xgboost import XGBRegressor

#mod = XGBRegressor(random_state=42,
#                   n_estimators=200,
#                   max_depth=3,
#                   ).fit(xt,yt)

'''

mod = RandomForestRegressor(random_state=42,
                            n_estimators=50,
                            max_depth=15,
                            n_jobs=-1,
                            bootstrap=True,
                            max_samples=None,
                            max_features=None,
                            verbose=2
                            ).fit(xt,yt)
'''
'''
from sklearn.tree import DecisionTreeRegressor

mod = DecisionTreeRegressor(max_depth=5,
                            min_samples_leaf=10,
                            random_state=42
                            ).fit(xt,yt)
'''


from sklearn.neural_network import MLPRegressor

mod = MLPRegressor(hidden_layer_sizes=(100,25,),activation = 'logistic',
                   max_iter=1000,verbose=True,n_iter_no_change=100,
                   random_state=42,tol=1e-6).fit(xt,yt)

yfit=mod.predict(xt)
ypred=mod.predict(xv)

results = pd.DataFrame({'true':yv,
                        'pred':ypred})

results[['TRACK','DATE','RACENUM','Runner','ml_prob','odds']]=dat[~train_records][
    ['TRACK','DATE','RACENUM','Runner','ml_prob','odds']]

results['PRED']=results.groupby(
    ['TRACK','DATE','RACENUM'])['pred'].rank(ascending=False)

results['NAIVE']=results.groupby(
    ['TRACK','DATE','RACENUM'])['ml_prob'].rank(ascending=False)

results['NAIVE_WIN']=np.where(results['NAIVE']==1,1,0)
results['MODEL_WIN']=np.where(results['PRED']==1,1,0)

results['NAIVE_CORRECT']=np.where((results['true']==1)&
                                  (results['NAIVE_WIN']==1),1,0)

results['MODEL_CORRECT']=np.where((results['true']==1)&
                                  (results['MODEL_WIN']==1),1,0)

race_results = results.groupby(['TRACK','DATE','RACENUM']).agg({'MODEL_CORRECT':'max',
                                                 'NAIVE_CORRECT':'max'}).reset_index()

race_results = race_results.merge(results[results['true']==1][
    ['TRACK','DATE','RACENUM','odds']],
                   on=['TRACK','DATE','RACENUM'],
                   how='left')

#race_results=race_results[~race_results['odds'].isna()].copy()

race_results['MODEL_WINNINGS']=np.where(
    race_results['MODEL_CORRECT']==1,race_results['odds'],-1)
race_results['NAIVE_WINNINGS']=np.where(
    race_results['NAIVE_CORRECT']==1,race_results['odds'],-1)

print('model_winnings: ',np.nansum(race_results['MODEL_WINNINGS']),
      'naive_winnings: ',np.nansum(race_results['NAIVE_WINNINGS']),
      'accuracies: ',np.mean(race_results[['MODEL_CORRECT','NAIVE_CORRECT']],axis=0),
      'model over naive: ',sum(np.where((race_results['MODEL_CORRECT']==1)&
               (race_results['NAIVE_CORRECT']==0),1,0)),
      'naive over model: ',sum(np.where((race_results['MODEL_CORRECT']==0)&
               (race_results['NAIVE_CORRECT']==1),1,0))
      )


plt.plot(race_results['MODEL_WINNINGS'].cumsum())
plt.plot(race_results['NAIVE_WINNINGS'].cumsum())
plt.show()



plt.plot(race_results.groupby(['DATE']).agg({'MODEL_CORRECT':'mean',
                                    'NAIVE_CORRECT':'mean'}))
plt.show()


### INFERENCE ###

tomorrow = dt.datetime.strftime(dt.date.today()+dt.timedelta(1),'%Y-%m-%d')

query = f"/entries-results/{tomorrow}"
base_url = f"https://entries.horseracingnation.com{query}"

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

tables = pd.read_html(base_url)

t = tables[0]

tracks = list(t['Track'].str.lower().str.replace(
    " ","-").str.replace("---","-").str.replace("'","-")
    )
tracks = [
    ''.join([i for i in j if (i.isalnum())|(i=="-")]) for j in tracks]

#track = tracks[0]
out = pd.DataFrame()
for track in tracks:
    query = f"/entries-results/{track}/{tomorrow}"
    base_url = f"https://entries.horseracingnation.com{query}"

    tables = pd.read_html(base_url)

    runner_tables = [i for i in tables if ('PP' in i.columns)&('ML' in i.columns)]
    for i in range(len(runner_tables)):
        runner_tables[i]['RACENUM']=i+1
        
    runners = pd.concat(runner_tables)
    runners['DATE'] = tomorrow
    runners['TRACK']=track

    
    runners[['Runner','Sire']]=runners['Horse / Sire'].str.split("  ",expand=True)
    runners[['Trainer','Jockey']]=runners['Trainer / Jockey'].str.split("  ",expand=True)
    
    print(track)
    out = out.append(runners)


inference = dat.append(out)[
    list(out.columns)+['PLACE']]




inference = featurize(inference)

inference = inference[inference['DATE']==tomorrow].copy()


xtest = inference[feats].fillna(0).copy()


yout=pd.DataFrame({'TRACK':inference['TRACK'],
                   'RACENUM':inference['RACENUM'],
                   'DATE':inference['DATE'],
                   'Runner':inference['Runner'],
                   'odds':inference['odds'],
                   'ml_prob':inference['ml_prob'],
                   'pred':mod.predict(xtest)})



yout['PRED']=yout.groupby(
    ['TRACK','DATE','RACENUM'])['pred'].rank(ascending=False)

yout['NAIVE']=yout.groupby(
    ['TRACK','DATE','RACENUM'])['ml_prob'].rank(ascending=False)


yout[yout['PRED']==1].to_csv(f"horse_predictions_{tomorrow}.csv",index=False)


len(yout[yout['PRED']==1])






