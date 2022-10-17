

import pandas as pd

import os
import numpy as np

files = os.listdir()

files = [i for i in files if ("Premier_League" in i)&("_b" in i)]


dat = pd.DataFrame()
for i in files:
    temp = pd.read_csv(i)
    dat = dat.append(temp)
    
dat['date'] = pd.to_datetime(dat['date'])

dat.sort_values(by=['date'],ascending=True,inplace=True)
dat.reset_index(inplace=True,drop=True)

dat['homewin']=np.where(dat['team_home_90min_goals']>dat['team_away_90min_goals'],1,0)
dat['awaywin']=np.where(dat['team_home_90min_goals']<dat['team_away_90min_goals'],1,0)
dat['draw']=np.where(dat['team_home_90min_goals']==dat['team_away_90min_goals'],1,0)

seasons = dat['seasonName'].unique()

def process_seasons(season_name,dat):
    teststr = """
        home_record = dat[dat['seasonName'].str.contains("1999/2000")].groupby(
                ['homeName'])[['team_home_90min_goals','team_away_90min_goals','homewin','awaywin','draw']].cumsum()
        home_record['team'] = dat['homeName']
        home_record['date'] = dat['date']
        home_record.columns=['gf','ga','wins','losses','draws','team','date']
        
        
        
        away_record = dat[dat['seasonName'].str.contains("1999/2000")].groupby(
                ['awayName'])[['team_away_90min_goals','team_home_90min_goals','awaywin','homewin','draw']].cumsum()
        away_record['team'] = dat['awayName']
        away_record['date'] = dat['date']
        away_record.columns=['gf','ga','wins','losses','draws','team','date']
        season_name = seasons[0]
        """
    
    home_game = dat[dat['seasonName']==season_name][
            ['homeName','date','team_home_90min_goals','team_away_90min_goals','homewin','awaywin','draw']].copy()
    home_game.columns=['team','date','gf','ga','wins','losses','draws']
    
    
    away_game = dat[dat['seasonName']==season_name][
            ['awayName','date','team_away_90min_goals','team_home_90min_goals','awaywin','homewin','draw']].copy()
    away_game.columns=['team','date','gf','ga','wins','losses','draws']

    
    season = home_game.append(away_game)
    season.sort_values(by='date',ascending=True,inplace=True)
    season.reset_index(inplace=True,drop=True)
    
    form = season.groupby(['team'])[['wins','losses','draws']].rolling(window=5,min_periods=0).sum().reset_index()
    form.set_index('level_1',inplace=True)
    form.sort_values(by='level_1',inplace=True)
    form['l5_points'] = form['wins']*3+form['draws']
    
    cumeseason = season.groupby(['team'])[['gf','ga','wins','losses','draws']].cumsum()
    cumeseason['l5_wins'] = form['wins']
    cumeseason['l5_losses'] = form['losses']
    cumeseason['l5_draws'] = form['draws']
    cumeseason['l5_points'] = form['l5_points']
    cumeseason['team']=season['team']
    
    cumeseason['points'] = 3*cumeseason['wins']+1*cumeseason['draws']
    
    cumeseason['round'] = cumeseason.groupby(['team']).cumcount()+1

    cumeseason['season'] = season_name
    
    check = """
    cumeseason[cumeseason['team']=='Liverpool']
    
    dat[(dat['seasonName']==season_name)&
        ((dat['homeName']=='Liverpool')|(dat['awayName']=='Liverpool'))].reset_index()
    
    
    dat[(dat['seasonName']==season_name)&
        ((dat['round']==28))].reset_index()
        
    season[season['team']=='Liverpool'].reset_index()
    home_game[home_game['team']=='Liverpool'].reset_index()
    away_game[away_game['team']=='Liverpool'].reset_index()"""
    
    
    return cumeseason

season_name = seasons[-2]

cumeseasons = pd.DataFrame()
for  season_name in seasons:
    temp = process_seasons(season_name,dat)
    cumeseasons=cumeseasons.append(temp)

cumeseasons['gd']=cumeseasons['gf']-cumeseasons['ga']
cumeseasons.sort_values(by=['points','gd'],ascending=[False,False],inplace=True)


cumeseasons['place']=cumeseasons.groupby(['season','round']).cumcount()+1

cumeseasons['relegation'] = np.where(cumeseasons['place']>17,1,0)

cumeseasons[(cumeseasons['team']=='Liverpool')&(cumeseasons['season'].str.contains("2019/2020"))]
cumeseasons[(cumeseasons['round']==25)&
            (cumeseasons['season'].str.contains("2019/2020"))]

cumeseasons[(cumeseasons['round']==38)&
            (cumeseasons['season'].str.contains("2019/2020"))]

dat[(dat['seasonName']=='Premier League - 2019/2020')&
    ((dat['homeName']=='Liverpool')|(dat['awayName']=='Liverpool'))]


week14=cumeseasons[cumeseasons['round']==14].copy()
week10=cumeseasons[cumeseasons['round']==10].copy()


week33=cumeseasons[cumeseasons['round']==33].copy()

week14['key']=week14['team']+" "+week14['season']
week10['key']=week10['team']+" "+week10['season']


week33['key']=week33['team']+" "+week33['season']

week10=week10[['key','gf','ga','gd','points','place','wins','losses','draws','l5_wins','l5_losses','l5_draws','l5_points']]
week14=week14[['key','gf','ga','gd','points','place','wins','losses','draws','l5_wins','l5_losses','l5_draws','l5_points']]

week33=week33[['key','relegation']]

week10.columns = ['key']+['week_10_'+i for i in week10.columns if i!='key']
week14.columns = ['key']+['week_14_'+i for i in week14.columns if i!='key']

M = week10.merge(week14,on=['key'],how='left')
M=M.merge(week33,on=['key'],how='left')

M['int']=1



val = M[
        (M['key'].str.contains('2019'))|
        (M['key'].str.contains('2018'))#|
       # (M['key'].str.contains('2017'))
        ].copy()

train = M[~(
        (M['key'].str.contains('2019'))|
        (M['key'].str.contains('2018'))#|
       # (M['key'].str.contains('2017'))
        )].copy()

train=train[~train['key'].str.contains('2020')].copy()

ars = M[(M['key'].str.contains('Arsenal'))&(M.key.str.contains("2020/2021"))].copy()

np.mean(M[(M['week_14_l5_wins']==0)&
          (M['week_14_l5_points']<=1)&
          (M['week_14_points']<=14)&
          (~M['key'].str.contains('2021'))]['relegation'])

val.corr()['relegation'].sort_values()

features=['week_10_gf', 'week_10_ga', 'week_10_gd', 'week_10_points',
       'week_10_place', 'week_10_wins', 'week_10_losses', 'week_10_draws',
       'week_10_l5_wins', 'week_10_l5_losses', 'week_10_l5_draws',
       'week_10_l5_points', 'week_14_gf', 'week_14_ga', 'week_14_gd',
       'week_14_points', 'week_14_place', 'week_14_wins',
       'week_14_losses', 'week_14_draws', 'week_14_l5_wins',
       'week_14_l5_losses', 'week_14_l5_draws', 'week_14_l5_points','int']

features=['week_10_l5_losses','week_14_place','week_10_place','week_14_losses','week_10_losses','week_10_ga','int']
kpi = 'relegation'

xt=train[features].copy()
xv=val[features].copy() 
xtest = ars[features].copy()

yt = train[kpi]
yv = val[kpi]

yt_odds = yt/(1-yt)
yt_odds=np.where(yt_odds>=1,1e6,1e-6)
yt_log_odds = np.log(yt_odds)


yv_odds = yv/(1-yv)
yv_odds=np.where(yv_odds>=1,1e6,1e-6)
yv_log_odds = np.log(yv_odds)


#logistic

coefs = np.linalg.pinv(xt.T@xt)@(xt.T@yt_log_odds)


yt_log_odds_pred = xt@coefs
yv_log_odds_pred = xv@coefs
ytest_log_odds_pred = xtest@coefs

yt_odds_pred=np.exp(yt_log_odds_pred)
yv_odds_pred=np.exp(yv_log_odds_pred)
ytest_odds_pred=np.exp(ytest_log_odds_pred)

yt_pred = yt_odds_pred/(1+yt_odds_pred)
yv_pred = yv_odds_pred/(1+yv_odds_pred)
ytest_pred = ytest_odds_pred/(1+ytest_odds_pred)
plt.hist(yt_pred)

resultsv = pd.DataFrame()
resultsv['y_pred']=np.where(yv_pred>.33,1,0)
resultsv['y']=yv.values



resultst = pd.DataFrame()
resultst['y_pred']=np.where(yt_pred>.33,1,0)
resultst['y']=yt.values


resultst['accuracy']=np.where((resultst['y']==0)&(resultst['y_pred']==0),'tn',
         np.where((resultst['y']==1)&(resultst['y_pred']==1),'tp',
         np.where((resultst['y']==0)&(resultst['y_pred']==1),'fp',
         np.where((resultst['y']==1)&(resultst['y_pred']==0),'fn','other'))))

resultsv['accuracy']=np.where((resultsv['y']==0)&(resultsv['y_pred']==0),'tn',
         np.where((resultsv['y']==1)&(resultsv['y_pred']==1),'tp',
         np.where((resultsv['y']==0)&(resultsv['y_pred']==1),'fp',
         np.where((resultsv['y']==1)&(resultsv['y_pred']==0),'fn','other'))))



resultst.groupby(['accuracy']).count()
resultsv.groupby(['accuracy']).count()


val['pred']=yv_pred
val['pred']=np.where(val['pred']>=.33,1,0)
val[val['pred']==1]

ars
#sklearn logistic

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(fit_intercept=False).fit(xt,yt)
yv_pred = lr.predict(xv)
yt_pred = lr.predict(xt)


resultst = pd.DataFrame()
resultst['y_pred']=yt_pred
resultst['y']=yt.values
resultst['accuracy']=np.where((resultst['y']==0)&(resultst['y_pred']==0),'tn',
         np.where((resultst['y']==1)&(resultst['y_pred']==1),'tp',
         np.where((resultst['y']==0)&(resultst['y_pred']==1),'fp',
         np.where((resultst['y']==1)&(resultst['y_pred']==0),'fn','other'))))

resultst.groupby(['accuracy']).count()


resultsv = pd.DataFrame()
resultsv['y_pred']=yv_pred
resultsv['y']=yv.values
resultsv['accuracy']=np.where((resultsv['y']==0)&(resultsv['y_pred']==0),'tn',
         np.where((resultsv['y']==1)&(resultsv['y_pred']==1),'tp',
         np.where((resultsv['y']==0)&(resultsv['y_pred']==1),'fp',
         np.where((resultsv['y']==1)&(resultsv['y_pred']==0),'fn','other'))))

resultsv.groupby(['accuracy']).count()


#Tree
from sklearn.tree import DecisionTreeRegressor

rfc = DecisionTreeRegressor().fit(xt,yt)
yv_pred = rfc.predict(xv)
yt_pred = rfc.predict(xt)
ytest_pred = rfc.predict(xtest)


resultst = pd.DataFrame()
resultst['y_pred']=yt_pred
resultst['y']=yt.values
resultst['accuracy']=np.where((resultst['y']==0)&(resultst['y_pred']==0),'tn',
         np.where((resultst['y']==1)&(resultst['y_pred']==1),'tp',
         np.where((resultst['y']==0)&(resultst['y_pred']==1),'fp',
         np.where((resultst['y']==1)&(resultst['y_pred']==0),'fn','other'))))

resultst.groupby(['accuracy']).count()


resultsv = pd.DataFrame()
resultsv['y_pred']=yv_pred
resultsv['y']=yv.values
resultsv['accuracy']=np.where((resultsv['y']==0)&(resultsv['y_pred']==0),'tn',
         np.where((resultsv['y']==1)&(resultsv['y_pred']==1),'tp',
         np.where((resultsv['y']==0)&(resultsv['y_pred']==1),'fp',
         np.where((resultsv['y']==1)&(resultsv['y_pred']==0),'fn','other'))))

resultsv.groupby(['accuracy']).count()


importance = pd.DataFrame()
importance['imp'] = rfc.feature_importances_
importance['features'] = features
importance.sort_values(by='imp',ascending=False)


val['pred']=yv_pred
val[val['pred']==1]
ars

#linear
coefs = np.linalg.pinv(xt.T@xt)@(xt.T@yt)



yt_pred = xt@coefs
yv_pred = xv@coefs
ytest_pred = xtest@coefs

val['pred2']=yv_pred
val[val['pred']==1]

import matplotlib.pyplot as plt 
plt.scatter(yt,yt_pred)


plt.scatter(yv,yv_pred)

