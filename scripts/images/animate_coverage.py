

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt



import seaborn as sns
from matplotlib import animation


import os
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')

pd.set_option("display.max_columns",500)

#split features for constant vs variable
id_cols = ['gameId', 'playId', 'frameId'] 
pred_col = "coverage"
invariate_feats = ['CB','DB','DE','DL','FB','FS','HB','ILB','K','LB','LS','MLB','OL','OLB','P','QB','RB','S','SS','TE','WR',
 'quarter','down','yardsToGo','defendersInTheBox', 'numberOfPassRushers',"offenseFormation",
 'Dropback_TRADITIONAL','gameClock_sec','yds_ez', 'all_DBs','count_CB_DB','count_LB','count_RB','count_Rec','count_S',
 'def_pos_count','diff_def_pos_count_DB','off_pos_count'
]

variable_feats = ["defendersMovingBack","defendersWatchingQB",'defendersDeep','defendersDeepDeep','defendersInTheBox_atSnap','defendersNearLOS','defendersNearSide','defendersfarSide',
 'offBehindQB','offFarSide','offFrontQB','offInTheBox','offNearLOS','offNearSide',
 'def_backs_grp_avg_dist_los', 'line_backs_grp_avg_dist_los','other_def_grp_avg_dist_los', 'other_off_grp_avg_dist_los','rbs_grp_avg_dist_los','recievers_grp_avg_dist_los',
 'max_safety_dist_los', 'min_safety_dist_los', 'safety_maxmin_diff_dist_los', 'safeties_grp_avg_dist_los'
]





plays = pd.read_csv("plays.csv")

plays['gm_pl'] = plays[['gameId', 'playId']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)


weeks = pd.DataFrame()

filelist = [i for  i in os.listdir() if 'week' in i]


for  i in ['week1.csv',
 'week10.csv',
 'week11.csv',
 'week12.csv',
 'week13.csv',
 'week14.csv',
 'week15.csv',
 'week16.csv',
 'week17.csv',
 'week2.csv',
 'week3.csv',
 'week4.csv',
 'week5.csv',
 'week6.csv',
 'week7.csv',
 'week8.csv',
 'week9.csv']:
    weeks=weeks.append(pd.read_csv(i)[['x','y','event','position','frameId','gameId','playId','displayName']])


weeks['gm_pl'] = weeks[['gameId', 'playId']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
#ensure datat is ordered before shaping
weeks = weeks.sort_values(["gm_pl","frameId"])


weeks["D_O"]=np.where(weeks.position.isin(['CB','SS','MLB','OLB','ILB','LB','FB','DL','DB','NT','DE','S','FS']),"D",
     np.where(weeks.position.isin(['QB','WR','RB','TE','HB']),"O",np.nan))

weeks = weeks.merge(plays[['gm_pl','offenseFormation']],on='gm_pl',how='left')

#week1['xhat'] = np.where(week1['x']<=60,week1['x']-10,110-week1['x'])
weeks = weeks.merge(plays[['gm_pl','absoluteYardlineNumber']],on='gm_pl',how='left')


weeks['Ball_Snap'] = np.where(weeks['event']=='ball_snap',1,0)

weeks['Ball_Snap']=weeks.groupby(['gm_pl'])['Ball_Snap'].cumsum()

weeks=weeks[weeks['Ball_Snap']>0].copy()

qb_direction = weeks[(weeks['position']=='QB')&(weeks['event']=='ball_snap')][['gm_pl','x','absoluteYardlineNumber']].drop_duplicates().copy()

qb_direction['direction']=np.where(qb_direction['x']>qb_direction['absoluteYardlineNumber'],-1,
            np.where(qb_direction['x']<qb_direction['absoluteYardlineNumber'],1,np.nan))

weeks = weeks.merge(qb_direction[['gm_pl','direction']],on='gm_pl',how='left')

weeks['xhat']=np.where(weeks['direction']==-1,weeks['x']-weeks['absoluteYardlineNumber'],
         np.where(weeks['direction']==1,weeks['absoluteYardlineNumber']-weeks['x'],np.nan))
         

ball_center = weeks[(weeks['displayName']=='Football')&(weeks['event']=='ball_snap')][['gm_pl','y']].drop_duplicates().copy()
ball_center['y_init']=ball_center['y']

weeks = weeks.merge(ball_center[['gm_pl','y_init']],on='gm_pl',how='left')

weeks['yhat'] = weeks['y']-weeks['y_init']

weeks['frame_prime'] = weeks.groupby(['gm_pl','playId','displayName'])['gm_pl'].cumcount()+1


weeks = weeks.merge(plays[['gm_pl','typeDropback']],on='gm_pl',how='left')


w1ff = pd.read_csv("all_defense_predictions (1).csv")[['gameId','playId','coverage']]

w1ff['gm_pl'] = w1ff['gameId'].astype(str)+"_"+w1ff['playId'].astype(str)


weeks = weeks.merge(w1ff[['gm_pl','coverage']],on=['gm_pl'],how='left')

weeks.position.unique()

del weeks['x']
del weeks['y']
del weeks['event']
del weeks['position']
del weeks['frameId']
del weeks['gameId']
del weeks['playId']
del weeks['displayName']
del weeks['absoluteYardlineNumber']
del weeks['Ball_Snap']
del weeks['direction']
del weeks['y_init']

#weeks[weeks['coverage']=='Prevent Zone'].position.unique()
#weeks.head()


from scipy.ndimage.filters import gaussian_filter

xlow,xhigh,ylow,yhigh=-25,5,-25,25


weeks['offenseFormation'].unique()
frame = weeks[(weeks['offenseFormation']=='SINGLEBACK')&
      (weeks['frame_prime']==1)&
      (weeks['D_O']=='D')&
      (~weeks['yhat'].isna())&(~weeks['xhat'].isna())
      ]
x=frame['xhat']
y=frame['yhat']



heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)

extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
heatmap = gaussian_filter(heatmap, sigma=1)

fig=plt.figure(figsize=(8,8))
ax = plt.axes()
ax.set_facecolor("#440154")
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.xlim(xlow,xhigh)
plt.ylim(ylow,yhigh)
plt.show()

heatmap.T.shape
#Animate

def return_heatmap_coverage(coverage='Cover 2 Man',sigma=1):
    frame = weeks[(weeks['coverage']==coverage)&
          (weeks['frame_prime']==1)&
          (weeks['D_O']=='D')&
          (~weeks['yhat'].isna())&(~weeks['xhat'].isna())
          ]
    x=frame['xhat']
    y=frame['yhat']
    fig=plt.figure(figsize=(8,8))
    ax = plt.axes()
    ax.set_facecolor("#440154")
    plt.xlim(xlow,xhigh)
    plt.ylim(ylow,yhigh)
    
    
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
    
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    
    def animate(i):
        frame = weeks[(weeks['coverage']==coverage)&
              (weeks['frame_prime']==i)&
              (weeks['D_O']=='D')&
              (~weeks['yhat'].isna())&(~weeks['xhat'].isna())
              ]
        x=frame['xhat']
        y=frame['yhat']
        
            
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
        
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        
        ax = plt.axes()
        ax.set_facecolor("#440154")
        plt.title(coverage)
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        ax.axvline(0,ymin=-25,ymax=25,color='orange')
        ax.axvline(-5,ymin=-25,ymax=25,linestyle='--')
        ax.axvline(-10,ymin=-25,ymax=25,linestyle='--')
        ax.axvline(-15,ymin=-25,ymax=25,linestyle='--')
        ax.axvline(-20,ymin=-25,ymax=25,linestyle='dashed')
        ax.axhline(y=3,xmin=-25,xmax=5,linestyle='dashed')
        ax.axhline(y=-3,xmin=-25,xmax=5,linestyle='dashed')
    
    
    anim = animation.FuncAnimation(fig, animate, frames=30, repeat = False)
    anim.save("frame_animation_"+coverage+".gif")
      

for i in ['Cover 2 Man', 'Cover 1 Man', 'Cover 3 Zone', 'Cover 4 Zone',
       'Cover 0 Man', 'Cover 2 Zone', 'Cover 6 Zone', 'Prevent Zone']:
    
    return_heatmap_coverage(coverage=i,sigma=1)

