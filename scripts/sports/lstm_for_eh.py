







# lstm model
from numpy import mean
from numpy import std
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout,LSTM
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import StratifiedKFold,train_test_split,GroupShuffleSplit

from tensorflow.keras.layers import TimeDistributed
import pandas as pd
import numpy as np
from random import sample

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





week1_frame_features = pd.read_csv("week1_frame_features.csv")

#concat id columns and group by to split train test game,play combinations since playid is not unique but the combination is
week1_frame_features['gm_pl'] = week1_frame_features[['gameId', 'playId']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
week1_frame_features = week1_frame_features[['gm_pl','frameId','coverage']+variable_feats]
#ensure datat is ordered before shaping
week1_frame_features = week1_frame_features.sort_values(["gm_pl","frameId"])



pred_col = 'coverage'
#encode labels and select first subset of features
le = LabelEncoder()

week1_frame_features.drop_duplicates(inplace=True)


week1_frame_features["pred_code"] = le.fit_transform(week1_frame_features[pred_col].astype(str))
pred_col_code = "pred_code"

train_inds, test_inds = next(GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 7).split(week1_frame_features, groups=week1_frame_features['gm_pl']))

gm_pl = list(week1_frame_features['gm_pl'].unique())

test_gm_pl=sample(gm_pl,int(len(gm_pl)*.2))
train_gm_pl = [i for  i in gm_pl if i not in test_gm_pl]
val_gm_pl = sample(train_gm_pl,int(len(gm_pl)*.1))
train_gm_pl = [i for  i in train_gm_pl if i not in val_gm_pl]

train_X =np.array([np.array(df[1].drop(["gm_pl","frameId",pred_col,pred_col_code],axis=1).values) for df in week1_frame_features[week1_frame_features['gm_pl'].isin(train_gm_pl)].groupby("gm_pl")])
test_X =np.array([np.array(df[1].drop(["gm_pl","frameId",pred_col,pred_col_code],axis=1).values) for df in week1_frame_features[week1_frame_features['gm_pl'].isin(test_gm_pl)].groupby("gm_pl")])
val_X =np.array([np.array(df[1].drop(["gm_pl","frameId",pred_col,pred_col_code],axis=1).values) for df in week1_frame_features[week1_frame_features['gm_pl'].isin(val_gm_pl)].groupby("gm_pl")])


all_y = pd.DataFrame(to_categorical(week1_frame_features["pred_code"]),index=week1_frame_features.index)

all_y['gm_pl']=week1_frame_features['gm_pl']

train_y=np.array(all_y[all_y['gm_pl'].isin(train_gm_pl)][range(len(week1_frame_features['pred_code'].unique()))])
test_y=np.array(all_y[all_y['gm_pl'].isin(test_gm_pl)][range(len(week1_frame_features['pred_code'].unique()))])
val_y=np.array(all_y[all_y['gm_pl'].isin(val_gm_pl)][range(len(week1_frame_features['pred_code'].unique()))])

"""
train_y = to_categorical(week1_frame_features[week1_frame_features['gm_pl'].isin(train_gm_pl)]["pred_code"])
test_y = to_categorical(week1_frame_features[week1_frame_features['gm_pl'].isin(test_gm_pl)]["pred_code"])
val_y = to_categorical(week1_frame_features[week1_frame_features['gm_pl'].isin(val_gm_pl)]["pred_code"])
"""
#np.stack(train_X).shape
train_X=np.stack(train_X)
test_X=np.stack(test_X)
val_X=np.stack(val_X)

train_y = train_y.reshape(train_X.shape[0],train_X.shape[1],train_y.shape[1])
test_y = test_y.reshape(test_X.shape[0],test_X.shape[1],test_y.shape[1])
val_y = val_y.reshape(val_X.shape[0],val_X.shape[1],val_y.shape[1])

val_y.shape
test_y.shape
train_y.shape

verbose, epochs, batch_size = 1, 200, 1
n_timesteps, n_features, n_outputs = train_X[0].shape[0], train_X[0].shape[1], train_y.shape[2]
model = Sequential()
model.add(LSTM(8,dropout=.15))
#model.add(Dropout(0.5))
#model.add(Dense(15, activation='sigmoid'))
#model.add(Dense(n_outputs, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#train_y = train_y.reshape(train_y.shape[0]*train_y.shape[1],train_y.shape[2])
#train_y.shape
# fit network    .reshape(train_X.shape[0],train_X.shape[1],train_X.shape[2])
model.fit(train_X, 
          train_y, 
          validation_data=(val_X,val_y),
          epochs=epochs, 
          batch_size=batch_size, 
          verbose=verbose)

model.summary()

train_y.shape
# evaluate model
_, accuracy = model.evaluate(train_X, test_y, batch_size=batch_size, verbose=0)







train_X =np.array(week1_frame_features[week1_frame_features['gm_pl'].isin(train_gm_pl)][variable_feats])
test_X =np.array(week1_frame_features[week1_frame_features['gm_pl'].isin(test_gm_pl)][variable_feats])
val_X =np.array(week1_frame_features[week1_frame_features['gm_pl'].isin(val_gm_pl)][variable_feats])


all_y = pd.DataFrame(to_categorical(week1_frame_features["pred_code"]),index=week1_frame_features.index)

all_y['gm_pl']=week1_frame_features['gm_pl']

train_y=np.array(all_y[all_y['gm_pl'].isin(train_gm_pl)][range(len(week1_frame_features['pred_code'].unique()))])
test_y=np.array(all_y[all_y['gm_pl'].isin(test_gm_pl)][range(len(week1_frame_features['pred_code'].unique()))])
val_y=np.array(all_y[all_y['gm_pl'].isin(val_gm_pl)][range(len(week1_frame_features['pred_code'].unique()))])

train_X.shape



verbose, epochs, batch_size = 1, 200, 1
n_timesteps, n_features, n_outputs = train_X.shape[0], train_X.shape[1], train_y.shape[1]
model = Sequential()
model.add(LSTM(100,dropout=.15))
model.add(Dropout(0.5))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(n_outputs, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#train_y = train_y.reshape(train_y.shape[0]*train_y.shape[1],train_y.shape[2])
#train_y.shape
# fit network    .reshape(train_X.shape[0],train_X.shape[1],train_X.shape[2])
model.fit(train_X.reshape(train_X.shape[0],train_X.shape[1],1), 
          train_y, 
          validation_data=(val_X.reshape(val_X.shape[0],val_X.shape[1],1),
                           val_y),
          epochs=epochs, 
          batch_size=batch_size, 
          verbose=verbose)

