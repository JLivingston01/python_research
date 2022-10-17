

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


dat=pd.read_csv("c:/users/jliv/downloads/HOF_players_and_dwight.csv")

pd.set_option("display.max_columns",15)


metrics = ['G','PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG%', '3P%',
       'FT%', 'WS', 'WS/48']



dwight=dat[dat['Name']=='Dwight Howard'].copy()
hof=dat[dat['Name']!='Dwight Howard'].copy()

medians=list(np.nanmedian(hof[metrics],axis=0))

medians_map={metrics[i]:medians[i] for i in range(len(metrics))}

for i in metrics:
    hof[i].fillna(medians_map[i],inplace=True)

mu_hof=np.mean(hof[metrics],axis=0)
sd_hof=np.std(hof[metrics],axis=0)

hof_std=(hof[metrics]-mu_hof)/sd_hof

dwight_std=(dwight[metrics]-mu_hof)/sd_hof


dists = cdist(dwight_std[metrics],hof_std[metrics],metric='euclidean')

hof['dists']=dists[0]

hof.sort_values(by='dists',ascending=True).head(10)
dwight