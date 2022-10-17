
import pandas as pd
import numpy as np

pd.set_option("display.max_columns",20)
dat=pd.read_csv("info.csv")


dat.sort_values(by='upside',ascending=False).head(30)

list(dat.columns)

target_columns = [
 'symbol','shortName','sector',
 'upside',
 'recommendationMean',
 'numberOfAnalystOpinions',
 'currentPrice',
 'targetLowPrice',
 'targetMeanPrice',
 'targetMedianPrice',
 'targetHighPrice',
 ]

tab = dat[
    target_columns].copy()

conditions= (tab['upside']>=.05)&\
    (tab['upside']<=4)&\
    (tab['numberOfAnalystOpinions']>=5)
    
    
    
tab[conditions].sort_values(by='upside',ascending=False).head(30)

