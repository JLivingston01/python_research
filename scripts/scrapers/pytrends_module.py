


import pandas as pd
import datetime as dt
from pytrends.request import TrendReq
import numpy as np
import matplotlib.pyplot as plt


pd.set_option("display.max_columns",500)

now = dt.datetime.now()

then = now-dt.timedelta(hours = 5)


months_3 = now - dt.timedelta(days = 90)

timeframe = dt.datetime.strftime(months_3,"%Y-%m-%d")+" "+dt.datetime.strftime(now,"%Y-%m-%d")



alert_assets = {}
alert_assets['gpu'] = ['NVDA','AMD','MU','XSD','SMH','SOXX','LSCC','AMAT','NVMI']
alert_assets['AI'] = ['NVDA','AMD']
alert_assets['video games'] = ['NVDA','AMD']
alert_assets['artificial intelligence'] = ['NVDA','AMD']
alert_assets['houses for sale'] = ['SUI','MAA','ELS']
alert_assets['houses for rent'] = ['SUI','MAA','ELS']
alert_assets['rent relief'] = ['SUI','MAA','ELS']
alert_assets['apartments for rent'] = ['SUI','MAA','ELS']


keys = list(alert_assets.keys())


#Interests over time

def get_alerts(keys):
    pytrends = TrendReq(hl='en-US', tz=360)
    
    dat = pd.DataFrame()
    for i in keys:
            
        kw_list = [i]
        
        pytrends.build_payload(kw_list, cat=0, 
                               #timeframe='today 5-y',
                               timeframe='today 3-m',
                               #timeframe=timeframe, 
                               #timeframe='2016-12-14 2017-01-25', 
                               geo='US', gprop='')
        iot=pytrends.interest_over_time()
        
        iot=iot.rename(mapper={i:"trend"},axis='columns')
        iot['_rolling_7']=iot["trend"].rolling(window=7,min_periods=0).mean()
        iot['_rolling_7_shift1']=iot['_rolling_7'].shift(1)
        iot['_7diff']=iot['_rolling_7']-iot['_rolling_7_shift1']
        iot['query'] = i
        iot['upper'] = iot['_rolling_7_shift1']+np.percentile(iot['_7diff'].values[8:],95)
        iot['alert']=iot['_rolling_7']-iot['upper']
        iot['alert_flg'] = np.where(iot['alert']>0,1,0)
        
        iot = iot.tail(len(iot)-8)
        
        iot.reset_index(inplace=True,drop=False)
        
        dat=dat.append(iot)
    
    
    dat.reset_index(inplace=True,drop=True)
    
    curr_dat = dat[dat['date']>= max(dat['date'])-dt.timedelta(3)].copy()

    alerts = curr_dat[curr_dat['alert_flg']>0].copy()

    return dat,curr_dat,alerts


dat,curr_dat,alerts=get_alerts(keys)

"""
plt.plot(iot["trend"])
plt.plot(iot['_rolling_7_shift1'])
plt.plot(iot['_rolling_7'])
plt.show()

plt.plot(iot['_7diff'])
plt.show()

plt.hist(iot['_7diff'])
plt.show()



plt.plot(iot["trend"])
plt.plot(iot['_rolling_7'])
plt.plot(iot['upper'])
plt.show()


plt.plot(iot['alert'])
plt.show()"""

