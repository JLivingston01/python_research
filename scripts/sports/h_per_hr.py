
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


h=pd.read_csv("c:/users/jliv/downloads/h_career.csv")
hr=pd.read_csv("c:/users/jliv/downloads/hr_career.csv")


h.columns = ['rank','player','h','bats']
hr.columns = ['rank','player','hr','bats',
              'hrlog']

h=h[h['player']!='Player (yrs, age)'].copy()
hr=hr[hr['player']!='Player (yrs, age)'].copy()



hr=hr.merge(h[['player','h']],on=['player'],how='left')

hr['hr']=hr['hr'].astype(float)
hr['h']=hr['h'].astype(float)

hr['h/hr']=hr['h']/hr['hr']


hr.sort_values(by='h/hr',ascending=True,inplace=True)

plt.plot(hr['h/hr'].values)
plt.show()


plt.hist(hr['h/hr'],bins=np.linspace(2.5,37,45))
plt.show()

hr[~hr['h'].isna()]
mu=np.mean(np.log(hr['h/hr']))
sd=np.std(np.log(hr['h/hr']))

sampled=np.random.normal(mu,sd,size=500*len(hr[~hr['h'].isna()]))


plt.hist(np.log(hr['h/hr']),bins=30,alpha=.5,density=True)
plt.hist(sampled,bins=30,alpha=.5,density=True)
plt.xticks(np.linspace(1,3.5,6),np.round(np.exp(np.linspace(1,3.5,6)),2))
plt.xlim(.5,4)
plt.show()



hr['log_h/hr']=np.log(hr['h/hr'])

hr['likelihood']=norm(loc=mu,scale=sd).cdf(hr['log_h/hr'])

1/hr['likelihood']

hr.head(50)







