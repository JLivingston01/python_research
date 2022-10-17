
import requests
import pandas as pd
pd.set_option("display.max_columns", 500)
import numpy as np
import ssl
from xgboost import XGBRegressor
import matplotlib.pyplot as plt


from scipy.optimize import minimize


ssl._create_default_https_context = ssl._create_unverified_context

url = 'https://www.numbeo.com/crime/in/Corpus-Christi'

"""
response = requests.get(url)

response.text
"""

period = '2022'
regions = [
    '011',
    '014',
    '017',
    '018',
    '015',
    '029',
    '013',
    '021',
    '005',
    '030','034','035','143','145',
    '151','154','155','039',
    '009'
    ]

region = regions[0]

def pull_period(period):
    
    url = f'https://www.numbeo.com/cost-of-living/rankings.jsp?title={period}'
    tables = pd.read_html(url)
    COL = tables[1]
    COL.drop('Rank',axis=1,inplace=True)
    
    url = f"https://www.numbeo.com/property-investment/rankings.jsp?title={period}"
    tables = pd.read_html(url)
    PROP = tables[1]
    PROP.drop('Rank',axis=1,inplace=True)
    
    
    url = f"https://www.numbeo.com/quality-of-life/rankings.jsp?title={period}"
    tables = pd.read_html(url)
    QOL = tables[1]
    QOL.drop('Rank',axis=1,inplace=True)
    
    
    url = f"https://www.numbeo.com/crime/rankings.jsp?title={period}"
    tables = pd.read_html(url)
    CRI = tables[1]
    CRI.drop('Rank',axis=1,inplace=True)
    
    
    url = f"https://www.numbeo.com/health-care/rankings.jsp?title={period}"
    tables = pd.read_html(url)
    HEA = tables[1]
    HEA.drop('Rank',axis=1,inplace=True)
    
    
    url = f"https://www.numbeo.com/pollution/rankings.jsp?title={period}"
    tables = pd.read_html(url)
    POL = tables[1]
    POL.drop('Rank',axis=1,inplace=True)
    
    
    url = f"https://www.numbeo.com/traffic/rankings.jsp?title={period}"
    tables = pd.read_html(url)
    TRA = tables[1]
    TRA.drop('Rank',axis=1,inplace=True)
    
    
    key = ['City']
    OUT = COL.merge(PROP,on=key,
              how='outer').merge(QOL[['City','Quality of Life Index']],on=key,
              how='outer').merge(CRI,on=key,
              how='outer').merge(HEA,on=key,
              how='outer').merge(POL,on=key,
              how='outer').merge(TRA,on=key,
              how='outer')
    OUT['PERIOD'] = period
    
    return OUT

OUT2022 = pull_period('2022')
OUT2021 = pull_period('2021')
OUT2020 = pull_period('2020')
OUT2019 = pull_period('2019')
OUT2018 = pull_period('2018')
OUT2017 = pull_period('2017')

OUT = OUT2022.append(OUT2021).append(OUT2020).append(OUT2019).append(OUT2018).append(OUT2017)
OUT.reset_index(inplace=True,drop=True)

OUT = OUT[~OUT['Quality of Life Index'].isna()].copy()
for col in OUT.columns[1:-1]:
    OUT[col].fillna(np.nanmedian(OUT[col]),inplace=True)
    
    
    
kpi ='Quality of Life Index'
feats=[ i for i in OUT.columns if i not in ['PERIOD','City',kpi]]
 

mod = XGBRegressor(random_state=42).fit(OUT[feats],OUT[kpi])


importance = pd.DataFrame({'feat':feats,
              'imp':mod.feature_importances_})
importance.sort_values(by='imp',ascending=False,inplace=True)

from sklearn.ensemble import RandomForestRegressor
mod = RandomForestRegressor(random_state=42).fit(OUT[feats],OUT[kpi])

importance = pd.DataFrame({'feat':feats,
              'imp':mod.feature_importances_})
importance.sort_values(by='imp',ascending=False,inplace=True)


OUT['INT']=100
linear_feats = ['Pollution Index','Local Purchasing Power Index',
 'Mortgage As A Percentage Of Income','Affordability Index','Crime Index',
 'Health CareExp. Index','Traffic Index','INT']

x = OUT[linear_feats].copy()

x=x/100

#x = np.log(x)
x['INT']=1

params = np.random.normal(0,.05,len(linear_feats))

def model(x,params):
    return np.exp(x@params)

def cost(params):
    Yexp = model(x,params)
    return np.mean(.5*(OUT[kpi] - Yexp)**2)

cost_optimized = minimize(cost,params,method = 'BFGS',#method = 'L-BFGS-B',
         options={'maxiter':10000})

pred = model(x,cost_optimized['x'])


plt.scatter(pred,OUT[kpi])
plt.show()

R2 = 1-sum((OUT[kpi]-pred)**2)/sum((OUT[kpi]-np.mean(OUT[kpi]))**2)
R2

coefs = pd.DataFrame({
    'feat':linear_feats,
    'coef':cost_optimized['x']
    })

coefs['avg_factor'] = np.exp(coefs['coef'])

coefs['factor_at_1.5'] = np.exp(coefs['coef']*1.5)
coefs['factor_at_.5'] = np.exp(coefs['coef']*.5)



x = OUT[linear_feats].copy()

#x=x/100

#x = np.log(x)
x['INT']=1

params = np.random.normal(0,.05,len(linear_feats))

def model(x,params):
    return x@params

def cost(params):
    Yexp = model(x,params)
    return np.mean(.5*(OUT[kpi] - Yexp)**2)

cost_optimized = minimize(cost,params,method = 'BFGS',#method = 'L-BFGS-B',
         options={'maxiter':10000})

pred = model(x,cost_optimized['x'])


plt.scatter(pred,OUT[kpi])
plt.show()

R2 = 1-sum((OUT[kpi]-pred)**2)/sum((OUT[kpi]-np.mean(OUT[kpi]))**2)
R2

coefs = pd.DataFrame({
    'feat':linear_feats,
    'coef':cost_optimized['x']
    })




