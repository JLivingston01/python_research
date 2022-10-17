
from dotenv import dotenv_values
import pandas as pd
import matplotlib.pyplot as plt

config = dotenv_values("c:/users/jliv/ml_api/.env")

AT = config['NYC_OPEN_ML_APP_TOKEN']

URL = f"https://data.cityofnewyork.us/resource/qgea-i56i.csv?$$app_token={AT}&$select=cmplnt_fr_dt,ofns_desc&$limit=8000000&LAW_CAT_CD=FELONY"

complaints_hist = pd.read_csv(URL)

complaints_hist.columns = ['DATE','DESC']

complaints_hist['DATE']=complaints_hist['DATE'].str.replace("1019",'2019')
complaints_hist['DATE']=complaints_hist['DATE'].str.replace("1016",'2016')
complaints_hist['DATE']=complaints_hist['DATE'].str.replace("1017",'2017')
complaints_hist['DATE']=complaints_hist['DATE'].str.replace("1027",'2017')
complaints_hist['DATE']=complaints_hist['DATE'].str.replace("1026",'2016')
complaints_hist['DATE']=complaints_hist['DATE'].str.replace("1028",'2018')
complaints_hist['DATE']=complaints_hist['DATE'].str.replace("1029",'2019')
complaints_hist['DATE']=complaints_hist['DATE'].str.replace("1015",'2015')
complaints_hist['DATE']=complaints_hist['DATE'].str.replace("1018",'2019')

complaints_hist['DATE'] = pd.to_datetime( complaints_hist['DATE'])

daily = complaints_hist.groupby(['DATE']).agg({'DESC':'count'})

plt.plot(daily['DESC'])

del complaints_hist