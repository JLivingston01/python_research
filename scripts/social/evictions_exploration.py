

import pandas as pd



data=pd.read_csv("c://users/jliv/downloads/evictions.csv")


pd.set_option("display.max_columns",40)

residential=data[data['Residential/Commercial']=='Residential'].copy()

residential['bl']=residential['BOROUGH'].str.lower()


residential=residential[~residential['Community Board'].isna()].copy()

residential['Community Board']=residential['Community Board'].astype(int)


residential.groupby(['BOROUGH','Community Board']).agg({'Docket Number ':'count'})


d2=pd.read_csv("c://users/jliv/downloads/New_York_City_Population_By_Community_Districts.csv")



d2['bl']=d2['Borough'].str.lower()

d2['Community Board']=d2['CD Number'].astype(int)

d2=d2[['bl','Community Board','2010 Population']].copy()


evictions = residential.groupby(['bl','Community Board']).agg({'Docket Number ':'count'}).reset_index()

evictions=evictions.merge(d2,on=['bl','Community Board'],how='left')


evictions.columns=['borough','cb','evictions','population']

evictions['evictions_per_100k']=evictions['evictions']*100000/evictions['population']