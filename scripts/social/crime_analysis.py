
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dat=pd.read_csv("c:/users/jliv/downloads/texas_inmates.csv")

pd.set_option("display.max_columns",25)
dat.head()

dat['Life']=np.where((dat['Sentence (Years)'].str.contains("Life"))|
        (dat['Sentence (Years)'].str.contains("LWOP")),1,0)


dat['Death']=np.where((dat['Sentence (Years)']=='Death'),1,0)

dat['S']=np.where((dat['Sentence (Years)'].str.contains("\.", na=False)),
   dat['Sentence (Years)'],
              np.nan).astype(float)


dat.head()

dat.groupby(['Race']).agg({'S':['mean','median','std','count']})
dat.groupby(['Gender']).agg({'S':['mean','median','std','count']})


bins = np.linspace(0,6,20)
plt.hist(np.log(dat[dat['Race']=='B']['S']),
         bins=bins,
         alpha=.5)
plt.xticks(range(10),np.round(np.exp(np.array(range(10))),2),
           rotation=90)


plt.hist(np.log(dat[dat['Race']=='W']['S']),
         bins=bins,
         alpha=.5)
plt.xticks(range(10),np.round(np.exp(np.array(range(10))),2),
           rotation=90)


plt.hist(np.log(dat[dat['Race']=='H']['S']),
         bins=bins,
         alpha=.5)
plt.xticks(range(10),np.round(np.exp(np.array(range(10))),2),
           rotation=90)
plt.show()


len(dat['Offense Code'].unique())
len(dat['TDCJ Offense'].unique())

dat.groupby(['Offense Code','TDCJ Offense']).agg({'S':'count'})

offense_codes=dat.groupby(['Offense Code']).agg({'TDCJ Offense':['min']}).reset_index()
offense_codes.columns = ['Offense Code','clean_offense_1']

offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("CAPITAL","CAP")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("MURDER","MUR")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("PROTECT","PROT")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("ABANDON","ABAN")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("ABAND","ABAN")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("ENDANGER","END")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("ENDANG","END")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("/"," ")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("AGGRAVATED","AG")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("AGGREGATED","AG")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("AGREGATE","AG")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("AGG","AG")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("ASSAULT","ASLT")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("PROSTITUTION","PROST")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("PROMO","PROM")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("TERRORISTIC","TER")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("TERROR","TER")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("TAMPERING","TAMP")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("TAMPER","TAMP")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("SEXUAL","SEX")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("INTERFERE","INTERFER")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("INTOXICATION","INTOX")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("INJURY","INJ")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("INDEC","IND")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("PERSON","PERS")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace(" IN "," ")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("CORR","COR")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("FACIL","FAC")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("HARASMT","HARASS")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("FRADULENT","FRAUD")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("FORGERY","FORG")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("FELONY","FEL")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("FELON","FEL")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace(" OF "," ")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace(" TO ","")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("OBTAIN","OBT")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("ACCIDENT","ACC")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("INVOLV","INV")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("ROBBERY","ROB")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("ROBB","ROB")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("RESTRAINT","REST")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("RESTR","REST")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("UNAUTH","UNA")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("UNATH","UNA")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("PROTIVE","PROT")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("ORDER","ORD")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("UNLAW","UNL")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("ORGANIZED","ORG")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace(" WHERE "," ")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("PROHIBITED","PROH")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("WEAPONS","WPN")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("WEAPON","WPN")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("SOLICIT","SOL")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("SOLIC","SOL")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("FIDUCIARY","FIDU")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("FIDUC","FIDU")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("MANUAL","MAN")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("ARREST","ARR")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("DETENTION","DET")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("DETN","DET")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("EVADING","EV")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("EVADES","EV")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("EVADE","EV")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("EVAD","EV")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("CRIMINAL","CRIM")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("MISCHIEF","MIS")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("MISCH","MIS")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("MISC","MIS")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("CONTINUOUS","CONT")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("TRAFFICKING","TRAFF")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("PERSS","PERS")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("CONSPIRACY","CONSP")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("COMPELLING","COMP")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("COMPEL","COMP")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("COMMIT","COMM")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("BURGLARY","BRG")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("BURG","BRG")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("BUILD","BLD")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace(" FAILAPPEAR","")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("ATTEMP","ATT")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("CAPTIAL","CAP")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("ALST","ASLT")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("AGNST","A")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("ASLT AG","ASLT A")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("PUBLIC","PUB")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("SERVANT","SERV")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("DEADLY WPN","DW")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("DDLY WPN","DW")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("KIDNAPPING","KDNP")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("KIDNAP","KDNP")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("KDNPP","KDNP")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("AGREGATE","AG")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("INJCHILD","INJ CHILD")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("INJCHILD","INJ CHILD")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("ELDERLY","ELDER")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("INJELDER","INJ ELDER")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("ARSON","ARSN")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("DRIVING","DRIV")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("DRIVE","DRIV")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("INJA","INJ")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("PORNOGRAPHY","PORN")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("PRONOGRAPHY","PORN")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("PROMTION","PROMO")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("CONT SUB","C S")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("SUB","S")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace(" CS"," C S")
offense_codes['clean_offense_1']=offense_codes['clean_offense_1'].str.replace("TERR","TER")




col=offense_codes['clean_offense_1']

Crime_groups=['ABAN END ','ACC ','ARSN','ASLT','KDNP','POSS','PROST','RAPE','CHILD','SEX ASLT',
     'THEFT','ATT','MUR','CAP MUR','BRG','CRIM MIS','CRIM SOL','DEL C','FORG','FRAUD',
     'IND','INJ CHILD','INJ ELDER','INTOX','RETAIL THEFT','POSS C S','POSS FIREARM',
     'SECUR','TAMP','TER THREAT','UNL REST','PROT ORD','VIOL BOND','VOYEURISM'
     ]





dat2=dat.merge(offense_codes,on=['Offense Code'],how='left')

for i in Crime_groups:
    dat2[i]=np.where(dat2['clean_offense_1'].str.contains(i),1,0)


for i in Crime_groups:
    subdat= dat2[dat2[i]==1].copy()
        
    bins = np.linspace(0,6,20)
    plt.hist(np.log(subdat[subdat['Race']=='B']['S']),
             bins=bins,
             alpha=.5,label='B')
    plt.xticks(range(10),np.round(np.exp(np.array(range(10))),2),
               rotation=90)
    
    
    plt.hist(np.log(subdat[subdat['Race']=='W']['S']),
             bins=bins,
             alpha=.5,label='W')
    plt.xticks(range(10),np.round(np.exp(np.array(range(10))),2),
               rotation=90)
    
    
    plt.hist(np.log(subdat[subdat['Race']=='H']['S']),
             bins=bins,
             alpha=.5,label='W')
    plt.xticks(range(10),np.round(np.exp(np.array(range(10))),2),
               rotation=90)
    plt.title(i)
    plt.legend()
    plt.show()
    
    print(i,subdat.groupby(['Race']).agg({'S':['mean','median','std','count']}))



offense_codes2=dat2.groupby(['clean_offense_1']).agg({'Offense Code':['min']}).reset_index()
offense_codes2.columns = ['clean_offense_1','clean_code_1']

offense_codes2['clean_offense_1'].unique()

dat3=dat2.merge(offense_codes2,on=['clean_offense_1'],how='left')
