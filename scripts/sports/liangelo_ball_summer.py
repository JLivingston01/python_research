
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dat=pd.read_csv("c:/users/jliv/downloads/summer_league.csv")


dat=dat[dat['Player']!='Player'].copy()

for i in dat.columns:
    if i!='Player':
        dat[i]=dat[i].astype(float)
        
dat['PPM']=dat['PTS']/dat['MIN']
dat['APM']=dat['AST']/dat['MIN']
dat['SPM']=dat['STL']/dat['MIN']
dat['RPM']=dat['REB']/dat['MIN']
dat['ORPM']=dat['OREB']/dat['MIN']
dat['DRPM']=dat['DREB']/dat['MIN']
dat['3PMPM']=dat['3PM']/dat['MIN']
dat['FTPM']=dat['FTM']/dat['MIN']
dat['FGPM']=dat['FGM']/dat['MIN']

dat.corr()
dat.sort_values(by='PPM',ascending=False).head(45)

def analyze(non_zero_field,comp_column,dat):
    data_to_analyse=dat[dat[non_zero_field]>0].copy()
    
    d=data_to_analyse[data_to_analyse['Player']=='LiAngelo Ball']
    
    xt=np.log(data_to_analyse[[non_zero_field]]).copy()
    xt['INT']=1
    yt=np.log(data_to_analyse[comp_column])
    
    coefs=np.linalg.pinv(xt.T@xt)@(xt.T@yt)
    
    ppm_pred=xt@coefs
    
    
    
    plt.scatter(data_to_analyse[non_zero_field],data_to_analyse[comp_column])
    plt.scatter(d[non_zero_field],d[comp_column],label='Ball')
    plt.plot(data_to_analyse[non_zero_field],np.exp(ppm_pred),color='red')
    plt.legend()
    plt.xlabel(non_zero_field)
    plt.ylabel(non_zero_field+' per Minute')
    plt.show()
    
    data_to_analyse[comp_column+' vs Trend'] = data_to_analyse[comp_column]-np.exp(ppm_pred)
    data_to_analyse[comp_column+' vs Trend'+' Percentile']=data_to_analyse[comp_column+' vs Trend'].rank()/len(data_to_analyse)
    #data_to_analyse.sort_values(by=comp_column+' vs Trend',ascending=False).head(35)
    v=data_to_analyse[data_to_analyse['Player']=='LiAngelo Ball'][comp_column+' vs Trend'+' Percentile'].values[0]
    print('Ball Percentile '+comp_column+" "+str(v))

non_zero_field='PTS'
comp_column='PPM'
analyze(non_zero_field,comp_column,dat)

non_zero_field='AST'
comp_column='APM'
analyze(non_zero_field,comp_column,dat)

non_zero_field='STL'
comp_column='SPM'
analyze(non_zero_field,comp_column,dat)



non_zero_field='REB'
comp_column='RPM'
analyze(non_zero_field,comp_column,dat)






non_zero_field='OREB'
comp_column='ORPM'
analyze(non_zero_field,comp_column,dat)



non_zero_field='DREB'
comp_column='DRPM'
analyze(non_zero_field,comp_column,dat)




non_zero_field='FTM'
comp_column='FTPM'
analyze(non_zero_field,comp_column,dat)





non_zero_field='FGM'
comp_column='FGPM'
analyze(non_zero_field,comp_column,dat)