


import yfinance as yf

import pandas as pd




symbols = list(set(list(pd.read_excel("financial_data/export-net (8).xlsx",header=1)['Symbol'])))

symbols.sort()


batches = [symbols[:int(len(symbols)/5)],
           symbols[int(len(symbols)/5):2*int(len(symbols)/5)],
           symbols[2*int(len(symbols)/5):3*int(len(symbols)/5)], 
           symbols[3*int(len(symbols)/5):4*int(len(symbols)/5)],       
           symbols[4*int(len(symbols)/5):]]


dicts=[]
for batch in batches:
    tickers = yf.Tickers(' '.join(batch))

    info=tickers.download_info()
    
    dicts.append(info)


info_dct={}
for dct in dicts:
    
    for k in list(dct.keys()):
        info_dct[k]=dct[k]
"""
len(info_dct.keys())


out=pd.DataFrame()
for symb in symbols:
    
    print(symb)
    try:
        t = info_dct[symb]
        
        current,upside,median,mean,minimum,opinions,reco_mean = t['currentPrice'],\
                        t['targetMedianPrice']/t['currentPrice'],\
                        t['targetMedianPrice'],\
                        t['targetMeanPrice'],\
                        t['targetLowPrice'],\
                        t['numberOfAnalystOpinions'],\
                        t['recommendationMean']
    
        out=out.append(
                pd.DataFrame([(symb,current,upside,median,mean,minimum,opinions,reco_mean)],columns = ['symb','current','upside','medianTarget','meanTarget','minTarget','opinions','recoMean'])
                    )
        
    except KeyboardInterrupt:
        print('interrupted!')
    except:
        print(symb," Missing Crutial Info")
        
out.to_csv("price_targets.csv",index=False)
"""

info_df=pd.DataFrame(info_dct).T

info_df['upside']=info_df['targetMedianPrice']/info_df['currentPrice']-1

cols = list(info_df.columns)

cols.sort()

info_df[cols].to_csv("info.csv")
