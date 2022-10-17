
import pandas as pd

d = pd.DataFrame()
d['GROUP']=[1,1,2,2,2,3,3,3,3]
d['INDEX_NEW'] = [1,2,3,4,5,6,7,8,9]

d.index=[0,1,2,3,4,5,6,7,8]

def funct(M):
    print(M,M['GROUP'].values[0],len(M['INDEX_NEW'].values),len(M.index))
    print(M['INDEX_NEW'].values)
    print(M.index)
    #M.index=M['INDEX_NEW'].values
    
d.groupby(['GROUP']).apply(funct)


