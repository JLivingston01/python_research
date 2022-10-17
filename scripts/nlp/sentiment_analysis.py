

import pandas as pd
import numpy as np

txt = ['this is fake text',
       "i don't care what this text says",
       "please don't abandon me",
       "it needs preprocessing but i really don't want to do that right now"]

txt = pd.DataFrame({'text':txt})

txt['txt'] = txt['text'].apply(lambda x: "".join([i for i in x if (i.isalpha())|(i==" ")]))


def get_emotions():
    emotions=pd.read_csv("lexicons/nrc_emotions.csv")
    emotions.drop(['Unnamed: 0'],axis=1, inplace=True)
    emotions = emotions[np.sum(emotions,axis=1)>0].copy()
    return emotions

def add_emotions(df):
    
    emotions = get_emotions()
        
    emotions.set_index('term',inplace=True)
    
    dimensions = emotions.columns.values
    
    df1=df.copy()
    
    for i in dimensions:
        temp = list(emotions[emotions[i]==1].index)
        df1['emotions_'+i]=df1.txt.apply(lambda x: len([i for i in x.split() if i in temp]))
        
    for i in dimensions:
        df1['emotions_'+i+'_norm'] = df1['emotions_'+i]/np.sum(df1[['emotions_'+j for j in dimensions]],axis=1)
        
    return df1

pd.set_option("display.max_columns",500)
add_emotions(txt)