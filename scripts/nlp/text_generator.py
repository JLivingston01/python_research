
doc = '''
Many challenges in automation, learning about your data (your business) and optimization can be tackled with Machine Learning solutions. Machine learning can be used in learning the impact of events on real outcomes, predicting and optimizing these outcomes, analyzing your business's under-utilized text, creating groups from your business units, customers or data for targeted, strategic action, and truthfully, an endless number of challenges that an organization may face.

These solutions may be simple and preexisting, and with guidance your team can implement techniques and manage analytics independently. Sometimes the appropriate solution requires the creativity of a Machine Learning Engineer to conceive a new model or algorithm, implement and optimize the solution for efficiency. Challenges to implementation may involve the mathematics that make algorithms possible, or having a large amount of data that requires an algorithm to work differently, and an understanding of computer science fundamentals.

The internet is home to a broad, vast array of information that has yet to be harvested for its potential value. Web data could be textual, media related, structured information, or come in an endless variety of unstructured formats requiring transformation and cleaning so the information can be understood and used. This information can inform scientific research, market research, paint the picture of a competitive landscape, or provide information to streamline or augment business operations.

Many internet data sources make information available via an easy-to-use API, and with training and guidance, you can interesting data in easy-to-process formats, and manage your own data feed independently. Often there is no easy-to-use interface to download information that you might otherwise discover in your browser. For these situations, website scraping could be the right solution for obtaining useful information, and processing it for ease of use. Challenges in this area are often related to the dynamic nature of websites under rebuild, and server-side limitations to information scraping processes.


Data Sourcing and Scraping

Website Tagging and Analytics
Whether you run a small business, a large organization, or maintain a personal internet presence, understanding who is visiting your website and what is drawing their interest is crucial for optimizing your content, your website design, and your marketing efforts. Thankfully, most website tagging solutions can be implemented and maintained independently thanks to cloud services like Google Analytics or open-source Matomo, free versions of which enable simple but effective reporting.

Often, website analytics needs are more nuanced, requiring considerations for what data you are allowed to collect and store, whether your data collection and analytics can live in a cloud service or if an on-premise solution makes more sense, whether a premium solution might better suit your needs, or whether A/B testing and consumer targeting need to be implemented to optimize your website or business. These considerations may require the expertise of an experienced website analyst.

Attributing business outcomes to marketing is challenging for several reasons. It is largely impossible to follow an individual from the point of marketing exposure to the point of sale, especially when marketing is performed offline or out-of-home. Individuals may be exposed to marketing but convert later, rather than shortly after exposure, and sales can be driven by non-marketing factors, or by synergies between multiple factors. Understanding the effects on sales from marketing and non-marketing factors is crucial for optimizing marketing operations and business strategies.

Thankfully, through machine learning and statistical modeling, the effects of changes in the media mix can be estimated, synergies discovered and measured, and non-marketing factors affecting sales can be defined for their impact on your business. This type of business analysis is typically referred to as a Media Mix Model or Marketing Mix Model, and often requires the expertise of a marketing data scientist who understands the technical details of implementation, the relationships between types of marketing, and what synergies or non-marketing events to begin accounting for.


Media Mix and Business Modeling

Analyzing your Text
You or your business may be sitting on telephone transcripts, emails, news articles or social media posts about your business, and processing them for action by reading them manually, which can be a slow and time consuming process. Text Analytics and Natural Language Processing has evolved so text can be programatically processed for sentiment and emotions, the presence of keywords, topic detection and automatic sorting, and sensible text can even be generated automating assistants with simple, useful responses.

Implementing a Text Analytics process requires expertise in several sub-topics of the field including the above mentioned sentiment and emotional analysis, topic modeling techniques, the subject matter of which are very related to Machine Learning, as well as an understanding of lexicons and how to preprocess your text to maximize its usefullness and value.

Analytics are valueless without a mechanism for effectively communicating findings, and working with your data. Because people are often more effective at learning visually, many businesses turn to dashboarding for performance reporting and monitoring the business. Dashboards can be highly interactive allowing you, the user, to toggle quickly between scenarios in order to easily process valuable insights. Dashboards can represent voluminous data in a simple assortment of graphs and tables for your consumption.

There are many softwares that allow the developer to design dashboards with easy-to-use point-and-click interfaces, such as Tableau. These softwares are typically highly capable, but the ease-of-use does come at the expense of versatility. In cases where a high degree of customization is desired, an expert in dashboarding frameworks can develop and deploy tailored applications for your needs. An expert in dashboard application programming can help you create a versatile dashboarding application that fits your use case and vision.
'''

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
import requests
from bs4 import BeautifulSoup
pd.set_option('display.max_columns', 500)

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

url = 'https://thedataprofessional.com/analytics.html'

response = requests.get(url)

response.text

soup = BeautifulSoup(response.text, 'html.parser')
    
texts = soup.find_all('p')
    
doclist = []
for t in texts:
    doclist.append(t.text)
    
doc = ' '.join(doclist)

def clean_doc(doc):
    doc = doc.lower().replace("\n"," ").replace("\t"," ")
    doc = ''.join([i for i in doc if 
                   (i.isalnum())|(i==' ')|(i=='.')]).replace("."," <break>")
    tokens = doc.split()
    token_set = set(tokens+["<none>"])
    token_ids = {i:j for i,j in zip(token_set,range(len(token_set)))}
    ids_tokens = {j:i for i,j in zip(token_set,range(len(token_set)))}
    
    return tokens,token_ids,ids_tokens

tokens,token_ids,ids_tokens = clean_doc(doc)

def calculate_lookback(tokens,token_ids,lookback):

    
    df = pd.DataFrame({'tokens':tokens})
    for i in range(lookback):
        ind = i+1
        df[f'tokens_shift{ind}']=df['tokens'].shift(ind)
        
    df.fillna("<none>",inplace=True)

    
    M = pd.DataFrame()
    for col in df.columns:
        
        M[col] = df[col].map(token_ids)
        
    return df,M

lookback=4
df,M=calculate_lookback(tokens,token_ids,lookback=lookback)

feats = ['tokens_shift1',
         'tokens_shift2',
         'tokens_shift3',
         'tokens_shift4',]
kpi = 'tokens'

def generate_new(gen_length,lookback,mod,init_word = '<none>',le=None):
    x = pd.DataFrame()
    for i in range(lookback):
        ind=i+1
        x[f'tokens_shift{ind}']=[token_ids['<none>']]
    
    x['tokens_shift1']=[token_ids[init_word]]
    
    predicted_token = mod.predict(x[feats])
    if le is not None:
        predicted_token=le.inverse_transform(predicted_token)
    x['tokens'] =predicted_token
    
    for i in range(gen_length):
        row = pd.DataFrame(x.iloc[i]).T
        for ind in range(lookback):
            to_col = lookback-ind
            from_col = lookback-(ind+1)
            if from_col>0:
                row[f'tokens_shift{to_col}'] = row[f'tokens_shift{from_col}']
            else:
                row[f'tokens_shift{to_col}'] = row['tokens']
        
        a = mod.predict(row[feats])
        if le is not None:
            a = le.inverse_transform(a)
            
        row['tokens']=a
        
        x=x.append(row).reset_index(drop=True)
    
    return x

mod = RandomForestClassifier(random_state=42)
mod.fit(M[feats],M[kpi])
gen_length=115
x = generate_new(gen_length,lookback,mod,init_word = '<none>')
" ".join(list(x['tokens'].map(ids_tokens)))


mod = MultinomialNB()
mod.fit(M[feats],M[kpi])
gen_length=115
x = generate_new(gen_length,lookback,mod,init_word = 'data')
" ".join(list(x['tokens'].map(ids_tokens)))


mod = MLPClassifier(hidden_layer_sizes=(100,100,100),activation="relu",
                    learning_rate='adaptive',max_iter=10000,verbose=True,
                    tol=1e-5)
mod.fit(M[feats],M[kpi])
gen_length=115
x = generate_new(gen_length,lookback,mod,init_word = '<none>')
" ".join(list(x['tokens'].map(ids_tokens)))


le = LabelEncoder()
y_train = le.fit_transform(M[kpi])
mod = XGBClassifier(random_state=42)
mod.fit(M[feats],y_train)
gen_length=115
x = generate_new(gen_length,lookback,mod,init_word = '<none>',le=le)
" ".join(list(x['tokens'].map(ids_tokens)))



mod = RandomForestClassifier(random_state=43)
y_train = le.fit_transform(M[kpi])
mod.fit(M[feats],y_train)
gen_length=115
x = generate_new(gen_length,lookback,mod,init_word = '<none>',le=le)
" ".join(list(x['tokens'].map(ids_tokens)))


mod = RandomForestClassifier(random_state=43)
mod.fit(M[feats],M[kpi])
gen_length=115
x = generate_new(gen_length,lookback,mod,init_word = '<none>')
" ".join(list(x['tokens'].map(ids_tokens)))


##Looking forward and back


def calculate_lookback_and_forward(tokens,token_ids,
                                   lookback=3,
                                   lookforward = 3):

    
    df = pd.DataFrame({'tokens':tokens})
    for i in range(lookback):
        ind = i+1
        df[f'tokens_shift{ind}']=df['tokens'].shift(ind)
        
    for i in range(lookforward):
        ind = i+1
        df[f'tokens_fshift{ind}']=df['tokens'].shift(-ind)
        
    df.fillna("<none>",inplace=True)
        
    M = pd.DataFrame()
    for col in df.columns:
        
        M[col] = df[col].map(token_ids)
        
    return df,M

lookback=4
lookforward=4
df,M=calculate_lookback_and_forward(tokens,token_ids,lookback=lookback,
                        lookforward=lookforward)

#Spewer
feats=['tokens_shift1', 'tokens_shift2', 'tokens_shift3', 'tokens_shift4']

mod = RandomForestClassifier(random_state=43)
mod.fit(M[feats],M[kpi])

#only predict by looking backwards
x = pd.DataFrame()
x['tokens']=None
for i in range(lookback):
    ind=i+1
    x[f'tokens_shift{ind}']=[token_ids['<none>']]
x['tokens_shift1']=[token_ids['<none>']]

gen_length=115
for r in range(gen_length):
    
    r=0
    xprime=x.iloc[r].copy()
    for i in range(lookforward+1):
        xprime['tokens']=mod.predict(pd.DataFrame(xprime[feats]).T)[0]
        xprime=xprime[['tokens']+feats]
        xprime=xprime.shift(1)


#Planner
ffeats=['tokens_shift1', 'tokens_shift2', 'tokens_shift3', 'tokens_shift4',
       'tokens_fshift1', 'tokens_fshift2', 'tokens_fshift3', 'tokens_fshift4']

mod2 = RandomForestClassifier(random_state=43)
mod2.fit(M[ffeats],M[kpi])


#predict three words forward with spewer
x = pd.DataFrame()
x['tokens']=None
for i in range(lookback):
    ind=i+1
    x[f'tokens_shift{ind}']=[token_ids['<none>']]
x['tokens_shift1']=[token_ids['<none>']]

gen_length=115
for r in range(gen_length):
    
    
    xprime=x.iloc[r].copy()
    for i in range(lookforward+1):
        xprime['tokens']=mod.predict(pd.DataFrame(xprime[feats]).T)[0]
        xprime=xprime[['tokens']+feats]
        xprime=xprime.shift(1)
    
    xprime.index = ['tokens']+[f'tokens_fshift{i+1}' for i in range(lookforward)]
    
    xprime=xprime.append(x.iloc[r][feats])
    
    xprime['tokens']=mod2.predict(pd.DataFrame(xprime[ffeats]).T)[0]
    
    row = x.iloc[r].copy()
    row['tokens'] = xprime['tokens']
    x.iloc[r]=row
    
    x=x.append(row.shift(1)).reset_index(drop=True)
    

" ".join(list(x[~x['tokens'].isna()]['tokens'].map(ids_tokens)))

