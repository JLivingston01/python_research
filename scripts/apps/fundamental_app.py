
import dash
from dash import (
    dcc,
    html,
    dash_table
)
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

from flask import Flask

import pandas as pd
import numpy as np
import requests
import json
from dotenv import load_dotenv
import os

def request_fundamentals(Tickers: list, consumerKey: str, how: str = 'DF') -> pd.DataFrame:
    """_summary_

    Args:
        Tickers (list)
        consumerKey (str)

    Returns:
        pd.DataFrame
    """
    symbolList = ",".join([i for i in Tickers])
    endpoint = f'''https://api.tdameritrade.com/v1/instruments?
        &symbol={symbolList}&projection=fundamental'''
    page = requests.get(url=endpoint, 
        params={'apikey' : consumerKey})
    content = json.loads(page.content)

    allDict = {}
    for k in list(content.keys()):
        allDict[k] = content[k]['fundamental']
    fundamentals = pd.DataFrame(allDict).T
    
    if how=='DF':
        return fundamentals
    else:
        return allDict

def request_quotes(consumerKey: str, Tickers: list) -> pd.DataFrame:
    """_summary_

    Args:
        consumerKey (str): _description_
        Tickers (list): _description_

    Returns:
        pd.DataFrame: _description_
    """
    
    symbolList = ",".join([i for i in Tickers])
    endpoint = f"https://api.tdameritrade.com/v1/marketdata/quotes?symbol={symbolList}"
    page = requests.get(url=endpoint, 
                params={'apikey' : consumerKey})
    content = json.loads(page.content)
    content = pd.DataFrame(content).T
    return content

def return_options() -> list:
    """_summary_

    Returns:
        list: _description_
    """
    suffixes = ['CD','Comms','CS','Ener','Fina','HC','Ind','IT','Mat','RE','Util']

    metricsFamilies = {
            'debt': {'columns': ['totalDebtToCapital','ltDebtToEquity','totalDebtToEquity'],
                    'ascending':True},
            'ratio': {'columns': ['quickRatio','currentRatio','interestCoverage'],
                    'ascending':False},
            'change': {'columns': ['epsChangePercentTTM',
            'epsChangeYear', 'epsChange', 'revChangeYear', 'revChangeTTM',
            'revChangeIn'],
                    'ascending':False},
            'profit': {'columns': ['epsTTM','grossMarginTTM', 'grossMarginMRQ', 'netProfitMarginTTM',
            'netProfitMarginMRQ', 'operatingMarginTTM', 'operatingMarginMRQ'],
                    'ascending':False},
            'value': {'columns': ['peRatio','pegRatio', 'pbRatio', 'prRatio',
            'pcfRatio'],
                    'ascending':True},
            'return': {'columns':  ['returnOnEquity', 'returnOnAssets', 'returnOnInvestment'],
                    'ascending':False},
        }

    return [suffixes,metricsFamilies]

def convert_to_ranking(rankdat: pd.DataFrame, 
    metricsFamilies: dict[str,dict]) -> pd.DataFrame:
    """_summary_

    Args:
        rankdat (pd.DataFrame): _description_
        metricsFamilies (dict[str,dict]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    families = list(metricsFamilies.keys())
    
    for f in families:
        columns = metricsFamilies[f]['columns']
        ascending= metricsFamilies[f]['ascending']
        for v in columns:
            rankdat[v] = rankdat[v].rank(ascending=ascending)

        rankdat[f'{f}Known']=np.sum(np.where(rankdat[columns].isna(),0,1),axis=1)

        rankdat[columns]=rankdat[columns]/np.nanmax(rankdat[columns],axis=0)

    return rankdat

def make_app() -> dash.Dash:
    """_summary_

    Returns:
        dash.Dash: _description_
    """

    load_dotenv(".env",override=True)
    consumerKey = os.environ.get("CONSUMER_KEY")

    options = return_options()

    external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app=dash.Dash(__name__,
                external_stylesheets=external_stylesheets,)
    app.config['suppress_callback_exceptions'] = True

    app.layout = html.Div( #START DOCUMENT
            children= [
                    html.Div( #START HEADER
                            [
                                    html.Div([html.H1("LOGO GOES HERE")],className="three columns"),
                                    html.Div([html.H5('MENU AND INTRO INFO')],className="nine columns"),
                                    ],
                            className='twelve columns'), #END HEADER
                    html.Div( #START BODY
                            [
                                    html.Div( #START ROW
                                            [
                                                    html.Div([
                                    html.Div([html.Label("Select Industry:")],style={'color':'white'}),
                        dcc.Dropdown(
                                    id = 'industry',
                                    options=[{'label': i, 'value': i} for i in options[0]],
                                        value=options[0][0]
                                    ),
                                                            ],className='three columns',id='pane_0'),
                                                    html.Div([
                                                            html.H1("CONTENT PANE 1"),
                                                            html.Div(id='output_1'),
                                                            ],className='five columns',id='pane_1'),
                                                    html.Div([
                                                            html.H1("CONTENT PANE 2"),
                                                            ],className='four columns',id='pane_2'),
                                                    ],
                                                className='twelve columns'), #END ROW
                                    html.Div( #START ROW
                                            [
                                                    html.Div([
                                                            html.H1("CONTENT PANE 3"),
                                                            ],className='six columns',id='pane_3'),
                                                    html.Div([
                                                            html.H1("CONTENT PANE 4"),
                                                            ],className='six columns',id='pane_4'),
                                                    ],
                                                className='twelve columns'), #END ROW
                                    
                                    ],
                            className='twelve columns'), #END BODY
                    ]
            )   #END DOCUMENT



    @app.callback(
        Output('output_1','children'),
        [Input(component_id='industry', component_property='value'),]
    )
    def render_content(industry_value,):
        
        symbols = pd.read_excel(f"data/static_files/LargeCap{industry_value}.xlsx",header=1)
        symbols['Symbol'].fillna("-",inplace=True)
        Tickers = list(symbols[(~symbols['Symbol'].astype(str).str.contains('-'))]['Symbol'])

    
        FDF = request_fundamentals(Tickers=Tickers,consumerKey=consumerKey)
        Quotes = request_quotes(Tickers=Tickers,consumerKey=consumerKey)

        dat = Quotes[['symbol','description','lastPrice',
            'volatility','peRatio','divAmount','divYield',
            ]].merge(FDF[['symbol']+[
                i for i in FDF.columns if i not in Quotes.columns
            ]],on='symbol',how='inner')

        dat=pd.DataFrame(np.where(dat==0,np.nan,dat),columns = dat.columns)

        rankdat = dat.copy()

        metricsFamilies = options[1]
        rankdat = convert_to_ranking(rankdat,metricsFamilies)
        rankedCols = []
        for k in metricsFamilies.keys():
            rankedCols=rankedCols+metricsFamilies[k]['columns']
            
        knownCols = [i for i in rankdat.columns if 'Known' in i]
        rankdat['metricsInformed'] = np.sum(rankdat[knownCols],axis=1)

        rankdat['meanRank']=np.sum(rankdat[rankedCols],axis=1)/rankdat['metricsInformed']

        out=rankdat[['symbol','description','meanRank','metricsInformed']].merge(
            dat[['symbol']+rankedCols],
            on=['symbol'],
            how='left'
        )

        for i in out.columns:
            try:
                out[i]=out[i].astype(float)
            except:
                pass

        out = np.round(out,2)
        out.sort_values(by='meanRank',inplace=True)

        
        data = out.to_dict('rows')
        columns =  [{"name": i, "id": i,} for i in (out.columns)]


        outTable = dash_table.DataTable(id='data_table',style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
            'lineHeight': '15px'
            },data=data, columns=columns)


        return [html.H5(industry_value),
                outTable
            ]

    return app

def main():
    """_summary_
    """
    app = make_app()

    app.run_server(debug=False,
                   port=8080)

    return

if __name__=='__main__':
    main()