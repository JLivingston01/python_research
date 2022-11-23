
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




def get_positions() -> dict:
    """_summary_

    Returns:
        dict: _description_
    """
    consumerKey = os.environ.get("CONSUMER_KEY")
    refreshToken = os.environ.get("TD_REFRESH_TOKEN")
    acctID = os.environ.get("TD_ACCT_ID")

    response = requests.post("https://api.tdameritrade.com/v1/oauth2/token",
        data = {'grant_type':'refresh_token',
            'refresh_token':refreshToken,
            'client_id':consumerKey,
            'redirect_uri':''}
    )
    accessToken = json.loads(response.content)['access_token']

    #get positions
    url = f'https://api.tdameritrade.com/v1/accounts/{acctID}?fields=positions'
    response = requests.get(url,
            headers={'Authorization' : f"Bearer {accessToken}"})

    positionsDict = json.loads(response.content)
    return positionsDict



def process_positions(positionsDict: dict) -> pd.DataFrame:
    balance = positionsDict['securitiesAccount']['currentBalances']['equity']
    positions = pd.DataFrame(positionsDict['securitiesAccount']['positions'])
    positions['symbol'] = positions['instrument'].apply(lambda x: x['symbol'])
    positions = positions[['symbol',
    'marketValue',
    'longQuantity','averagePrice']].copy()
    positions['myWeight'] = 1
    positions['equityShare'] = balance*positions['myWeight']/sum(positions['myWeight'])
    positions['currentPrice'] = positions['marketValue']/positions['longQuantity']
    return positions



def make_app() -> dash.Dash:
    """_summary_

    Returns:
        dash.Dash: _description_
    """

    load_dotenv(".env",override=True)

    consumerKey = os.environ.get("CONSUMER_KEY")


    positionsDict = get_positions()
    balance = positionsDict['securitiesAccount']['currentBalances']['equity']
    positions = process_positions(positionsDict)

    positionsWeights = positions[['symbol','myWeight']].copy()

    
    external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app=dash.Dash(__name__,
                external_stylesheets=external_stylesheets,)
    app.config['suppress_callback_exceptions'] = True

    app.layout = html.Div( #START DOCUMENT
            children= [
                    html.Div( #START HEADER
                            [
                                    html.Div([html.H1("LOGO GOES HERE")],className="three columns"),
                                    html.Div([html.H5('Portfolio Balancer'),
                                              html.Button("Refresh Balance",id='balance-refresh',n_clicks=0)
                                              ],className="nine columns"),
                                    dcc.Store(id='memory', storage_type = 'session',
                                        data = positions.to_dict('records') ),
                                    dcc.Store(id='balance-memory', storage_type = 'session',
                                        data = {'balance':balance} ),
                                    ],
                            className='twelve columns'), #END HEADER
                    html.Div( #START BODY
                            [
                                    html.Div( #START ROW
                                            [
                                                    html.Div([
                                    html.Div([html.Label("Select your weights:")],style={'color':'white'}),
                                    dash_table.DataTable(
                                                id='table-editing',
                                                columns=(
                                                    [{'id': c, 'name': c} for c in positionsWeights.columns]
                                                ),
                                                data=positionsWeights.to_dict('records'),
                                                editable=True,
                                                sort_mode='single',
                                                sort_action='native',
                                                filter_action="native",
                                                page_size=50,
                                            ),
                                                            ],className='three columns',id='pane_0'),
                                                    html.Div([
                                                            #html.H1("CONTENT PANE 1"),
                                                            html.Div(id='balance-out'),
                                                            html.Div(id='output_1'),
                                                            ],className='nine columns',id='pane_1'),
                                                    #html.Div([
                                                    #        #html.H1("CONTENT PANE 2"),
                                                    #        ],className='four columns',id='pane_2'),
                                                    ],
                                                className='twelve columns'), #END ROW
                                    html.Div( #START ROW
                                            [
                                                    html.Div([
                                                            #html.H1("CONTENT PANE 3"),
                                                            html.Div(id='output_2'),
                                                            ],className='twelve columns',id='pane_3'),
                                                    #html.Div([
                                                            #html.H1("CONTENT PANE 4"),
                                                    #        ],className='six columns',id='pane_4'),
                                                    ],
                                                className='twelve columns'), #END ROW
                                    
                                    ],
                            className='twelve columns'), #END BODY
                    ]
            )   #END DOCUMENT

    @app.callback(
        Output('memory','data'),
        Output('balance-memory','data'),
        Input('balance-refresh','n_clicks')
    )
    def refresh_balance(n_clicks):
        positionsDict = get_positions()
        balance = positionsDict['securitiesAccount']['currentBalances']['equity']
        positions = process_positions(positionsDict)
        return [positions.to_dict("records"),{'balance':balance}]
        
    @app.callback(
        Output('output_1', 'children'),
        Output('balance-out', 'children'),
        Input('table-editing', 'data'),
        Input('table-editing', 'columns'),
        Input('memory', 'data'),
        Input('balance-memory', 'data'),
        )
    def update_table(data,columns,positions,balanceDict):
        
        df = pd.DataFrame(data, columns=[c['name'] for c in columns])
        df.index = df['symbol']
        balance = balanceDict['balance']

        positions = pd.DataFrame(positions)
        positions.index=positions['symbol']
        positions['myWeight'] = df['myWeight']
        positions['equityShare'] = float(balance)*positions['myWeight'].astype(int)/sum(positions['myWeight'].astype(int))
        positions['diff']  = positions['equityShare']  - positions['marketValue']

        data = positions.to_dict('records')
        columns = [{'id': c, 'name': c} for c in positions.columns]

        outTable = dash_table.DataTable(id='data_table',style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
            'lineHeight': '15px'
            },
            sort_mode='single',
            sort_action='native',
            filter_action="native",
            page_size=50,
            data=data, columns=columns)

        return [outTable,html.H4(f"Balance: {balance}")]


    return app

def main():
    app = make_app()

    app.run_server(debug=True,
                   port=8081)

    return

if __name__=='__main__':
    main()