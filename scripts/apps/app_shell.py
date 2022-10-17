

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import dash_table

from flask import Flask


server = Flask(__name__)

external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']

app=dash.Dash(__name__,
              external_stylesheets=external_stylesheets,
                server=server,
                url_base_pathname='/dash/')
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
                                html.Div([html.Label("Select Ticker:")],style={'color':'white'}),
                       dcc.Dropdown(
                                   id = 'ticker',
                                   options=[{'label': i, 'value': i} for i in ['A','E','I','O','U','Y']],
                                    value='Y'
                                   ),
                                                        html.H5("SELECTIONS"),
                                                        html.H5("DROPDOWNS"),
                                                        html.H5("SLIDER"),
                                                        html.H5("CHECK BOXES"),
                                                        html.H5("RADIO BUTTONS"),
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
    [Input(component_id='ticker', component_property='value'),]
)
def render_content(ticker_value,):
    
    
    return (html.H5(ticker_value))
        


app2=dash.Dash(__name__,
              external_stylesheets=external_stylesheets,
                server=server,
                url_base_pathname='/dash2/')
app2.config['suppress_callback_exceptions'] = True

app2.layout = html.Div( #START DOCUMENT
        children= [
                html.Div( #START HEADER
                        [
                                html.Div([html.H1("LOGO GOES HERE FOR APP2")],className="three columns"),
                                html.Div([html.H5('MENU AND INTRO INFO')],className="nine columns"),
                                ],
                        className='twelve columns'), #END HEADER
                html.Div( #START BODY
                        [
                                html.Div( #START ROW
                                        [
                                                html.Div([
                                                        html.Div([html.Label("Select Ticker:")],style={'color':'white'}),
                                                       dcc.Dropdown(
                                                                   id = 'ticker',
                                                                   options=[{'label': i, 'value': i} for i in ['A','B','C']],
                                                                    value='C'
                                                                   ),
                                                        html.H5("SELECTIONS"),
                                                        html.H5("DROPDOWNS"),
                                                        html.H5("SLIDER"),
                                                        html.H5("CHECK BOXES"),
                                                        html.H5("RADIO BUTTONS"),
                                                          ],className='three columns',id='pane_0'),
                                                html.Div([
                                                        html.H1("CONTENT PANE 1"),
                                                        ],className='five columns',id='pane_1'),
                                                html.Div([
                                                        html.H1("CONTENT PANE 2"),
                                                        html.Div(id='output_1'),
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

@app2.callback(
    Output('output_1','children'),
    [Input(component_id='ticker', component_property='value'),]
)
def render_content2(ticker_value,):
    
    
    return (html.H5(ticker_value))
        

#if __name__ == '__main__':
#    app.run_server(debug=False,
#                   port=8080)

@server.route('/', methods=['GET', 'POST'])
def index():
    
    page='''
    <html>
    <a href='/dash'> app 1 </a><br>
    <a href='/dash2'> app 2 </a><br>
    </html>
    '''
    return page


@server.route("/dash", methods=['GET', 'POST'])
def my_dash_app():
    return app.index()


@server.route("/dash2", methods=['GET', 'POST'])
def my_dash_app2():
    return app2.index()
    

server.run(debug=False, port=8081)

'''
if __name__ == '__main__':
    app.run_server(debug=False,
                   port=8080)
'''