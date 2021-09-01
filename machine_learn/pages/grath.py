import dash
import pandas as pd
import numpy as np
import pathlib
from utils import Header, get_table

import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.express as px
import pathlib

from dash.dependencies import Input, Output

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve

#correlation plot
df =pd.read_excel('../machine_learn/quality_data_all.xlsx')
dfcor = df.corr()['2 days MPa'].sort_values(ascending=True)
dfcor.drop(labels = '2 days MPa', inplace = True)
dfcord = round( pd.DataFrame(dfcor),2)
dfcord.reset_index(inplace = True)
dfcord.rename(columns = {"index" : "quality parameters", "2 days MPa" : "correlation"}, inplace =True)

#factor analysis

dataset = df[['R 008, %', 'SO₃, %', 'QI1, g/t', 'GA2, g/t',
       'st. clinker,%', 't, cement, ° С', 'moisture,%', 'outdoor temp, ° С',
       'Free_lime,%', 'limestone,%', 'Eq.Na2O,%', 'C3S%', 'C3A%']]
from factor_analyzer import FactorAnalyzer
fa = FactorAnalyzer(n_factors=2,rotation='varimax')
fa.fit(dataset)
with np.printoptions(suppress=True,precision=6):
      factor_df= pd.DataFrame(fa.loadings_,index=dataset.columns)

factor_df.columns = ['prehidration', 'additive']
factor_df.reset_index(inplace=True)
factor_df.rename(columns = {'index':'quality parameters'}, inplace = True)
factor_df= round (factor_df.sort_values(by = 'prehidration', ascending = False))

color_1 = "#003399"
color_2 = "#71C8F3"
color_3 = "#002277"
color_b = "#F8F8FF"

available_indicators = df.columns

description = pd.DataFrame({'Name': ['2 days MPa','R 008, %', 'SO₃, %',
'QI1, g/t', 'GA2, g/t', 'st. clinker,%','t, cement, ° С',  'moisture,%',
'outdoor temp, ° С', 'Free_lime,%','limestone,%', 'Eq.Na2O,%'],

'Description' :['2 days compressive strength', 'sieve residue, cement fineness\
 indicator','SO3 content in cement','quality improver (dosage g/t), additive \
 with the ability to increase 2D compressive strength','cement additive-grinding\
  aid, trialed over several days' ,  'open storage clinker', 'temperature\
    collected after mill outlet', 'cement moisture',  'outdoor air temperature,\
     24 h mean', 'free lime of clinker',  'filler, supplementary material',\
       'alkali content'
]})

def create_layout(app):
    return html.Div([

        Header(app),

html.Div([html.H6('Table, CEM I 42,5, quality parameters and 2 days compressive strength')], className="page-7d",),
        (html.Div([get_table(df,"2 days MPa",df.columns, 'all-table' )])),

html.Div([
        html.Div([html.H6( "A total of 9 quality parameters were presented \
          to test the cement:")]),
        html.Div([get_table (description, 'Name', description.columns, 'description')])
        ]),

html.Div([html.H6('Quality  parameaters, trend lines')], className="page-7d"),



html.Div([

    html.Div([
        dcc.Dropdown(
            id='xaxis-column',
            options=[{'label': i, 'value': i} for i in available_indicators],
            value='SO₃, %'
        ),

    ],
    style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        dcc.Dropdown(
            id='yaxis-column',
            options=[{'label': i, 'value': i} for i in available_indicators],
            value='2 days MPa'
        ),

    ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
], className ="row"),

dcc.Graph(id='indicator-graphic'),

html.Div([

html.Div([html.H6('Correlation plot, 2 days and quality parameters')], className="page-7d"),



html.Div( [dcc.Graph (figure = (px.bar(dfcor, color = 'value'))),


          ]),
], className="six columns"),

html.Div ([

html.Div([html.H6("Peason correlation")], className="page-7d"),

html.Div([dash_table.DataTable( data=dfcord.to_dict(
                                                    "records"
                                                ),
                                                columns=[
                                                    {"id": c, "name": c}
                                                    for c in dfcord.columns
                                                ],
                                                style_data_conditional=[
                                                    {
                                                        "if": {"row_index": "odd"},
                                                        "backgroundColor": "#edf2fa",
                                                    },
                                                    {
                                                        "if": {
                                                            "column_id": "quality parameters"
                                                        },
                                                        "backgroundColor": "#edf2fa",
                                                        "color": "black",'textAlign': 'left'
                                                    },
                                                ],
                                                style_header={
                                                    "backgroundColor": "#edf2fa",
                                                    "fontWeight": "bold",
                                                    "color": "balck",'textAlign': 'left'
                                                },
                                                fixed_rows={"headers": True},
                                                style_cell={"width": "20px"},

                                                ),
                                            ] ),


          ], className = "six columns",  style={"padding": "5px"}),

html.Div ([

html.Div([html.H6("Factor analysis, rounded")], className="page-7d"),

html.Div([dash_table.DataTable( data= factor_df.to_dict(
                                                    "records"
                                                ),
                                                columns=[
                                                    {"id": c, "name": c}
                                                    for c in factor_df.columns
                                                ],
                                                style_data_conditional=[
                                                    {
                                                        "if": {"row_index": "odd"},
                                                        "backgroundColor": "#edf2fa",
                                                    },
                                                    {
                                                        "if": {
                                                            "column_id": "quality parameters"
                                                        },
                                                        "backgroundColor": "#edf2fa",
                                                        "color": "black",'textAlign': 'center'
                                                    },
                                                ],
                                                style_header={
                                                    "backgroundColor": "#edf2fa",
                                                    "fontWeight": "bold",
                                                    "color": "balck",'textAlign': 'center'
                                                },
                                                fixed_rows={"headers": True},
                                                style_cell={"width": "20px"},

                                                ),
                                            ] ),


          ], className = "six columns",  style={"padding": "5px"}),

html.Div ([

html.Div([html.H6("Description")], className="page-7d"),
html.Div([html.H6("Factor analysis is used to describe variability among\
 correlated variables. In this dataset, 3 correlated variables were present:\
  outdoor temperature, the temperature of cement, and cement moisture. All 3 \
  parameters were an indication of cement prehydration. These parameters can be\
   replaced by the LOI (loss on ignition) parameter, and this could likely\
    improve prediction. 1 -st parameter is SO3. 2 -nd important parameter factor is the additive used.")], className = "page-7d"),
          ], className = "six columns",  style={"padding": "5px"}),


],className= 'page_l')
