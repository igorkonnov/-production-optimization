#71C8F3import dash
import pathlib
from utils import Header
from dash import  dcc
from dash import html
from dash import  dash_table
import plotly.express as px
from dash.dependencies import Input, Output
from utils import get_table, model_list
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


import pandas as pd
import numpy as np
import plotly.express as px

color_1 = "#003399"
color_2 = "#71C8F3"
color_3 = "#002277"
color_b = "#F8F8FF"


PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()
cost = pd.read_csv('../machine_learn/cost.csv',  sep = ';')
output = pd.read_csv('../machine_learn/output.csv', sep = ';')
df =pd.read_excel('../machine_learn/quality_data_all.xlsx')
output_cost = pd.read_excel('../machine_learn/output_cost.xlsx')


Z1 =df[['R 008, %', 'SO₃, %', 'QI1, g/t',
       'GA2, g/t', 'st. clinker,%', 't, cement, ° С', 'moisture,%',
       'outdoor temp, ° С', 'Free_lime,%', 'limestone,%', 'Eq.Na2O,%', 'C3S%',
       'C3A%']]




def create_layout(app):
    return html.Div([

        Header(app),
        html.Div([html.H6("Input table, material cost and mill parameters, in $\
         value. 'All parameters are adjustable and the graphs are interactive.")],  className="page-7d"),

html.Div([ dash_table.DataTable(
                 id='table-editing-cost',  data= cost.to_dict('records'),
                 columns=[{'id': c, 'name': c} for c in cost.columns],
                 editable=True,
                 style_header={
                         "backgroundColor": color_1,
                         "fontWeight": "bold",
                         "color": "white",'textAlign': 'center', "fontSize": "8pt"
                             },
                 fixed_rows={"headers": True},
                 style_cell={"width": "60px", "fontSize": "11pt"}
                             )
         ]),

html.Div([dcc.Loading(id = "loading-2", children = [html.Div([ dcc.Graph(id='cost_prediction2')])])], ),


html.Div([html.H6("Input table. Cement quality parameters")]),
         html.Div([ dash_table.DataTable(
                 id='table-quality',  data= df.tail(1).to_dict('records'),
                 columns=[{'id': p, 'name': p} for p in Z1.columns],

                 editable=True,

                 style_header={
                         "backgroundColor": color_1,
                         "fontWeight": "bold",
                         "color": "white",'textAlign': 'center', "fontSize": "8pt"
                     },
                 fixed_rows={"headers": True},
                 style_cell={"width": "60px", "fontSize": "11pt"}
             )
         ]),
html.Div([html.H6("For any combination of raw materials, cement quality,\
 and material costs the  ML model can find a unique lowest-cost composition of\
  cement. This algorithm uses multiple machine learning models, depends\
   on predictors and values can switch between several models.")], className = "page-7d"),

html.Div([html.H6('Output table, predicted 2 days Mpa and cement composition cost, $/t')]),



html.Div([ dash_table.DataTable(
        id='table-outputcost',  data= output_cost.to_dict('records'),
        columns=[{'id': p, 'name': p} for p in output_cost.columns],
        editable=True,

       style_header={
              "backgroundColor": color_1,
              "fontWeight": "bold",
               "color": "white",'textAlign': 'center', "fontSize": "8pt" },     fixed_rows={"headers":True },
      style_cell={"width": "60px", 'font-size' : '11pt'}





     )
 ],),



html.Div([
html.Div([ dcc.Dropdown(id='rounding', options=[{'label': i, 'value': i} for i in range(0,4) ],   placeholder="Precision"),

     ],style={'width': '30%', 'float': 'left', 'padding-bottom' : '30 px'}, ),


        ], className = 'row'),


 html.Div([dcc.Loading(id = "loading-1", children = [html.Div([ dcc.Graph(id='cost_vs_quality')])],
 )],  ),


 #



 ], className= 'page_l')
