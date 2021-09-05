import dash
import pathlib
from utils import Header
from dash import dcc
from dash import  html
from dash import dash_table
import plotly.express as px
import pathlib
from dash.dependencies import Input, Output
from utils import get_table, model_list


import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error

color_1 = "#003399"
color_2 = "#71C8F3"
color_3 = "#002277"
color_b = "#F8F8FF"

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()

df =pd.read_excel('../machine_learn/quality_data_all.xlsx')
############################################################################
Z1 =df[['R 008, %', 'SO₃, %', 'QI1, g/t',
       'GA2, g/t', 'st. clinker,%', 't, cement, ° С', 'moisture,%',
       'outdoor temp, ° С', 'Free_lime,%', 'limestone,%', 'Eq.Na2O,%', 'C3S%',
       'C3A%']]


Y = df['2 days MPa']

modeldataframe = pd.DataFrame({'model': []})



model_list( Z1, Y)

modeldataframe['model'] = model_list(Z1, Y)

x_train, x_test, y_train, y_test = train_test_split(Z1, Y, test_size=15, random_state=42)

def score1(x):
    return round(x.score(x_test, y_test),2)

def mse(x):
    return round( mean_squared_error(x.predict(x_test),y_test),2)

def maxe(x):
    return  round(max_error(x.predict(x_test),y_test),2),

def mabse(x):
    return  round(mean_absolute_error(x.predict(x_test),y_test),2)

from sklearn.model_selection import cross_val_score

def cvs (x):
    return round(cross_val_score ( x , Z1 , Y, cv =5).mean(),2)

modeldataframe['r_squared'] = modeldataframe['model'].apply(score1)
modeldataframe['mean_squared_error'] = modeldataframe['model'].apply(mse)
modeldataframe['max error'] = modeldataframe['model'].apply(maxe)
modeldataframe['mean_absolute_error'] = modeldataframe['model'].apply(mabse)
modeldataframe['cross-validation score'] = modeldataframe['model'].apply(cvs)


modeldata1 = modeldataframe
modeldata1['model']= ['Ridge regression',  'HistGradientBoosting regression', 'Linear regression', 'Lasso', 'Polinomal regression' ]
############################################################################
zc = list(Z1.columns)
pimportance = pd.DataFrame({'importances':[],'importances_mean': [], 'importances_std': []})
from sklearn.inspection import permutation_importance
Ridgem = model_list(Z1, Y)[0]
r = permutation_importance(Ridgem, x_test, y_test, n_repeats=20, random_state=2)
for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 0.5 * r.importances_std[i] > 0:
         name = zc[i]
         imp_mean= round( r.importances_mean[i],3)
         imp_std =round(r.importances_std[i],3)
         pimportance =pimportance.append ({'importances':name, 'importances_mean': imp_mean,'importances_std':imp_std},ignore_index=True)

###########################################################################3#


def create_layout(app):
    return html.Div([

        Header(app),
        html.Div([html.H6("Machine learning models and metrics")],  className="page-7d"),
        html.Div ([
        html.Div([html.H6 ("Several regression regression ML models were tested:"),
                  html.H6 ('Ridge, Multilinear, GradientBoosting, Lasso, MultiPolinomal')]),


         ], className="page-7d"),
         html.Div([html.H6("*Change the values in cells to see ML prediction")],),
         html.Div([
         html.Div([ dash_table.DataTable(
                 id='table-editing-simple',  data= df.head(1).to_dict('records'),
                 columns=[{'id': p, 'name': p} for p in zc],

                 editable=True,

                 style_header={
                         "backgroundColor": color_1,
                         "fontWeight": "bold",
                         "color": "white",'textAlign': 'center'
                     },
                 fixed_rows={"headers": True},
                 style_cell={"width": "60px"}





             )
         ]),

                 ]),

        html.Div([html.Output(id='danger', style={'width': '20%', 'height': 8,'font-size':15, 'margin-bottom':0, 'color': 'red' })],className = "no-page"),
        html.Div([
                          html.Label("Ridge", className = 'output4' ),
                          html.Label("G.Booster", className ="output4"),
                          html.Label("M.Linear", className = "output4"),
                          html.Label("Lasso", className = "output4" ),
                          html.Output("M.Polinomal", className ="output4"),
                        ], className = 'no-page' ),

        html.Div([
                  html.Output(id="ridge",title = "Ridge" , className ='output3' ),

                  html.Output(id = 'booster', className = 'output3'),
                  html.Output(id = 'linear', className = 'output3'),
                  html.Output(id = 'lasso', className = 'output3'),
                  html.Output(id = 'polinonal', className = 'output3'),
                ], className = 'no-page' ),
        html.Div([html.H6('Models performance')], className = "page-7d"),
        get_table(modeldata1, 'model', modeldata1.columns, 'id'),

        html.Div ([
        html.Div ([html.H6 ('Selected models : Ridge regression, Gradient boosting'),
                 html.H6 ("Gradient boosting is better for non-linear processes and extreme values (e.g., SO3 above 3.5%)"),
                 html.H6 ("With Ridge regression was possible to predict 73% of 2d compressive strength data(average validation score)"),
                  html.H6 ("Using Gradient Boosting, it was possible to predict 69% of the 2D Mpa data.")])
                 ],  className = "page-7d"
                  ),

###############################################################################
html.Div([
html.Div([ html.Div([
        dcc.Dropdown(
            id='model_select',
            options=[{'label': i, 'value': i} for i in modeldata1['model'] ],
            value= 'Ridge regression'
        ),

    ],
    style={'width': '48%', 'display': 'inline-block'}),]),
       html.Div([dcc.Slider(
        id='degree1',
        min=10,
        max=50,
        step=1,
        value=20,
    ),],style={'width': '48%', 'display': 'inline-block'} ),

html.Div([ dcc.Graph(id='mmodel')]),




    ]),
###############################################################################
html.Div([html.H6("This table is built on ridge regression and demonstrates which parameter contributes most to the quality as well as what is most important to control in terms of SO3 and the dosage of the quality improver. ")],className = "page-7d" ),
get_table (pimportance, 'importances', pimportance.columns, 'import_name'),
html.Div ([dcc.Graph (figure = (px.bar(pimportance,  x = 'importances', y = ['importances_mean', 'importances_std'],  barmode='group')))])




                    ],className= 'page_l')
