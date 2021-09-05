from dash import  html
from dash import  dcc
from dash import dash_table
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

color_1 = "#003399"
color_2 = "#71C8F3"
color_3 = "#002277"
color_b = "#F8F8FF"

df =pd.read_excel('../machine_learn/quality_data_all.xlsx')

Z1 =df[['R 008, %', 'SO₃, %', 'QI1, g/t',
       'GA2, g/t', 'st. clinker,%', 't, cement, ° С', 'moisture,%',
       'outdoor temp, ° С', 'Free_lime,%', 'limestone,%', 'Eq.Na2O,%', 'C3S%',
       'C3A%']]

Y = df['2 days MPa']

def Header(app):
    return html.Div([html.Br([]), get_menu(), html.Hr([])])

def get_menu():
    menu = html.Div(
        [
            dcc.Link(
                "Overview",
                href="/machine_learn/overv",
                className="tab first",
            ),
            dcc.Link(
                "Statistics and quality predictors",
                href="/machine_learn/grath",
                className="tab",
            ),
            dcc.Link(
                "Machine learning models",
                href="/machine_learn/calc",
                className="tab",
            ),

            dcc.Link(
                "Cement composition cost optimization by ML",
                href="/machine_learn/optim",
                className="tab",
            ),




        ],
        className="row all-tabs",
    )
    return menu

def get_table (df, column, column_list , id):
    table = html.Div(
    [dash_table.DataTable(id = id, data= df.to_dict( "records"),
                                                        columns=[
                                                            {"id": c, "name": c, "selectable": True}
                                                            for c in  column_list
                                                        ],
                                                        filter_action="native",
                                                        sort_action="native",


                                                        style_data_conditional=[
                                                            {
                                                                "if": {"row_index": "odd"},
                                                                "backgroundColor": color_b,'textAlign': 'center'
                                                            },
                                                            {
                                                                "if": {
                                                                    "column_id": column
                                                                },
                                                                "backgroundColor": color_2,
                                                                "color": "black",'textAlign': 'center'
                                                            },
                                                        ],
                                                        style_header={
                                                            "backgroundColor": color_1,
                                                            "fontWeight": "bold",
                                                            "color": "white",'textAlign': 'center'
                                                        },
                                                        fixed_rows={"headers": True},
                                                        style_cell={"width": "70px", "fontSize": "8pt", 'textAlign': 'center'},

                                                        ),
                                                    ])
    return table


def smoothTriangle(data, degree):
    triangle=np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1]))
    smoothed=[]

    for i in range(degree, len(data) - degree * 2):
        point=data[i:i + len(triangle)] * triangle
        smoothed.append(np.sum(point)/np.sum(triangle))
    # Handle boundaries
    smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed


#######################################
def model_list( Z1, Y):
    Ridgem =Ridge(alpha=0.001,fit_intercept = True,  normalize=True )
    Linear_regression =LinearRegression (normalize = True )
    HistGradientBoosting = HistGradientBoostingRegressor(learning_rate=0.1, max_iter = 100, max_leaf_nodes = 40, min_samples_leaf = 20 )
    Lassom = Lasso(alpha = 0.001 , normalize=True )
    InputPoly=[ ('polynomial', PolynomialFeatures(include_bias=False, degree =2)), ('model',LinearRegression(normalize=True))]
    Polinomalreg =Pipeline(InputPoly)
    x_train, x_test, y_train, y_test = train_test_split(Z1, Y, test_size=15, random_state=42)
    Ridgem.fit(x_train, y_train)
    Linear_regression.fit( x_train , y_train )
    HistGradientBoosting.fit(x_train, y_train)
    Lassom.fit( x_train , y_train )
    Polinomalreg.fit( x_train , y_train )

    modeldata = [Ridgem,  HistGradientBoosting, Linear_regression, Lassom, Polinomalreg ]
    return modeldata

###############################################################################
