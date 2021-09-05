import dash
from dash import dash_table
from dash import  dcc
from dash import html
import plotly.graph_objs as go
from dash import dash_table
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

df =pd.read_excel('../machine_learn/quality_data_all.xlsx')

from dash.dependencies import Input, Output,  State

from utils import smoothTriangle, model_list

color_1 = "#003399"
color_2 = "#71C8F3"
color_3 = "#002277"
color_b = "#F8F8FF"

from pages import (overv, grath, calc, optim)

Z1 =df[['R 008, %', 'SO₃, %', 'QI1, g/t',
       'GA2, g/t', 'st. clinker,%', 't, cement, ° С', 'moisture,%',
       'outdoor temp, ° С', 'Free_lime,%', 'limestone,%', 'Eq.Na2O,%', 'C3S%',
       'C3A%']]

Y = df['2 days MPa']
#
Ridgem =Ridge(alpha=0.001,fit_intercept = True,  normalize=True )

HistGradientBoosting = HistGradientBoostingRegressor(learning_rate = 1, max_iter = 10, max_bins = 10, min_samples_leaf = 20 )


x_train, x_test, y_train, y_test = train_test_split(Z1, Y, test_size=15, random_state=42)
Ridgem.fit(x_train, y_train)

HistGradientBoosting.fit(x_train, y_train)

prediction = model_list( Z1, Y)
# mill output prediction
output = pd.read_csv('../machine_learn/output.csv', sep = ';')
Inputpoly=[ ('polynomial', PolynomialFeatures(include_bias=False, degree =2)), ('model',LinearRegression(normalize=True))]
Polinomalreg =Pipeline(Inputpoly)

Polinomalreg.fit( output[['R 008. %', 'QI1. g/t']] , output['output'])
##############################################################################

app = dash.Dash(
    __name__, suppress_callback_exceptions=True, meta_tags=[{"name": "viewport", "content": "width=device-width"}])
server = app.server


app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)




@app.callback(Output("page-content", "children"), [Input("url", "pathname")] )



def display_page(pathname):
    if pathname == "/machine_learn/overv":
        return overv.create_layout(app)
    elif pathname == "/machine_learn/grath":
        return grath.create_layout(app)
    elif pathname == "/machine_learn/calc":
        return calc.create_layout(app)

    elif pathname == "/machine_learn/optim":
        return optim.create_layout(app)


    else:
        return overv.create_layout(app)



@app.callback(
    Output('indicator-graphic', 'figure'),
    Input('xaxis-column', 'value'),
    Input('yaxis-column', 'value'))

def update_graph(xaxis_column_name, yaxis_column_name):

    fig = px.scatter(df,x= xaxis_column_name, y=yaxis_column_name, trendline = "lowess", color = yaxis_column_name )

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    fig.update_xaxes(title=xaxis_column_name)

    fig.update_yaxes(title=yaxis_column_name)

    return fig



@app.callback(
    Output('mmodel', 'figure'),
    Input('degree1', 'value'),
    Input('model_select', 'value'))

def update2 (degr, model_select):

    fig = px.scatter(df, x= df.index, y = Y,    width=1000, height=700, title ="Models and trendline predictions")
    fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01))


    fig.add_trace(go.Scatter(
    x= df.index,
    y=smoothTriangle(Y, degr),
    mode='markers',
    marker=dict(
        size=2,
       color='blue',
        symbol='0'
      ),
        name='trendline real 2D MPa '
      ), secondary_y= False)
    modeldata  = ['Ridge regression',  'HistGradientBoosting regression', 'Linear regression', 'Lasso', 'Polinomal regression' ]
    u = modeldata.index (model_select)
    fig.add_trace(go.Scatter(
    x= df.index,
    y=prediction[u].predict(Z1),
    mode='markers',
    marker=dict(
        size=4,
       color='red',
        symbol='0'
      ),
        name='predicted 2D MPa'
      ), secondary_y= False)


    fig.add_trace(go.Scatter(
    x= df.index,

    y = smoothTriangle(prediction[u].predict(Z1), degr),  # setting degree
    mode='lines+markers',
    marker=dict(
            size=2,
           color='red',
            symbol='0'
          ),
            name='trendline predicted 2D MPa'
          ), secondary_y= False)


    return fig

global k

@app.callback(

    Output ('ridge', 'children'),
    Output ('booster', 'children'),
    Output ('linear', 'children'),
    Output ('lasso', 'children'),
    Output ('polinonal', 'children'),
    Output('danger', 'children'),
    Input('table-editing-simple', 'data'),
    Input('table-editing-simple', 'columns'))

def display_output(rows, columns):

    model_demonstration = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    model_demonstration = model_demonstration.astype('float')

    if  model_demonstration['R 008, %'][0] >4 or  model_demonstration['SO₃, %'][0] >4\
    or model_demonstration['QI1, g/t'][0] >1000 or  model_demonstration['GA2, g/t'][0] >500\
    or model_demonstration['st. clinker,%'][0] >100 or model_demonstration['t, cement, ° С'][0] > 130\
    or model_demonstration['moisture,%'][0] > 1 or model_demonstration['outdoor temp, ° С'][0]> 40  or model_demonstration['Free_lime,%'][0]>2 :
     #or model_demonstration['limestone,%'] > 10 :
     danger = 'a value is out of range'

    elif  model_demonstration['QI1, g/t'][0]  and model_demonstration['GA2, g/t'][0] != 0 :
        danger = 'not possible to use 2 adds, prediction is incorrect'
    elif  model_demonstration['limestone,%'][0] > 5 :
        danger ="limestone above 5% is restricted by the standard"


    else : danger =''

    ridge = round(model_list(Z1, Y)[0].predict(model_demonstration).item(0),3)
    booster = round(model_list(Z1, Y)[1].predict(model_demonstration).item(0),3)
    linear = round(model_list(Z1, Y)[2].predict(model_demonstration).item(0),3)
    lasso = round(model_list(Z1, Y)[3].predict(model_demonstration).item(0),3)
    polinomal = round(model_list(Z1, Y)[4].predict(model_demonstration).item(0),3)

    return ridge, booster, linear, lasso, polinomal, danger




@app.callback(

    Output ('table-outputcost' , 'data'),
    Output ('table-outputcost' , 'columns'),
    Output('cost_prediction2', 'figure'),

    Input('table-quality', 'data'),
    Input('table-quality', 'columns'),
    Input('table-editing-cost', 'data'),
    Input('table-editing-cost', 'columns'))
def display_output4(rows, columns, data1, columns1):

    cem_param = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    cost_input = pd.DataFrame(data1, columns = [l['name'] for l in columns1])

    output_predicted = round(Polinomalreg.predict(cem_param[['R 008, %', 'QI1, g/t']]).item(0), 1)

    strength_ridge = round(Ridgem.predict(cem_param).item(0),2)
    strength_booster = round( HistGradientBoosting.predict(cem_param).item(0),2)
    cem =cem_param.loc[0].to_numpy(dtype= 'float')
    costs = cost_input.loc[0].to_numpy(dtype = 'float')


    if cem.item(0) < 2 or cem.item(2) > 300 or cem.item(2) <100 or cem.item(2) < 1 or cem.item(2) > 3 :
        strength_ridge1 = HistGradientBoosting.predict(cem_param).item(0)
    else:
        strength_ridge1 = Ridgem.predict(cem_param).item(0)


    totpower = costs.item(6)+costs.item(7)
    cost_energypt = round( totpower/output_predicted*costs.item(1), 2)
    rmcost = round(costs.item(4)*58/output_predicted,1)
    add_cost = costs.item(5)/1000000*cem.item(2)
    composition_cost = round((100-cem.item(9)-cem.item(1))*1.25 *(costs.item(0)/100)+ (cem.item(9) * costs.item(2)/100)+(cem.item(1)*1.25 * costs.item(3)/100),1)
    total_cost = round( cost_energypt+add_cost+composition_cost+rmcost, 2)
    output_cost = pd.DataFrame({'Ridge,prediction': [strength_ridge],
    'G.boos.,prediction' : [strength_booster], 'Mill output,t/h': [output_predicted],
    'Electricity $/t': [cost_energypt], 'R&M cost': [rmcost] , 'Additive t.cost $/t ': [add_cost], 'Composition cost $/t':[ composition_cost], 'Total var.cost, $/t': [total_cost]})

    data= output_cost.to_dict( "records")
    columns=[{"id": c, "name": c} for c in  output_cost.columns]

    fig = go.Figure()
    df_optim = pd.read_csv('../machine_learn/costquality.csv')
    fig.add_scatter( x = df_optim['quality'], y = df_optim['cost'], opacity= 1,
                    mode = 'markers', marker = dict(size =4, color = '#2C21B3'), name = 'predicted cost and quality, all possible values' )

    fig.add_scatter(y=[total_cost], x =[strength_ridge1], mode="markers",opacity= 0.8,
                    marker=dict(size=30, color="#FFFF00", symbol='circle-dot', line_width=2),
                    name="actual cost and quality")

    fig.add_scatter(y=[total_cost], x =[strength_ridge1], mode="markers",opacity= 0.5,
                    marker=dict(size=100, color="white", symbol='circle-dot', line_width=2, line_color ='blue'),
                    name="actual cost and quality")

    fig.update_layout( width=1000, height=600,legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01))

    fig.update_layout(title_text="All posssible values, sum of variable costs and quality, 2 dayd MPa ")
    fig.update_yaxes(title_text="<b>Varible cement composition cost, $</b>")
    fig.update_xaxes(title_text="<b>Quality, 2d MPa</b> ")




    return data, columns, fig


dfnum = np.empty([1,2])
@app.callback(
    Output('cost_vs_quality', 'figure'),
    Input('rounding', 'value'),
    Input('table-editing-cost', 'data'),
    Input('table-editing-cost', 'columns'),
    Input('table-quality', 'data'),
    Input('table-quality', 'columns'),

    )

def dispgrath(rounding, rows, columns, rows1, columns1):

    cost_input = pd.DataFrame(rows, columns = [k['name'] for k in columns])
    cem_param = pd.DataFrame(rows1, columns=[c['name'] for c in columns1])
    cem =cem_param.loc[0].to_numpy(dtype= 'float')


    global dfnum
    for res in range (0 , 9 ):
                res = res/4
                for dos in range (0, 1000, 20):
                    for so3 in range (1,12):
                        so3=so3/3
                        for limestone in range (0,5):
                            cem_param =[[ res, so3, dos, 0, cem.item(4), cem.item(5), cem.item(6),cem.item(7), cem.item(8), limestone, cem.item(10), cem.item(11), cem.item(12)]]
                            output_predicted = round(Polinomalreg.predict([[res, dos]]).item(0), 1)
                            if res < 2 or dos > 300 or dos <100 or so3 < 1 or so3 > 2 :
                                strength_ridge = HistGradientBoosting.predict(cem_param).item(0)
                            else:
                                strength_ridge = Ridgem.predict(cem_param).item(0)


                            cem =np.asarray(cem_param)
                            costs = cost_input.loc[0].to_numpy(dtype = 'float')
                            totpower = costs.item(6)+costs.item(7)
                            cost_energypt = round( totpower/output_predicted*costs.item(1), 2)
                            rmcost = round(costs.item(4)*58/output_predicted,1)
                            add_cost = round (costs.item(5)/1000000*cem.item(2),1)
                            composition_cost = (100-cem.item(9)-cem.item(1))*1.25 *(costs.item(0)/100)+ (cem.item(9) * costs.item(2)/100)+(cem.item(1)*1.25 * costs.item(3)/100)
                            total_cost = ( cost_energypt+add_cost+composition_cost+rmcost)
                            rawcost = np.array([[strength_ridge, total_cost]])
                            dfnum=np.concatenate((dfnum, rawcost), axis = 0)
    dfnum= np.delete(dfnum, 0, 0)

    df_optim=pd.DataFrame(dfnum)

    df_optim.columns = ['quality', 'cost']


    df_optim.to_csv('../machine_learn/costquality.csv')

    opt = df_optim

    opt['qualityr'] = round(opt['quality'], rounding)

    opt = opt [[ 'cost', 'qualityr']]

    opt['cost1'] = opt['cost']

    opt1 = opt.groupby([ 'qualityr', 'cost' ]).min()

    mincost = opt1.groupby(level=0).min()

    mincost.reset_index(inplace =True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_scatter( y = mincost['qualityr'], x = mincost.qualityr,
               mode = 'markers+lines', marker = dict(size =2, color = 'red'), name = '2 days, MPa ')

    fig.add_scatter(y = mincost['cost1'], x = mincost.qualityr,  mode="markers+lines",
                marker=dict(size=2, color="blue"),
                name="cement varible costs", secondary_y= True)
    fig.update_xaxes(range=[15, 27.5])
    fig.update_layout(title_text="Cement quality vs cost, after optimization")
    fig.update_yaxes(title_text="<b>Quality, 2d MPa</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Cost, $</b> ", secondary_y=True)
    fig.update_xaxes(title_text="<b>Quality, 2d MPa</b> ")
    nob = rounding



    dfnum = np.array([[0,0]])
    return fig







if __name__ == "__main__":
    app.run_server(port=8000,debug=True, dev_tools_ui= True,dev_tools_props_check= True)
